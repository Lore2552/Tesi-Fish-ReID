"""
Riferimento supervisionato: ArcFace (s = 64, m = 0,5) sullo split fisso.
Classi = coppie (pesce, lato) di train; backbone ViT-S/14 (--init dinov2 | zero), embedding = token di classe.
Un ritaglio per foto (scala 0,6-1); AdamW, lr backbone 5e-5 (5e-4 da zero),
lr classi 1e-3. Checkpoint scelto sulla validation, test una volta alla fine.

Opzioni principali:
  --classi catture   solo i (pesce, lato) fotografati in almeno due date
  --prototipi        pesi delle classi inizializzati con la media degli embedding delle loro foto (weight imprinting)
  --epoche_testa N   prime N epoche con il backbone congelato
  --mescola          etichette permutate tra le foto

    python arcface.py --init dinov2 --epochs 60 --ogni 4 --classi catture --prototipi --checkpointing
    output: runs/arcface_<init>[_catture][_prototipi][_<etichetta>]/
"""

import os
import csv
import json
import math
import time
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import build_records
from split_fisso import carica_split
from reid_locale import sveglio
from lejepa_multivista import FotoCache, Viste, lr_al_passo, valuta_backbone, LATO_GLOBALE
from riferimenti_globali import embedding, foto


class ArcFace(nn.Module):
    def __init__(self, d, n_classi, s=64.0, m=0.5):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_classi, d) * 0.01)
        self.s, self.m = s, m

    def forward(self, e, y):
        cos = F.normalize(e, dim=-1) @ F.normalize(self.w, dim=-1).T
        ang = torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6))
        giusto = F.one_hot(y, cos.shape[1]).bool()
        logit = torch.where(giusto, torch.cos(ang + self.m), cos) * self.s
        return F.cross_entropy(logit, y)


def main():
    ap = argparse.ArgumentParser(description="ArcFace sullo split fisso")
    ap.add_argument("--init", choices=["dinov2", "zero"], required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--ogni", type=int, default=2)
    ap.add_argument("--passi_prova", type=int, default=0)
    ap.add_argument("--checkpointing", action="store_true", help="gradient checkpointing del backbone")
    ap.add_argument("--classi", choices=["tutte", "catture"], default="tutte",
                    help="tutte: ogni (pesce, lato) di train; catture: solo quelli fotografati in almeno due date")
    ap.add_argument("--prototipi", action="store_true",
                    help="pesi delle classi inizializzati con la media degli embedding iniziali delle loro foto")
    ap.add_argument("--s", type=float, default=64.0, help="scala di ArcFace")
    ap.add_argument("--m", type=float, default=0.5, help="margine di ArcFace (radianti)")
    ap.add_argument("--epoche_margine", type=int, default=0,
                    help="il margine sale linearmente da 0 a --m in queste epoche (0 = margine pieno da subito)")
    ap.add_argument("--inizio_margine", type=int, default=0,
                    help="epoche iniziali con margine 0 (softmax normalizzata) prima che il margine cominci a salire")
    ap.add_argument("--lr_backbone", type=float, default=None, help="default: 5e-5 da DINOv2, 5e-4 da zero")
    ap.add_argument("--clip", type=float, default=0.0, help="norma massima del gradiente (0 = nessun limite)")
    ap.add_argument("--seme", type=int, default=0, help="seme del training (ordine dei batch, ritagli, pesi)")
    ap.add_argument("--etichetta", default="", help="suffisso della cartella di output")
    ap.add_argument("--epoche_testa", type=int, default=0,
                    help="LP-FT: nelle prime N epoche il backbone e' congelato e si allena solo la testa")
    ap.add_argument("--mescola", action="store_true",
                    help="etichette permutate a caso tra le foto (le classi non sono piu' pesci)")
    ap.add_argument("--da_backbone", default="",
                    help="checkpoint (chiave 'backbone') da cui partire, es. runs/lejepa_multivista_zero/migliore.pth; con --init zero")
    args = ap.parse_args()
    sveglio()
    torch.manual_seed(args.seme)
    device = torch.device("cuda")
    out_dir = os.path.join("runs", f"arcface_{args.init}" + ("_catture" if args.classi == "catture" else "")
                           + ("_prototipi" if args.prototipi else "") + (f"_{args.etichetta}" if args.etichetta else ""))
    os.makedirs(out_dir, exist_ok=True)
    import timm
    import kornia.augmentation as K

    insiemi = carica_split(build_records("head"))
    tr = insiemi["train"]
    if args.classi == "catture":
        date = defaultdict(set)
        for r in tr:
            date[(r["fish_id"], r["side"])].add(r["date"])
        tr = [r for r in tr if len(date[(r["fish_id"], r["side"])]) >= 2]
    classi = {c: i for i, c in enumerate(sorted({(r["fish_id"], r["side"]) for r in tr}))}
    etichette = torch.tensor([classi[(r["fish_id"], r["side"])] for r in tr])
    if args.mescola:
        etichette = etichette[torch.randperm(len(etichette), generator=torch.Generator().manual_seed(123))]
        print("controllo di fuga: etichette mescolate tra le foto", flush=True)
    loader = DataLoader(FotoCache(tr), batch_size=args.batch, shuffle=True, drop_last=True, num_workers=0, pin_memory=True)
    viste = Viste().to(device)
    ritaglio = K.RandomResizedCrop((LATO_GLOBALE, LATO_GLOBALE), scale=(0.6, 1.0)).to(device)
    fotometriche = viste.globale[1:]
    backbone = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=args.init == "dinov2", img_size=LATO_GLOBALE,
                                 num_classes=0).to(device)
    if args.da_backbone:
        backbone.load_state_dict(torch.load(args.da_backbone, map_location="cpu", weights_only=False)["backbone"])
        print(f"backbone caricato da {args.da_backbone}", flush=True)
    if args.checkpointing:
        backbone.set_grad_checkpointing(True)
    testa = ArcFace(384, len(classi), s=args.s, m=args.m).to(device)
    if args.prototipi:
        backbone.eval()
        e = torch.from_numpy(embedding(lambda x: backbone(x), foto(tr, LATO_GLOBALE), device)).float()
        backbone.train()
        somma_classe = torch.zeros(len(classi), e.shape[1]).index_add_(0, etichette, e)
        testa.w.data.copy_(F.normalize(somma_classe, dim=-1).to(device))
    lr_b = args.lr_backbone or (5e-5 if args.init == "dinov2" else 5e-4)
    opt = torch.optim.AdamW([{"params": backbone.parameters(), "lr": lr_b, "base": lr_b},
                             {"params": testa.parameters(), "lr": 1e-3, "base": 1e-3}], weight_decay=0.05)
    totale = args.epochs * len(loader)
    warmup = max(1, round(0.05 * totale))
    print(f"{len(classi)} classi (pesce, lato), {len(tr)} foto, {len(loader)} passi per epoca", flush=True)
    log = os.path.join(out_dir, "train_log.csv")
    if not args.passi_prova:
        with open(log, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["epoca", "loss", "lr", "val_rank1", "val_rank5", "val_mAP", "minuti"])
    t0, passo, migliore = time.time(), 0, -1.0
    backbone.train()
    for epoca in range(1, args.epochs + 1):
        somma, n = 0.0, 0
        for u8, idx in loader:
            congelato = epoca <= args.epoche_testa
            # backbone fermo nelle prime epoche, poi warmup e coseno propri
            passo_b, totale_b = passo - args.epoche_testa * len(loader), totale - args.epoche_testa * len(loader)
            opt.param_groups[0]["lr"] = 0.0 if congelato else lr_al_passo(passo_b, totale_b, max(1, round(0.05 * totale_b)),
                                                                           opt.param_groups[0]["base"])
            opt.param_groups[1]["lr"] = lr_al_passo(passo, totale, warmup, opt.param_groups[1]["base"])
            if args.epoche_margine or args.inizio_margine:
                salita = (passo - args.inizio_margine * len(loader)) / (max(1, args.epoche_margine) * len(loader))
                testa.m = args.m * min(1.0, max(0.0, salita))
            with torch.no_grad():
                x = fotometriche(ritaglio(u8.to(device, non_blocking=True).float() / 255)).clamp(0, 1)
                x = (x - viste.mean) / viste.std
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16), torch.set_grad_enabled(not congelato):
                e = backbone(x)
            loss = testa(e.float(), etichette[idx].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.clip:
                torch.nn.utils.clip_grad_norm_(list(backbone.parameters()) + list(testa.parameters()), args.clip)
            opt.step()
            somma, n, passo = somma + loss.item(), n + 1, passo + 1
            if args.passi_prova and passo >= args.passi_prova:
                print(f"[prova] {passo} passi | loss {loss.item():.3f} | {(time.time() - t0) / passo:.2f} s/passo | "
                      f"memoria max {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
                return
        riga = [epoca, f"{somma / n:.4f}", f"{opt.param_groups[0]['lr']:.2e}", "", "", "", f"{(time.time() - t0) / 60:.1f}"]
        testo = f"[epoca {epoca}/{args.epochs}] loss {somma / n:.3f} | {(time.time() - t0) / 60:.0f} min"
        if epoca % args.ogni == 0 or epoca == args.epochs:
            _, m, _ = valuta_backbone(backbone, insiemi["validation"], device)
            riga[3:6] = [m["rank-1"][0], m["rank-5"][0], m["mAP"][0]]
            testo += f" | validation rank-1 {m['rank-1'][0]:.2f} mAP {m['mAP'][0]:.2f}"
            if m["mAP"][0] > migliore:
                migliore = m["mAP"][0]
                torch.save({"backbone": backbone.state_dict(), "epoca": epoca, "val": m, "args": vars(args)},
                           os.path.join(out_dir, "migliore.pth"))
        print(testo, flush=True)
        with open(log, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(riga)

    ck = torch.load(os.path.join(out_dir, "migliore.pth"), map_location="cpu", weights_only=False)
    backbone.load_state_dict(ck["backbone"])
    _, m_test, pf = valuta_backbone(backbone, insiemi["test"], device)
    ris = {"init": args.init, "epoca scelta (validation)": ck["epoca"], "validation": ck["val"], "test": m_test, "test per fascia": pf}
    with open(os.path.join(out_dir, "risultati.json"), "w", encoding="utf-8") as f:
        json.dump(ris, f, indent=1, ensure_ascii=False, default=float)
    print(f"FINE: epoca scelta {ck['epoca']} | test {m_test}", flush=True)


if __name__ == "__main__":
    main()
