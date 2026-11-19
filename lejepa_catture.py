"""
Invarianza tra catture con la loss di LeJEPA: le due viste globali sono due catture dello stesso pesce e lato in date
diverse (pesci di train con almeno due date), senza condizionamento sul tempo.
  --da dinov2_puro | dinov2 | zero   backbone di partenza: DINOv2, oppure il modello multi-vista da DINOv2 o da zero
  --viste lejepa                     1 globale + 3 locali per cattura (default)
  --viste arcface                    un ritaglio 224 (scala 0,6-1) per cattura, come in arcface.py
  --senza_proiettore                 loss e SIGReg sul token di classe

    python lejepa_catture.py --da dinov2_puro --epochs 60 --checkpointing     output: runs/lejepa_catture_<da>[_<etichetta>]/
"""

import os
import csv
import json
import time
import random
import argparse
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from dataset import build_records
from split_fisso import carica_split
from reid_locale import sveglio
from ijepa_model import SIGReg
from lejepa_multivista import FotoCache, Viste, Modello, loss_lejepa, lr_al_passo, valuta_backbone, N_GLOBALI

N_LOCALI_PER_CATTURA = 3


class CoppieCatture(Dataset):
    """Per ogni (pesce, lato) con almeno due date: due foto da due date diverse scelte a caso."""

    def __init__(self, records, seed=0):
        self.foto = FotoCache(records)
        per_chiave = defaultdict(lambda: defaultdict(list))
        for i, r in enumerate(records):
            per_chiave[(r["fish_id"], r["side"])][r["date"]].append(i)
        self.chiavi = [list(d.values()) for d in per_chiave.values() if len(d) >= 2]
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.chiavi)

    def __getitem__(self, k):
        date = self.chiavi[k]
        a, b = self.rng.sample(range(len(date)), 2)
        return self.foto[self.rng.choice(date[a])][0], self.foto[self.rng.choice(date[b])][0]


@torch.no_grad()
def viste_coppia(viste, ua, ub):
    """Catture A e B -> 2 viste globali [2B, ...] (A poi B) e 6 locali [6B, ...] (3 da A, 3 da B)."""
    xa, xb = ua.float() / 255, ub.float() / 255
    g = torch.cat([viste.globale(xa), viste.globale(xb)])
    l = torch.cat([viste.locale(xa) for _ in range(N_LOCALI_PER_CATTURA)] + [viste.locale(xb) for _ in range(N_LOCALI_PER_CATTURA)])
    return (g.clamp(0, 1) - viste.mean) / viste.std, (l.clamp(0, 1) - viste.mean) / viste.std


@torch.no_grad()
def viste_coppia_arcface(ritaglio, fotometriche, viste, ua, ub):
    """Viste come in arcface.py: un ritaglio 224 (scala 0,6-1) per cattura, nessuna vista locale."""
    g = torch.cat([fotometriche(ritaglio(u.float() / 255)) for u in (ua, ub)]).clamp(0, 1)
    return (g - viste.mean) / viste.std, None


def main():
    ap = argparse.ArgumentParser(description="Invarianza tra catture con la loss di LeJEPA")
    ap.add_argument("--da", choices=["dinov2", "zero", "dinov2_puro"], required=True,
                    help="dinov2_puro: pesi di DINOv2; dinov2 / zero: modello multi-vista corrispondente")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lr_backbone", type=float, default=5e-5)
    ap.add_argument("--lam", type=float, default=0.05)
    ap.add_argument("--ogni", type=int, default=5)
    ap.add_argument("--checkpointing", action="store_true")
    ap.add_argument("--riprendi", action="store_true")
    ap.add_argument("--passi_prova", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--viste", choices=["lejepa", "arcface"], default="lejepa",
                    help="lejepa: 1 globale + 3 locali per cattura; arcface: un ritaglio per cattura")
    ap.add_argument("--senza_proiettore", action="store_true", help="loss e SIGReg sul token di classe del backbone")
    ap.add_argument("--etichetta", default="", help="suffisso della cartella di output")
    args = ap.parse_args()
    sveglio()
    torch.manual_seed(0)
    device = torch.device(args.device)
    out_dir = os.path.join("runs", f"lejepa_catture_{args.da}" + (f"_{args.etichetta}" if args.etichetta else ""))
    os.makedirs(out_dir, exist_ok=True)

    insiemi = carica_split(build_records("head"))
    dati = CoppieCatture(insiemi["train"])
    loader = DataLoader(dati, batch_size=args.batch, shuffle=True, drop_last=True, num_workers=0, pin_memory=device.type == "cuda")
    viste = Viste().to(device)
    modello = Modello("dinov2" if args.da == "dinov2_puro" else "zero", proiettore=not args.senza_proiettore).to(device)
    if args.viste == "arcface":
        import kornia.augmentation as K
        from lejepa_multivista import LATO_GLOBALE
        ritaglio = K.RandomResizedCrop((LATO_GLOBALE, LATO_GLOBALE), scale=(0.6, 1.0)).to(device)
        fotometriche = viste.globale[1:]
        fai_viste = lambda ua, ub: viste_coppia_arcface(ritaglio, fotometriche, viste, ua, ub)
    else:
        fai_viste = lambda ua, ub: viste_coppia(viste, ua, ub)
    partenza = os.path.join("runs", f"lejepa_multivista_{args.da}", "migliore.pth")
    if args.da == "dinov2_puro":
        print("backbone: pesi di DINOv2, senza il passo 1", flush=True)
    elif os.path.exists(partenza):
        ck = torch.load(partenza, map_location="cpu", weights_only=False)
        modello.backbone.load_state_dict(ck["backbone"])
        print(f"backbone dal passo 1: {partenza}, epoca {ck['epoca']}, validation {ck['val']['mAP'][0]:.2f} mAP", flush=True)
    elif not args.passi_prova:
        raise FileNotFoundError(f"manca il modello del passo 1: {partenza}")
    if args.checkpointing:
        modello.backbone.set_grad_checkpointing(True)
    sigreg = SIGReg().to(device)
    gruppi = [{"params": modello.backbone.parameters(), "lr": args.lr_backbone, "base": args.lr_backbone}]
    gruppi.append({"params": list(modello.proj.parameters()), "lr": args.lr, "base": args.lr})   # vuoto senza proiettore
    opt = torch.optim.AdamW(gruppi, weight_decay=0.05)
    totale = args.epochs * len(loader)
    warmup = max(1, round(0.05 * totale))
    epoca0, passo, migliore = 1, 0, -1.0
    log = os.path.join(out_dir, "train_log.csv")
    print(f"{len(dati)} coppie (pesce, lato) con almeno due date, {len(loader)} passi per epoca", flush=True)
    if args.riprendi and os.path.exists(os.path.join(out_dir, "ultimo.pth")):
        ck = torch.load(os.path.join(out_dir, "ultimo.pth"), map_location="cpu", weights_only=False)
        modello.load_state_dict(ck["modello"])
        opt.load_state_dict(ck["opt"])
        epoca0, passo, migliore = ck["epoca"] + 1, ck["passo"], ck["migliore"]
        print(f"riprendo dall'epoca {epoca0}", flush=True)
    elif not args.passi_prova:
        with open(log, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["epoca", "loss", "invarianza", "sigreg", "lr", "val_rank1", "val_rank5", "val_mAP", "minuti"])
        _, m, _ = valuta_backbone(modello.backbone, insiemi["validation"], device)
        print(f"[prima del training] validation: {m}", flush=True)
        with open(log, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([0, "", "", "", "", m["rank-1"][0], m["rank-5"][0], m["mAP"][0], 0])

    t0 = time.time()
    modello.train()
    for epoca in range(epoca0, args.epochs + 1):
        somme, n = np.zeros(3), 0
        for ua, ub in loader:
            for gr in opt.param_groups:
                gr["lr"] = lr_al_passo(passo, totale, warmup, gr["base"])
            g, l = fai_viste(ua.to(device, non_blocking=True), ub.to(device, non_blocking=True))
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                z = modello(g, l)
            loss, sim, sig = loss_lejepa(z, sigreg, passo, args.lam)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            somme += [loss.item(), sim.item(), sig.item()]
            n, passo = n + 1, passo + 1
            if args.passi_prova and passo >= args.passi_prova:
                print(f"[prova] {passo} passi | loss {loss.item():.4f} (invarianza {sim.item():.4f}, sigreg {sig.item():.3f}) | "
                      f"{(time.time() - t0) / passo:.2f} s/passo | z {tuple(z.shape)}", flush=True)
                return
        medie = somme / max(n, 1)
        minuti = (time.time() - t0) / 60
        riga = [epoca, f"{medie[0]:.5f}", f"{medie[1]:.5f}", f"{medie[2]:.4f}", f"{opt.param_groups[1]['lr']:.2e}", "", "", "", f"{minuti:.1f}"]
        testo = f"[epoca {epoca}/{args.epochs}] loss {medie[0]:.4f} | invarianza {medie[1]:.4f} | sigreg {medie[2]:.3f} | {minuti:.0f} min"
        if epoca % args.ogni == 0 or epoca == args.epochs:
            _, m, _ = valuta_backbone(modello.backbone, insiemi["validation"], device)
            riga[5:8] = [m["rank-1"][0], m["rank-5"][0], m["mAP"][0]]
            testo += f" | validation rank-1 {m['rank-1'][0]:.2f} mAP {m['mAP'][0]:.2f}"
            if m["mAP"][0] > migliore:
                migliore = m["mAP"][0]
                torch.save({"backbone": modello.backbone.state_dict(), "epoca": epoca, "val": m, "args": vars(args)},
                           os.path.join(out_dir, "migliore.pth"))
            torch.save({"modello": modello.state_dict(), "opt": opt.state_dict(), "epoca": epoca, "passo": passo,
                        "migliore": migliore}, os.path.join(out_dir, "ultimo.pth"))
        print(testo, flush=True)
        with open(log, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(riga)

    ck = torch.load(os.path.join(out_dir, "migliore.pth"), map_location="cpu", weights_only=False)
    modello.backbone.load_state_dict(ck["backbone"])
    _, m_test, pf = valuta_backbone(modello.backbone, insiemi["test"], device)
    ris = {"da": args.da, "epoca scelta (validation)": ck["epoca"], "validation": ck["val"], "test": m_test, "test per fascia": pf}
    with open(os.path.join(out_dir, "risultati.json"), "w", encoding="utf-8") as f:
        json.dump(ris, f, indent=1, ensure_ascii=False, default=float)
    print(f"FINE: epoca scelta {ck['epoca']} | test {m_test}", flush=True)


if __name__ == "__main__":
    main()
