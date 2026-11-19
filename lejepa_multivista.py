"""
LeJEPA multi-vista senza etichette
  viste      : 2 globali (224 px, scala 0,3-1) + 6 locali (98 px, scala 0,05-0,3) da foto 320x320, ColorJitter,
               scala di grigi e sfocatura; niente flip e niente solarizzazione
  modello    : ViT-S/14 (--init dinov2 | zero), embedding = token di classe; proiettore MLP 384-2048-2048-512
  loss       : (1 - lambda) * distanza delle viste dal centro delle globali + lambda * SIGReg, lambda = 0,05
  ottimizz.  : AdamW, lr 5e-4 (backbone DINOv2 5e-5), wd 0,05, warmup 5% e coseno fino a lr/1000, bf16
Validation ogni --ogni epoche, test una volta sul checkpoint migliore.

    python lejepa_multivista.py --init dinov2 --epochs 30 --checkpointing      output: runs/lejepa_multivista_<init>/
"""

import os
import csv
import json
import math
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import MLP

from dataset import build_records, build_image_cache, IMAGENET_MEAN, IMAGENET_STD
from split_fisso import carica_split
from reid import protocollo
from reid_locale import sveglio
from ijepa_model import SIGReg
from riferimenti_globali import embedding, valuta, foto

N_GLOBALI, N_LOCALI = 2, 6
LATO_FOTO, LATO_GLOBALE, LATO_LOCALE = 320, 224, 98


class FotoCache(Dataset):
    """Foto uint8 [3, 320, 320] lette dalla cache."""

    def __init__(self, records):
        self.path = build_image_cache("head", (LATO_FOTO, LATO_FOTO))
        with open(self.path + ".txt", encoding="utf-8") as f:
            riga = {s: i for i, s in enumerate(f.read().split("\n"))}
        self.righe = [riga[r["stem"]] for r in records]
        self.dim = 3 * LATO_FOTO * LATO_FOTO
        self.f = None

    def __len__(self):
        return len(self.righe)

    def __getitem__(self, i):
        if self.f is None:
            self.f = open(self.path, "rb", buffering=0)
        self.f.seek(self.righe[i] * self.dim)
        x = np.frombuffer(self.f.read(self.dim), dtype=np.uint8).reshape(3, LATO_FOTO, LATO_FOTO)
        return torch.from_numpy(x.copy()), i


class Viste(nn.Module):
    """Batch uint8 [B, 3, 320, 320] -> viste globali [2B, ...] e locali [6B, ...] normalizzate, sulla GPU."""

    def __init__(self):
        super().__init__()
        import kornia.augmentation as K

        def fotometriche():
            return [K.ColorJitter(0.4, 0.4, 0.2, 0.1, p=0.8), K.RandomGrayscale(p=0.2),
                    K.RandomGaussianBlur((9, 9), (0.1, 2.0), p=0.5)]
        self.globale = nn.Sequential(K.RandomResizedCrop((LATO_GLOBALE, LATO_GLOBALE), scale=(0.3, 1.0)), *fotometriche())
        self.locale = nn.Sequential(K.RandomResizedCrop((LATO_LOCALE, LATO_LOCALE), scale=(0.05, 0.3)), *fotometriche())
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, u8):
        x = u8.float() / 255
        g = torch.cat([self.globale(x) for _ in range(N_GLOBALI)])
        l = torch.cat([self.locale(x) for _ in range(N_LOCALI)])
        return (g.clamp(0, 1) - self.mean) / self.std, (l.clamp(0, 1) - self.mean) / self.std


class Modello(nn.Module):
    def __init__(self, init, proiettore=True):
        super().__init__()
        import timm
        self.backbone = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=init == "dinov2", img_size=LATO_GLOBALE,
                                          dynamic_img_size=True, num_classes=0)
        self.proj = MLP(384, [2048, 2048, 512], norm_layer=nn.BatchNorm1d) if proiettore else nn.Identity()

    def forward(self, g, l):
        """Viste globali g e locali l (o None) -> proiezioni [V, B, K], prima le globali."""
        e = self.backbone(g) if l is None else torch.cat([self.backbone(g), self.backbone(l)])
        z = self.proj(e).float()
        return z.view(-1, g.shape[0] // N_GLOBALI, z.shape[-1])


def loss_lejepa(z, sigreg, passo, lam):
    centro = z[:N_GLOBALI].mean(0)
    sim = (centro - z).square().mean()
    sig = torch.stack([sigreg(z[v], seed=passo * len(z) + v) for v in range(len(z))]).mean()
    return (1 - lam) * sim + lam * sig, sim.detach(), sig.detach()


def lr_al_passo(passo, totale, warmup, base):
    if passo < warmup:
        return base * (passo + 1) / warmup
    p = (passo - warmup) / max(1, totale - warmup)
    fine = base / 1000
    return fine + 0.5 * (base - fine) * (1 + math.cos(math.pi * p))


def valuta_backbone(backbone, records, device):
    galleria, query = protocollo(records)
    backbone.eval()
    emb = embedding(lambda x: backbone(x), foto(records, LATO_GLOBALE), device)
    backbone.train()
    return valuta(emb, records, galleria, query)


def main():
    ap = argparse.ArgumentParser(description="LeJEPA multi-vista senza etichette")
    ap.add_argument("--init", choices=["dinov2", "zero"], required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lr_backbone", type=float, default=None, help="default: 5e-5 da DINOv2, --lr da zero")
    ap.add_argument("--lam", type=float, default=0.05)
    ap.add_argument("--ogni", type=int, default=2, help="epoche tra due valutazioni sulla validation")
    ap.add_argument("--checkpointing", action="store_true", help="gradient checkpointing del backbone (meno memoria)")
    ap.add_argument("--workers", type=int, default=0, help="processi di caricamento (0: nel processo principale)")
    ap.add_argument("--riprendi", action="store_true")
    ap.add_argument("--passi_prova", type=int, default=0, help="solo N passi (tempi e memoria)")
    args = ap.parse_args()
    sveglio()
    torch.manual_seed(0)
    device = torch.device("cuda")
    out_dir = os.path.join("runs", f"lejepa_multivista_{args.init}")
    os.makedirs(out_dir, exist_ok=True)
    lr_b = args.lr_backbone if args.lr_backbone is not None else (5e-5 if args.init == "dinov2" else args.lr)

    insiemi = carica_split(build_records("head"))
    loader = DataLoader(FotoCache(insiemi["train"]), batch_size=args.batch, shuffle=True, drop_last=True,
                        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    viste = Viste().to(device)
    modello = Modello(args.init).to(device)
    if args.checkpointing:
        modello.backbone.set_grad_checkpointing(True)
    sigreg = SIGReg().to(device)
    opt = torch.optim.AdamW([{"params": modello.backbone.parameters(), "lr": lr_b, "base": lr_b},
                             {"params": modello.proj.parameters(), "lr": args.lr, "base": args.lr}], weight_decay=0.05)
    totale = args.epochs * len(loader)
    warmup = max(1, round(0.05 * totale))
    epoca0, passo, migliore = 1, 0, -1.0
    log = os.path.join(out_dir, "train_log.csv")
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
        for u8, _ in loader:
            for gr in opt.param_groups:
                gr["lr"] = lr_al_passo(passo, totale, warmup, gr["base"])
            g, l = viste(u8.to(device, non_blocking=True))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                z = modello(g, l)
            loss, sim, sig = loss_lejepa(z, sigreg, passo, args.lam)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            somme += [loss.item(), sim.item(), sig.item()]
            n, passo = n + 1, passo + 1
            if args.passi_prova:
                if passo == 1:
                    t_primo = time.time()
                if passo >= args.passi_prova:
                    dt = (time.time() - t_primo) / max(1, passo - 1)
                    print(f"[prova] {passo} passi | loss {loss.item():.4f} (invarianza {sim.item():.4f}, sigreg {sig.item():.3f}) | "
                          f"{dt:.2f} s/passo -> {dt * len(loader) / 60:.1f} min/epoca | memoria max "
                          f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
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
    ris = {"init": args.init, "epoca scelta (validation)": ck["epoca"], "validation": ck["val"], "test": m_test, "test per fascia": pf}
    with open(os.path.join(out_dir, "risultati.json"), "w", encoding="utf-8") as f:
        json.dump(ris, f, indent=1, ensure_ascii=False, default=float)
    print(f"FINE: epoca scelta {ck['epoca']} | test {m_test}", flush=True)


if __name__ == "__main__":
    main()
