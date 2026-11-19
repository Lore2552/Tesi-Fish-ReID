"""
Controllo della loss LeJEPA su un dataset pubblico: ResNet-18 da zero su Flowers102, sonda lineare sul train.

    python riproduzione_lejepa.py --epochs 400         output: runs/riproduzione_lejepa/
"""

import os
import csv
import json
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import MLP
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ijepa_model import SIGReg
from lejepa_multivista import loss_lejepa, lr_al_passo, N_GLOBALI, N_LOCALI
from reid_locale import sveglio

DATI = os.path.join("..", "dati_pubblici")
OUT = os.path.join("runs", "riproduzione_lejepa")
MEDIA = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
DEV_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def carica(split):
    """Flowers102 -> immagini uint8 [N, 3, 256, 256] (lato corto 256, ritaglio centrale) ed etichette."""
    from torchvision.datasets import Flowers102
    import torchvision.transforms.v2 as T
    ds = Flowers102(DATI, split=split, download=True)
    tr = T.Compose([T.PILToTensor(), T.Resize(256, antialias=True), T.CenterCrop(256)])
    x = torch.stack([tr(ds[i][0]) for i in range(len(ds))])
    y = torch.tensor([ds[i][1] for i in range(len(ds))])
    return x, y


class Viste(nn.Module):
    """2 viste globali e 6 locali con flip e solarizzazione, sulla GPU."""

    def __init__(self):
        super().__init__()
        import kornia.augmentation as K

        def fot():
            return [K.RandomHorizontalFlip(p=0.5), K.ColorJitter(0.4, 0.4, 0.2, 0.1, p=0.8), K.RandomGrayscale(p=0.2),
                    K.RandomGaussianBlur((9, 9), (0.1, 2.0), p=0.5),
                    K.RandomSolarize(thresholds=0.0, additions=0.0, p=0.2)]
        self.g = nn.Sequential(K.RandomResizedCrop((224, 224), scale=(0.3, 1.0)), *fot())
        self.l = nn.Sequential(K.RandomResizedCrop((98, 98), scale=(0.05, 0.3)), *fot())

    @torch.no_grad()
    def forward(self, u8):
        x = u8.float() / 255
        g = torch.cat([self.g(x) for _ in range(N_GLOBALI)]).clamp(0, 1)
        l = torch.cat([self.l(x) for _ in range(N_LOCALI)]).clamp(0, 1)
        m, s = MEDIA.to(x.device), DEV_STD.to(x.device)
        return (g - m) / s, (l - m) / s


class Modello(nn.Module):
    def __init__(self):
        super().__init__()
        import timm
        self.backbone = timm.create_model("resnet18", pretrained=False, num_classes=0)
        self.proj = MLP(512, [2048, 2048, 512], norm_layer=nn.BatchNorm1d)

    def forward(self, g, l):
        e = torch.cat([self.backbone(g), self.backbone(l)])
        return self.proj(e).float().view(N_GLOBALI + N_LOCALI, g.shape[0] // N_GLOBALI, -1)


@torch.no_grad()
def caratteristiche(backbone, x, dev, batch=256):
    backbone.eval()
    out = []
    for i in range(0, len(x), batch):
        u = x[i:i + batch, :, 16:240, 16:240].to(dev).float() / 255      # ritaglio centrale 224
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out.append(backbone((u - MEDIA.to(dev)) / DEV_STD.to(dev)).float().cpu())
    backbone.train()
    return torch.cat(out).numpy()


def sonda(backbone, dati, dev):
    (xtr, ytr), (xte, yte) = dati
    ftr, fte = caratteristiche(backbone, xtr, dev), caratteristiche(backbone, xte, dev)
    sc = StandardScaler().fit(ftr)
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(ftr), ytr.numpy())
    return 100 * float((clf.predict(sc.transform(fte)) == yte.numpy()).mean())


def main():
    ap = argparse.ArgumentParser(description="LeJEPA, ResNet-18 da zero su Flowers102")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--lam", type=float, default=0.05)
    ap.add_argument("--ogni", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    sveglio()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device("cuda")
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    xtr, ytr = carica("train")
    xte, yte = carica("test")
    print(f"Flowers102: {len(xtr)} immagini di training, {len(xte)} di test ({time.time() - t0:.0f} s)", flush=True)
    xtr_dev = xtr.to(dev)

    viste, modello, sigreg = Viste().to(dev), Modello().to(dev), SIGReg().to(dev)
    opt = torch.optim.AdamW(modello.parameters(), lr=args.lr, weight_decay=args.wd)
    passi_epoca = len(xtr) // args.batch
    totale = args.epochs * passi_epoca
    warmup = max(1, round(0.05 * totale))
    log = os.path.join(OUT, "train_log.csv")
    with open(log, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoca", "loss", "invarianza", "sigreg", "accuratezza_sonda", "minuti"])
    acc0 = sonda(modello.backbone, ((xtr, ytr), (xte, yte)), dev)
    print(f"[prima del training] sonda lineare {acc0:.2f}%", flush=True)
    passo, t0 = 0, time.time()
    modello.train()
    for epoca in range(1, args.epochs + 1):
        perm = torch.randperm(len(xtr), device=dev)
        somme = np.zeros(3)
        for k in range(passi_epoca):
            for g in opt.param_groups:
                g["lr"] = lr_al_passo(passo, totale, warmup, args.lr)
            g_, l_ = viste(xtr_dev[perm[k * args.batch:(k + 1) * args.batch]])
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                z = modello(g_, l_)
            loss, sim, sig = loss_lejepa(z, sigreg, passo, args.lam)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            somme += [loss.item(), sim.item(), sig.item()]
            passo += 1
        m = somme / passi_epoca
        riga = [epoca, f"{m[0]:.5f}", f"{m[1]:.5f}", f"{m[2]:.4f}", "", f"{(time.time() - t0) / 60:.1f}"]
        testo = f"[epoca {epoca}/{args.epochs}] loss {m[0]:.4f} | invarianza {m[1]:.4f} | sigreg {m[2]:.3f} | {(time.time() - t0) / 60:.0f} min"
        if epoca % args.ogni == 0 or epoca == args.epochs:
            acc = sonda(modello.backbone, ((xtr, ytr), (xte, yte)), dev)
            riga[4] = f"{acc:.2f}"
            testo += f" | sonda lineare {acc:.2f}%"
        print(testo, flush=True)
        with open(log, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(riga)
    ris = {"riferimento": 74.57, "nostra riproduzione": acc, "prima del training": acc0,
           "epoche": args.epochs, "args": vars(args)}
    with open(os.path.join(OUT, "risultati.json"), "w", encoding="utf-8") as f:
        json.dump(ris, f, indent=1, ensure_ascii=False)
    print(f"FINE: sonda lineare {acc:.2f}% (riferimento 74,57%)", flush=True)


if __name__ == "__main__":
    main()
