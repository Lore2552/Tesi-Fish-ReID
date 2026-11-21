"""
Come riconosce i pesci un modello, e se usa scorciatoie. Sul test:
  esempi      : query e prime 5 foto della classifica, a caso e tra gli errori
  occlusione  : un quadrato grigio 32x32 scorre sulla foto (passo 16); mappa del calo di coseno con la foto giusta
  degradazioni: re-identificazione con l'input privato di un'informazione alla volta (colore, strisce, contenuto)

    python controllo_scorciatoie.py                         output: runs/controllo_scorciatoie/
"""

import os
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision.transforms.functional import gaussian_blur

from dataset import build_records, normalize_uint8
from split_fisso import carica_split
from reid import protocollo, figura_esempi, caso
from reid_locale import FONDO_MIN
from riferimenti_globali import embedding, foto, valuta
from lejepa_multivista import LATO_GLOBALE

MODELLI = {"5c": os.path.join("runs", "lejepa_catture_dinov2_puro_viste_arcface_contrastiva_cls", "migliore.pth"),
           "ArcFace": os.path.join("runs", "arcface_dinov2_catture_prototipi", "migliore.pth"),
           "DINOv2": "dinov2"}
OUT_DIR = os.path.join("runs", "controllo_scorciatoie")
LATO_OCCL, PASSO_OCCL = 32, 16


def grigi(x):
    g = (0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]).round().clamp(0, 255)
    return g.to(torch.uint8).expand(3, -1, -1).clone()


def sfocata(sigma):
    k = int(6 * sigma) | 1
    return lambda x: gaussian_blur(x.float()[None], [k, k], [sigma, sigma])[0].round().clamp(0, 255).to(torch.uint8)


def sagoma(x):
    pesce = x.min(0).values <= FONDO_MIN
    return torch.where(pesce, 90, 255).to(torch.uint8).expand(3, -1, -1).clone()


def colore_medio(x):
    return x.float().mean((1, 2), keepdim=True).round().to(torch.uint8).expand_as(x).clone()


DEGRADAZIONI = {"originale": None, "scala di grigi": grigi, "sfocata (sigma 2 px)": sfocata(2.0),
                "sfocata (sigma 4 px)": sfocata(4.0), "solo sagoma": sagoma, "solo colore medio": colore_medio}


class Variante:
    def __init__(self, ds, fn):
        self.ds, self.fn = ds, fn

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x, j = self.ds[i]
        return (x if self.fn is None else self.fn(x)), j


def carica(nome, device):
    import timm
    p = MODELLI[nome]
    b = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=p == "dinov2", img_size=LATO_GLOBALE,
                          num_classes=0).to(device).eval()
    if p != "dinov2":
        b.load_state_dict(torch.load(p, map_location="cpu", weights_only=False)["backbone"])
    return b


@torch.no_grad()
def mappa_occlusione(backbone, x, e_bersaglio, device):
    """Calo del coseno con e_bersaglio quando un quadrato grigio copre ogni posizione di x [3, H, W] uint8 -> [H, W]."""
    H = x.shape[-1]
    pos = list(range(0, H - LATO_OCCL + 1, PASSO_OCCL))
    lotto = []
    for i in pos:
        for j in pos:
            y = x.clone()
            y[:, i:i + LATO_OCCL, j:j + LATO_OCCL] = x.float().mean((1, 2), keepdim=True).to(torch.uint8)
            lotto.append(y)
    lotto = torch.stack([x] + lotto).to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        e = F.normalize(backbone(normalize_uint8(lotto)).float(), dim=-1)
    s = e @ e_bersaglio
    calo = (s[0] - s[1:]).cpu().numpy()
    somma, conta = np.zeros((H, H)), np.zeros((H, H))
    for n, (i, j) in enumerate((i, j) for i in pos for j in pos):
        somma[i:i + LATO_OCCL, j:j + LATO_OCCL] += calo[n]
        conta[i:i + LATO_OCCL, j:j + LATO_OCCL] += 1
    return somma / np.maximum(conta, 1), float(s[0])


def figura_occlusione(backbone, ds, recs, righe, E, device, path, n=6, seed=2):
    giuste = [r for r in righe if r["r1"] == 1]
    scelte = [giuste[k] for k in np.random.default_rng(seed).choice(len(giuste), n, replace=False)]
    fig, axes = plt.subplots(n, 4, figsize=(11, 2.8 * n))
    for row, r in enumerate(scelte):
        q, g = r["q"], r["primi"][0]
        xq, xg = ds[q][0], ds[g][0]
        mq, sq = mappa_occlusione(backbone, xq, torch.from_numpy(E[g]).to(device), device)
        mg, _ = mappa_occlusione(backbone, xg, torch.from_numpy(E[q]).to(device), device)
        lim = max(np.abs(mq).max(), np.abs(mg).max(), 1e-6)
        for c, (img, mappa, titolo) in enumerate(((xq, None, f"query, {recs[q]['date']}"),
                                                  (xq, mq, "query: dove conta"),
                                                  (xg, mg, "foto giusta: dove conta"),
                                                  (xg, None, f"foto giusta, {recs[g]['date']}"))):
            a = axes[row, c]
            a.imshow(img.permute(1, 2, 0).numpy())
            if mappa is not None:
                a.imshow(mappa, cmap="magma", alpha=0.6, vmin=0, vmax=lim)
            a.set_title(titolo + (f"\ncoseno {sq:.2f}, {r['giorni']} giorni" if c == 0 else ""), fontsize=8)
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Modello 5c: zone che fanno riconoscere il pesce (più chiaro = calo maggiore se coperte)", fontsize=12, y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.98))
    plt.savefig(path, dpi=110)
    plt.close()


def figura_degradazioni(ds, path):
    i = 0
    fig, axes = plt.subplots(1, len(DEGRADAZIONI), figsize=(2.4 * len(DEGRADAZIONI), 2.8))
    for a, (nome, fn) in zip(axes, DEGRADAZIONI.items()):
        x = ds[i][0] if fn is None else fn(ds[i][0])
        a.imshow(x.permute(1, 2, 0).numpy())
        a.set_title(nome, fontsize=9)
        a.set_xticks([]); a.set_yticks([])
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Esempi, mappe di occlusione e input degradati sul test")
    ap.add_argument("--modelli", nargs="+", default=list(MODELLI))
    args = ap.parse_args()
    device = torch.device("cuda")
    os.makedirs(OUT_DIR, exist_ok=True)
    recs = carica_split(build_records("head"))["test"]
    galleria, query = protocollo(recs)
    ds = foto(recs, LATO_GLOBALE)
    figura_degradazioni(ds, os.path.join(OUT_DIR, "degradazioni.png"))

    out, righe_testo = {}, [f"Test: {len(query)} query, galleria {len(galleria)} foto", ""]
    for nome in args.modelli:
        backbone = carica(nome, device)
        out[nome] = {}
        for dnome, fn in DEGRADAZIONI.items():
            E = embedding(lambda x: backbone(x), Variante(ds, fn), device)
            righe, m, _ = valuta(E, recs, galleria, query)
            out[nome][dnome] = m
            righe_testo.append(f"{nome:8s} | {dnome:22s} | " + " | ".join(f"{k} {v[0]:6.2f} ({v[1]:.2f}-{v[2]:.2f})" for k, v in m.items()))
            print(righe_testo[-1], flush=True)
            if nome == "5c" and fn is None:
                out["classifica casuale"] = [round(float(v), 2) for v in caso(righe)]
                figura_esempi(ds, recs, righe, os.path.join(OUT_DIR, "esempi_5c.png"), "Modello 5c: query a caso e prime 5 risposte")
                errori = [r for r in righe if r["r1"] == 0]
                figura_esempi(ds, recs, errori, os.path.join(OUT_DIR, "errori_5c.png"),
                              f"Modello 5c: query sbagliate al primo posto ({len(errori)} su {len(righe)})", seed=3)
                figura_occlusione(backbone, ds, recs, righe, E, device, os.path.join(OUT_DIR, "occlusione_5c.png"))
        righe_testo.append("")
        del backbone
        torch.cuda.empty_cache()
    righe_testo.append(f"classifica casuale: rank-1 / rank-5 / mAP {out.get('classifica casuale')}")
    with open(os.path.join(OUT_DIR, "risultati.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(righe_testo) + "\n")
    with open(os.path.join(OUT_DIR, "risultati.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
