"""
Figura query / galleria sul test: per ogni fascia di intervallo una query riconosciuta, piu' alcuni errori; per ogni query
le prime 5 foto della galleria (stesso lato) con data e coseno.

    python figura_query_galleria.py --device cpu          output: runs/figure/query_galleria_<nome>.png
"""

import os
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import build_records
from split_fisso import carica_split
from reid import protocollo
from reid_aliked import FASCE
from riferimenti_globali import embedding, foto, valuta
from lejepa_multivista import LATO_GLOBALE

GIUSTO, SBAGLIATO = "#1b7f3b", "#c0392b"


def main():
    ap = argparse.ArgumentParser(description="Figura query / galleria")
    ap.add_argument("--backbone", default=os.path.join("runs", "lejepa_catture_dinov2_puro_viste_arcface_contrastiva_cls", "migliore.pth"))
    ap.add_argument("--nome", default="invarianza_contrastiva")
    ap.add_argument("--titolo", default="Invarianza + contrastiva")
    ap.add_argument("--errori", type=int, default=2)
    ap.add_argument("--seme", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    import timm
    device = torch.device(args.device)
    recs = carica_split(build_records("head"))["test"]
    galleria, query = protocollo(recs)
    ds = foto(recs, LATO_GLOBALE)
    b = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=False, img_size=LATO_GLOBALE, num_classes=0).to(device).eval()
    b.load_state_dict(torch.load(args.backbone, map_location="cpu", weights_only=False)["backbone"])
    E = embedding(lambda x: b(x), ds, device)
    righe, _, _ = valuta(E, recs, galleria, query)

    rng = np.random.default_rng(args.seme)
    scelte = []
    for nome, lo, hi in FASCE:
        giuste = [r for r in righe if r["r1"] == 1 and lo <= r["giorni"] <= hi]
        if giuste:
            scelte.append((f"riconosciuto, {nome}", giuste[rng.integers(len(giuste))]))
    errate = [r for r in righe if r["r1"] == 0]
    for k in rng.choice(len(errate), min(args.errori, len(errate)), replace=False):
        scelte.append(("non riconosciuto al 1° posto", errate[k]))

    fig, axes = plt.subplots(len(scelte), 6, figsize=(13, 2.85 * len(scelte)))
    for row, (etichetta, r) in enumerate(scelte):
        q = r["q"]
        a = axes[row, 0]
        a.imshow(ds[q][0].permute(1, 2, 0).numpy())
        a.set_title(f"QUERY ({etichetta})\npesce {r['fish']}, {recs[q]['date']}", fontsize=8, fontweight="bold")
        for s in a.spines.values():
            s.set_edgecolor("black")
            s.set_linewidth(2)
        for c, i in enumerate(r["primi"], start=1):
            giusto = recs[i]["fish_id"] == r["fish"]
            a = axes[row, c]
            a.imshow(ds[i][0].permute(1, 2, 0).numpy())
            a.set_title(f"{c}°: {'stesso pesce' if giusto else 'altro pesce'}\n{recs[i]['date']}, coseno {float(E[i] @ E[q]):.2f}",
                        fontsize=8, color=GIUSTO if giusto else SBAGLIATO)
            for s in a.spines.values():
                s.set_edgecolor(GIUSTO if giusto else SBAGLIATO)
                s.set_linewidth(3 if giusto else 1)
        for a in axes[row]:
            a.set_xticks([])
            a.set_yticks([])
    fig.suptitle(f"{args.titolo}: query e prime 5 foto della galleria (test, stesso lato)", fontsize=13, y=0.997)
    plt.tight_layout(rect=(0, 0, 1, 0.985))
    os.makedirs(os.path.join("runs", "figure"), exist_ok=True)
    path = os.path.join("runs", "figure", f"query_galleria_{args.nome}.png")
    plt.savefig(path, dpi=110)
    print("salvata", path)


if __name__ == "__main__":
    main()
