"""
Lo stesso pesce a distanza di anni: pesci di test con le catture piu' distanti nel tempo, una foto per data (stesso lato).
Sotto ogni foto: data, lunghezza, coseno con la prima cattura e quante foto di altri pesci di test (stesso lato) sono
piu' vicine di lei.

    python figura_stesso_pesce.py                    output: runs/figure/stesso_pesce_anni.png
"""

import os
import argparse
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import build_records
from split_fisso import carica_split
from reid_aliked import metadati
from riferimenti_globali import embedding, foto
from lejepa_multivista import LATO_GLOBALE


def giorno(r):
    return datetime.strptime(r["date"], "%d.%m.%Y")


def main():
    ap = argparse.ArgumentParser(description="Lo stesso pesce a distanza di anni")
    ap.add_argument("--backbone", default=os.path.join("runs", "lejepa_catture_dinov2_puro_viste_arcface_contrastiva_cls", "migliore.pth"))
    ap.add_argument("--pesci", type=int, default=4)
    ap.add_argument("--max_date", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    import timm
    device = torch.device(args.device)
    recs = carica_split(build_records("head"))["test"]
    meta = metadati()
    ds = foto(recs, LATO_GLOBALE)
    b = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=False, img_size=LATO_GLOBALE, num_classes=0).to(device).eval()
    b.load_state_dict(torch.load(args.backbone, map_location="cpu", weights_only=False)["backbone"])
    E = embedding(lambda x: b(x), ds, device)

    gruppi = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(recs):
        gruppi[(r["fish_id"], r["side"])][giorno(r)].append(i)
    candidati = []
    for chiave, date in gruppi.items():
        if len(date) >= 3:
            ordine = sorted(date)
            candidati.append(((ordine[-1] - ordine[0]).days, len(date), chiave))
    candidati.sort(reverse=True)
    scelti, pesci_visti = [], set()
    for _, _, chiave in candidati:
        if chiave[0] not in pesci_visti:
            scelti.append(chiave)
            pesci_visti.add(chiave[0])
        if len(scelti) == args.pesci:
            break

    lati = np.array([r["side"] for r in recs])
    pesci = np.array([r["fish_id"] for r in recs])
    n_col = min(args.max_date, max(len(gruppi[c]) for c in scelti))
    fig, axes = plt.subplots(len(scelti), n_col, figsize=(2.3 * n_col, 2.85 * len(scelti)))
    for row, chiave in enumerate(scelti):
        date = sorted(gruppi[chiave])
        if len(date) > n_col:
            date = [date[k] for k in np.linspace(0, len(date) - 1, n_col).round().astype(int)]
        indici = [min(gruppi[chiave][d], key=lambda n: recs[n]["stem"]) for d in date]
        prima = indici[0]
        for c in range(n_col):
            a = axes[row, c]
            a.set_xticks([])
            a.set_yticks([])
            if c >= len(indici):
                a.axis("off")
                continue
            i = indici[c]
            a.imshow(ds[i][0].permute(1, 2, 0).numpy())
            lung = meta[recs[i]["stem"]]["length"]
            testo = f"{recs[i]['date']}" + ("" if np.isnan(lung) else f", {lung:.0f} mm")
            if c == 0:
                testo += "\nprima cattura"
            else:
                sel = np.flatnonzero((lati == lati[i]) & (pesci != recs[i]["fish_id"]))
                altri = int(((E[sel] @ E[i]) > float(E[prima] @ E[i])).sum())
                testo += f"\ncoseno {float(E[prima] @ E[i]):.2f}, altri pesci più vicini: {altri}"
            a.set_title(testo, fontsize=8)
        anni = (date[-1] - date[0]).days / 365.25
        axes[row, 0].set_ylabel(f"pesce {chiave[0]}\nlato {chiave[1]}\n{anni:.1f} anni", fontsize=9)
    fig.suptitle("Lo stesso pesce a distanza di anni (pesci di test, mai visti dal modello)", fontsize=12, y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.975))
    os.makedirs(os.path.join("runs", "figure"), exist_ok=True)
    path = os.path.join("runs", "figure", "stesso_pesce_anni.png")
    plt.savefig(path, dpi=110)
    print("salvata", path)


if __name__ == "__main__":
    main()
