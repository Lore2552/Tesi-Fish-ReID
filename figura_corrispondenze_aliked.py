"""
Corrispondenze ALIKED + LightGlue + RANSAC tra due foto: stesso pesce a distanza di anni e pesce diverso. Si disegnano
solo gli accoppiamenti coerenti con l'omografia (inlier).

    python figura_corrispondenze_aliked.py            output: runs/figure/corrispondenze_aliked.png
"""

import os

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

from dataset import build_records
from split_fisso import carica_split
from reid_locale import foto_originale, N_PUNTI, SOGLIA_RANSAC

COPPIE = (("P8316144_2020", "P7066063_2023", "stesso pesce (423051), a quasi 3 anni di distanza"),
          ("P7066063_2023", None, "pesci diversi, stesso lato"))


@torch.no_grad()
def accoppia(estr, matcher, a, b, device):
    out = []
    for im in (a, b):
        x = torch.from_numpy(im).permute(2, 0, 1).float().div(255)[None].to(device)
        out.append(estr({"image": x}))
    m = matcher({"image0": out[0], "image1": out[1]})["matches"][0].cpu().numpy()
    p0 = out[0]["keypoints"][0].cpu().numpy()[m[:, 0]]
    p1 = out[1]["keypoints"][0].cpu().numpy()[m[:, 1]]
    if len(p0) < 4:
        return p0[:0], p1[:0], len(m)
    H, ok = cv2.findHomography(p0, p1, cv2.USAC_MAGSAC, SOGLIA_RANSAC, maxIters=10000, confidence=0.999)
    ok = np.zeros(len(p0), bool) if ok is None else ok.ravel().astype(bool)
    return p0[ok], p1[ok], len(m)


def main():
    from lightglue import ALIKED, LightGlue
    device = torch.device("cuda")
    recs = carica_split(build_records("head"))["test"]
    per_stem = {r["stem"]: r for r in recs}
    estr = ALIKED(max_num_keypoints=N_PUNTI, detection_threshold=-1).eval().to(device)
    matcher = LightGlue(features="aliked").eval().to(device)
    q = per_stem[COPPIE[0][1]]
    altri = [r for r in recs if r["side"] == q["side"] and r["fish_id"] != q["fish_id"]]
    im_q = foto_originale(q["path"])
    migliore, migliore_n = None, -1
    for r in np.random.default_rng(0).choice(len(altri), 300, replace=False):
        p0, _, _ = accoppia(estr, matcher, im_q, foto_originale(altri[r]["path"]), device)
        if len(p0) > migliore_n:
            migliore, migliore_n = altri[r], len(p0)

    fig, axes = plt.subplots(2, 2, figsize=(10, 9.2))
    for row, (sa, sb, titolo) in enumerate(COPPIE):
        ra, rb = per_stem[sa], (per_stem[sb] if sb else migliore)
        a, b = foto_originale(ra["path"]), foto_originale(rb["path"])
        p0, p1, n_acc = accoppia(estr, matcher, a, b, device)
        axes[row, 0].imshow(a)
        axes[row, 1].imshow(b)
        axes[row, 0].set_title(f"pesce {ra['fish_id']}, {ra['date']}", fontsize=9)
        axes[row, 1].set_title(f"pesce {rb['fish_id']}, {rb['date']}", fontsize=9)
        colore = "#1b7f3b" if ra["fish_id"] == rb["fish_id"] else "#c0392b"
        for (x0, y0), (x1, y1) in zip(p0, p1):
            fig.add_artist(ConnectionPatch(xyA=(x0, y0), coordsA=axes[row, 0].transData, xyB=(x1, y1),
                                           coordsB=axes[row, 1].transData, color=colore, lw=0.6, alpha=0.7))
        for a_ in axes[row]:
            a_.set_xticks([]); a_.set_yticks([])
        axes[row, 0].set_xlabel(f"{titolo}: {len(p0)} inlier su {n_acc} accoppiamenti", fontsize=11, color=colore,
                                loc="left", labelpad=8)
    plt.tight_layout(h_pad=2.5)
    os.makedirs(os.path.join("runs", "figure"), exist_ok=True)
    path = os.path.join("runs", "figure", "corrispondenze_aliked.png")
    plt.savefig(path, dpi=110)
    print("salvata", path, "| miglior pesce diverso su 300:", migliore["fish_id"], migliore_n)


if __name__ == "__main__":
    main()
