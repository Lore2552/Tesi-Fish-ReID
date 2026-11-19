"""
Controllo di fuga di informazione tra train e test:
    python controllo_fuga.py           output: runs/controllo_fuga/risultati.json
"""

import os
import json
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from dataset import build_records, FishImages
from split_fisso import carica_split
from reid_aliked import metadati

OUT = os.path.join("runs", "controllo_fuga")


def impronte(recs):
    ds = FishImages(recs, img_size=(224, 224), cache_path=os.path.join("cache", "head_224x224.u8"))
    h, p = [], []
    for i in range(0, len(ds), 256):
        x = torch.stack([ds[j][0] for j in range(i, min(i + 256, len(ds)))]).float().mean(1, keepdim=True)
        a = F.adaptive_avg_pool2d(x, 16).flatten(1)
        h.append((a > a.mean(1, keepdim=True)).numpy())
        b = F.adaptive_avg_pool2d(x, 64).flatten(1)
        b = (b - b.mean(1, keepdim=True)) / (b.std(1, keepdim=True) + 1e-6)
        p.append(b.numpy() / np.sqrt(b.shape[1]))
    return np.concatenate(h), np.concatenate(p).astype(np.float32)


def main():
    os.makedirs(OUT, exist_ok=True)
    s = carica_split(build_records("head"))
    out = {}
    pesci = {k: {r["fish_id"] for r in v} for k, v in s.items()}
    file = {k: {r["path"] for r in v} for k, v in s.items()}
    out["1. pesci in comune"] = {f"{a}-{b}": len(pesci[a] & pesci[b]) for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))}
    out["1. file in comune"] = {f"{a}-{b}": len(file[a] & file[b]) for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))}

    htr, ptr = impronte(s["train"])
    hte, pte = impronte(s["test"])
    vicini_h, vicini_p, esempi = 0, 0, []
    Htr = torch.from_numpy(htr).float()
    for i in range(0, len(hte), 512):
        Hte = torch.from_numpy(hte[i:i + 512]).float()
        dist = Hte.shape[1] - (Hte @ Htr.T + (1 - Hte) @ (1 - Htr).T)          # distanza di Hamming
        corr = torch.from_numpy(pte[i:i + 512]) @ torch.from_numpy(ptr).T
        vicini_h += int((dist.min(1).values <= 3).sum())
        m = corr.max(1)
        sel = (m.values > 0.99).nonzero().flatten()
        vicini_p += len(sel)
        for k in sel[:10].tolist():
            esempi.append((s["test"][i + k]["stem"], s["train"][int(m.indices[k])]["stem"], round(float(m.values[k]), 4)))
    out["2. foto di test con una foto di train quasi identica"] = {
        "impronta (Hamming <= 3 su 256)": vicini_h, "pixel 64x64 (correlazione > 0,99)": vicini_p,
        "foto di test": len(hte), "esempi (test, train, correlazione)": esempi}

    meta = metadati()
    untagged = {k: sorted({r["fish_id"] for r in v if str(r["fish_id"]).startswith("untagged")}) for k, v in s.items()}
    out["3. ID temporanei (untagged) per insieme"] = {k: len(v) for k, v in untagged.items()}
    chiave = defaultdict(set)
    for k, v in s.items():
        for r in v:
            m = meta[r["stem"]]
            chiave[(r["date"], m.get("length"))].add((k, r["fish_id"]))
    sospetti = [(d, l, sorted(x)) for (d, l), x in chiave.items()
                if len({i for _, i in x}) > 1 and len({k for k, _ in x}) > 1 and l == l]
    out["3. stessa data e stessa lunghezza, pesci diversi in insiemi diversi (coppie da guardare)"] = len(sospetti)
    print(json.dumps(out, indent=1, ensure_ascii=False))
    with open(os.path.join(OUT, "risultati.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False, default=str)


if __name__ == "__main__":
    main()
