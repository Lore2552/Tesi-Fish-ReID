"""
    python controllo_sessione.py dinov2 runs/arcface_dinov2_catture_prototipi/migliore.pth ...
"""

import os
import sys
import json

import numpy as np
import torch

from dataset import build_records
from split_fisso import carica_split
from reid import protocollo, bootstrap
from riferimenti_globali import embedding, foto
from lejepa_multivista import LATO_GLOBALE


def analisi(E, recs, galleria, query):
    ids = np.array([r["fish_id"] for r in recs])
    lati = np.array([r["side"] for r in recs])
    date = np.array([r["date"] for r in recs])
    g = np.array(galleria)
    diff, pesci, primo_stessa, quota_stessa = [], [], [], []
    for q in query:
        gi = g[(lati[g] == lati[q])]
        s = E[gi] @ E[q]
        altri = ids[gi] != ids[q]
        stessa = altri & (date[gi] == date[q])
        diversa = altri & (date[gi] != date[q])
        if stessa.any() and diversa.any():
            diff.append(s[stessa].mean() - s[diversa].mean())
            pesci.append(ids[q])
        primo = np.argmax(s)
        if ids[gi[primo]] != ids[q]:
            primo_stessa.append(date[gi[primo]] == date[q])
            quota_stessa.append(stessa.sum() / altri.sum())
    lo, hi = bootstrap(pesci, diff)
    return {"query con altri pesci della stessa sessione": len(diff),
            "coseno stessa sessione - date diverse": [round(float(np.mean(diff)), 4), round(lo, 4), round(hi, 4)],
            "prime posizioni sbagliate": len(primo_stessa),
            "di cui dalla stessa sessione (%)": round(100 * float(np.mean(primo_stessa)), 1) if primo_stessa else None,
            "attese per caso (%)": round(100 * float(np.mean(quota_stessa)), 1) if quota_stessa else None}


def main():
    import timm
    device = torch.device("cuda")
    recs = carica_split(build_records("head"))["test"]
    galleria, query = protocollo(recs)
    out = {}
    for percorso in sys.argv[1:]:
        b = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=percorso == "dinov2", img_size=LATO_GLOBALE,
                              num_classes=0).to(device).eval()
        if percorso != "dinov2":
            b.load_state_dict(torch.load(percorso, map_location="cpu", weights_only=False)["backbone"])
        E = embedding(lambda x: b(x), foto(recs, LATO_GLOBALE), device)
        out[percorso] = analisi(E, recs, galleria, query)
        print(percorso, json.dumps(out[percorso], ensure_ascii=False), flush=True)
        del b
        torch.cuda.empty_cache()
    os.makedirs(os.path.join("runs", "controllo_sessione"), exist_ok=True)
    with open(os.path.join("runs", "controllo_sessione", "risultati.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
