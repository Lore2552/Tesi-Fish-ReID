"""
Come cambia l'embedding di un pesce tra due catture nello spazio di un backbone addestrato, e quanto di quel cambiamento
si prevede da tempo, crescita, sesso e stato riproduttivo. Solo pesci di validation e test, mai visti in training.
  ampiezza  : d = 1 - coseno tra le foto di due catture, per intervallo; riferimenti: stessa cattura, pesci diversi
  previsione: quota di varianza spiegata fuori campione (5 fold per pesce) di d e del vettore e_dopo - e_prima

    python analisi_cambiamento.py                                   output: runs/cambiamento_<run>/
    python analisi_cambiamento.py --backbone dinov2
"""

import os
import json
import argparse
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dataset import build_records
from split_fisso import carica_split
from reid_aliked import metadati, FASCE
from riferimenti_globali import embedding, foto
from reid_locale import sveglio
from lejepa_multivista import LATO_GLOBALE

GRUPPI = ("tempo", "crescita", "sesso", "riproduzione")
CHIAVE_PESCI = "(pesce, lato) con piu' catture"


def giorno(r):
    return datetime.strptime(r["date"], "%d.%m.%Y").toordinal()


def coppie(records):
    """Coppie (i, j, peso) dello stesso pesce e lato tra date diverse (i prima di j) e nella stessa data.
    Ogni (pesce, lato) pesa 1 in totale."""
    gruppi = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(records):
        gruppi[(r["fish_id"], r["side"])][giorno(r)].append(i)
    tra, stessa = [], []
    for date in gruppi.values():
        ordine = sorted(date)
        c = [(i, j) for a, da in enumerate(ordine) for db in ordine[a + 1:] for i in date[da] for j in date[db]]
        s = [(i, j) for d in ordine for k, i in enumerate(date[d]) for j in date[d][k + 1:]]
        tra += [(i, j, 1 / len(c)) for i, j in c]
        stessa += [(i, j, 1 / len(s)) for i, j in s]
    tra, stessa = np.array(tra).reshape(-1, 3), np.array(stessa).reshape(-1, 3)
    return (tra[:, 0].astype(int), tra[:, 1].astype(int), tra[:, 2]), (stessa[:, 0].astype(int), stessa[:, 1].astype(int), stessa[:, 2])


def impostori(records, n, seed=0):
    ids = np.array([r["fish_id"] for r in records])
    lati = np.array([r["side"] for r in records])
    rng = np.random.default_rng(seed)
    i, j = rng.integers(0, len(records), 4 * n), rng.integers(0, len(records), 4 * n)
    ok = (ids[i] != ids[j]) & (lati[i] == lati[j])
    return i[ok][:n], j[ok][:n]


def caratteristiche(records, meta, s, t):
    """Passaggio dalla foto s alla foto t, per gruppo."""
    g = np.array([giorno(r) for r in records], dtype=float)
    mese = np.array([int(r["date"].split(".")[1]) for r in records])
    lung = np.array([meta[r["stem"]]["length"] for r in records]) / 10
    sesso = np.array([meta[r["stem"]]["sex"] for r in records])
    ripr = np.array([int(meta[r["stem"]]["spawning"]) for r in records])
    ang = lambda m: np.column_stack([np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)])
    uno = lambda v, valori: (v[:, None] == np.array(valori)[None]).astype(float)
    return {"tempo": np.column_stack([(g[t] - g[s]) / 365.25, ang(mese[s]), ang(mese[t])]),
            "crescita": np.column_stack([lung[t] - lung[s], lung[s]]),
            "sesso": np.column_stack([uno(sesso[s], list("mfs")), sesso[s] != sesso[t]]),
            "riproduzione": np.column_stack([uno(ripr[s], (0, 1, 2)), uno(ripr[t], (0, 1, 2))])}


def ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13)))


def fuori_fold(nuovo, X, Y, w, pesci):
    """Predizioni fuori fold e, come riferimento, la media pesata del fold di training."""
    P, B = np.zeros_like(Y), np.zeros_like(Y)
    for a, b in GroupKFold(5).split(X, Y, pesci):
        m = nuovo()
        chiave = "ridgecv__sample_weight" if hasattr(m, "steps") else "sample_weight"
        P[b] = m.fit(X[a], Y[a], **{chiave: w[a]}).predict(X[b])
        B[b] = np.average(Y[a], axis=0, weights=w[a])
    return P, B


def varianza_spiegata(Y, P, B, w, pesci, n=2000, seed=0):
    """100 * (1 - SSE / SSE della media), con intervallo bootstrap al 95% sui pesci."""
    em = w * ((Y - P) ** 2).reshape(len(Y), -1).sum(1)
    eb = w * ((Y - B) ** 2).reshape(len(Y), -1).sum(1)
    _, inv = np.unique(pesci, return_inverse=True)
    sm, sb = np.bincount(inv, em), np.bincount(inv, eb)
    k = np.random.default_rng(seed).integers(0, len(sm), (n, len(sm)))
    boot = 1 - sm[k].sum(1) / sb[k].sum(1)
    return [round(100 * (1 - sm.sum() / sb.sum()), 1), round(100 * np.percentile(boot, 2.5), 1),
            round(100 * np.percentile(boot, 97.5), 1)]


def main():
    ap = argparse.ArgumentParser(description="Cambiamento dell'embedding tra catture")
    ap.add_argument("--backbone", default=os.path.join("runs", "arcface_dinov2_catture_prototipi", "migliore.pth"),
                    help="checkpoint con chiave 'backbone', oppure dinov2")
    args = ap.parse_args()
    sveglio()
    device = torch.device("cuda")
    nome = "dinov2" if args.backbone == "dinov2" else os.path.basename(os.path.dirname(args.backbone))
    out_dir = os.path.join("runs", f"cambiamento_{nome}")
    os.makedirs(out_dir, exist_ok=True)
    import timm

    insiemi = carica_split(build_records("head"))
    recs = insiemi["validation"] + insiemi["test"]
    meta = metadati()
    backbone = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=args.backbone == "dinov2",
                                 img_size=LATO_GLOBALE, num_classes=0).to(device).eval()
    if args.backbone != "dinov2":
        backbone.load_state_dict(torch.load(args.backbone, map_location="cpu", weights_only=False)["backbone"])
    E = embedding(lambda x: backbone(x), foto(recs, LATO_GLOBALE), device).astype(np.float64)
    del backbone
    torch.cuda.empty_cache()

    (s, t, w), (s0, t0, w0) = coppie(recs)
    C = caratteristiche(recs, meta, s, t)
    ok = ~np.isnan(C["crescita"]).any(1)
    s, t, w = s[ok], t[ok], w[ok]
    C = {k: v[ok] for k, v in C.items()}
    pesci = np.array([recs[i]["fish_id"] for i in s])
    d = 1 - (E[s] * E[t]).sum(1)
    d0 = 1 - (E[s0] * E[t0]).sum(1)
    si, ti = impostori(recs, 20000)
    di = 1 - (E[si] * E[ti]).sum(1)
    giorni = (np.array([giorno(recs[i]) for i in t]) - np.array([giorno(recs[i]) for i in s]))
    media = lambda v, p: round(float(np.average(v, weights=p)), 4)

    out = {"backbone": args.backbone, "foto": len(recs), "coppie tra catture": int(len(s)),
           CHIAVE_PESCI: int(len({(recs[i]["fish_id"], recs[i]["side"]) for i in s})),
           "coppie nella stessa cattura": int(len(s0)),
           "d medio": {"stessa cattura": media(d0, w0), "tra catture": media(d, w), "pesci diversi": round(float(di.mean()), 4)},
           "d per intervallo": {}}
    for n, lo, hi in FASCE:
        sel = (giorni >= lo) & (giorni <= hi)
        if sel.any():
            out["d per intervallo"][n] = {"coppie": int(sel.sum()), "pesci": int(len(set(pesci[sel]))), "d": media(d[sel], w[sel])}
    rho_t = spearmanr(giorni, d)
    rho_l = spearmanr(np.abs(C["crescita"][:, 0]), d)
    out["correlazione di Spearman con d"] = {"intervallo": [round(float(rho_t.statistic), 3), float(rho_t.pvalue)],
                                            "|crescita|": [round(float(rho_l.statistic), 3), float(rho_l.pvalue)]}

    insiemi_X = {k: C[k] for k in GRUPPI}
    insiemi_X["tutti i metadati"] = np.column_stack([C[k] for k in GRUPPI])
    insiemi_X["tempo + crescita"] = np.column_stack([C["tempo"], C["crescita"]])
    out["varianza spiegata di d (%)"] = {}
    for k, X in insiemi_X.items():
        for nome_m, nuovo in (("lineare", ridge), ("alberi", lambda: HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05))):
            P, B = fuori_fold(nuovo, X, d, w, pesci)
            out["varianza spiegata di d (%)"][f"{k}, {nome_m}"] = varianza_spiegata(d, P, B, w, pesci)

    D = E[t] - E[s]
    ingressi = {"tutti i metadati": insiemi_X["tutti i metadati"], "embedding di partenza": E[s],
                "embedding di partenza + tutti i metadati": np.column_stack([E[s], insiemi_X["tutti i metadati"]])}
    for k in GRUPPI:
        ingressi[f"embedding di partenza + {k}"] = np.column_stack([E[s], C[k]])
    out["varianza spiegata di e_dopo - e_prima (%, lineare)"] = {}
    for k, X in ingressi.items():
        P, B = fuori_fold(ridge, X, D, w, pesci)
        out["varianza spiegata di e_dopo - e_prima (%, lineare)"][k] = varianza_spiegata(D, P, B, w, pesci)

    righe = [f"Backbone {args.backbone}; validation + test, {len(recs)} foto",
             f"{out['coppie tra catture']} coppie tra catture ({out[CHIAVE_PESCI]} (pesce, lato)), "
             f"{len(s0)} coppie nella stessa cattura", "",
             "d = 1 - coseno, media pesata per (pesce, lato):"]
    righe += [f"  {k}: {v:.4f}" for k, v in out["d medio"].items()]
    righe += [f"  {k}: {v['d']:.4f} ({v['coppie']} coppie, {v['pesci']} pesci)" for k, v in out["d per intervallo"].items()]
    righe += [f"  Spearman con l'intervallo {rho_t.statistic:.3f} (p {rho_t.pvalue:.1e}), con |crescita| {rho_l.statistic:.3f} "
              f"(p {rho_l.pvalue:.1e})", "", "Varianza spiegata di d fuori campione (%, IC 95%):"]
    righe += [f"  {k}: {v[0]:.1f} ({v[1]:.1f} .. {v[2]:.1f})" for k, v in out["varianza spiegata di d (%)"].items()]
    righe += ["", "Varianza spiegata di e_dopo - e_prima fuori campione (%, lineare, IC 95%):"]
    righe += [f"  {k}: {v[0]:.1f} ({v[1]:.1f} .. {v[2]:.1f})" for k, v in out["varianza spiegata di e_dopo - e_prima (%, lineare)"].items()]
    testo = "\n".join(righe)
    print(testo)
    with open(os.path.join(out_dir, "risultati.txt"), "w", encoding="utf-8") as f:
        f.write(testo + "\n")
    with open(os.path.join(out_dir, "risultati.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
