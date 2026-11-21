"""
World model con il backbone e i predittori di world_model_controllato.py (runs/world_model_controllato_<run>/), sul test.
  spostamento: quanto il predittore sposta l'embedding della galleria rispetto al cambiamento vero, per intervallo
  coerenza   : pesci con almeno tre date; previsione diretta 1 -> 3 contro due passi 1 -> 2 -> 3
  mai visti  : query (pesce presente in galleria) e foto di pesci con una sola data tolti dalla galleria;
               punteggio = coseno massimo con la galleria, senza e con il world model; AUROC

    python analisi_world_model.py --omega 0 0.5            output: risultati_analisi.txt/json nella stessa cartella
"""

import os
import json
import argparse
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from dataset import build_records
from split_fisso import carica_split
from reid import protocollo, bootstrap
from reid_aliked import metadati, FASCE
from lejepa_world_model import Info, caratteristiche, Predittore, punteggi_proiettati
from world_model_controllato import ruota, angoli


def giorno(r):
    return datetime.strptime(r["date"], "%d.%m.%Y").toordinal()


def media_ic(pesci, v):
    lo, hi = bootstrap(pesci, v)
    return [round(float(np.mean(v)), 4), round(float(lo), 4), round(float(hi), 4)]


@torch.no_grad()
def prevedi(pred, E, I, s, t, device):
    e = torch.from_numpy(E[s]).to(device)
    return pred(e, caratteristiche(I, np.asarray(s), np.asarray(t)).to(device)).cpu().numpy()


@torch.no_grad()
def spostamento(pred, E, I, recs, galleria, query, device):
    """Per ogni query e ogni foto vera in galleria: 1 - coseno tra galleria prevista e originale, e cambiamento vero."""
    ids = np.array([r["fish_id"] for r in recs])
    lati = np.array([r["side"] for r in recs])
    g = np.array(galleria)
    s, t = [], []
    for q in query:
        for i in g[(ids[g] == ids[q]) & (lati[g] == lati[q])]:
            s.append(i)
            t.append(q)
    s, t = np.array(s), np.array(t)
    P = prevedi(pred, E, I, s, t, device)
    giorni = I.giorno[t] - I.giorno[s]
    out = {}
    for n, lo, hi in FASCE:
        sel = (giorni >= lo) & (giorni <= hi)
        if sel.any():
            out[n] = {"coppie": int(sel.sum()),
                      "spostamento del predittore": round(float(np.mean(1 - (P[sel] * E[s[sel]]).sum(1))), 4),
                      "cambiamento vero": round(float(np.mean(1 - (E[t[sel]] * E[s[sel]]).sum(1))), 4),
                      "errore con predittore": round(float(np.mean(1 - (E[t[sel]] * P[sel]).sum(1))), 4)}
    return out


def terne(recs):
    """(i, j, k) una foto per data, tre date crescenti dello stesso pesce e lato."""
    gruppi = defaultdict(lambda: defaultdict(list))
    for n, r in enumerate(recs):
        gruppi[(r["fish_id"], r["side"])][giorno(r)].append(n)
    out = []
    for date in gruppi.values():
        if len(date) < 3:
            continue
        ordine = sorted(date)
        prima = [min(date[d], key=lambda n: recs[n]["stem"]) for d in ordine]
        out += [(prima[a], prima[b], prima[c]) for a in range(len(prima)) for b in range(a + 1, len(prima))
                for c in range(b + 1, len(prima))]
    return np.array(out)


def coerenza(pred, E, I, recs, device):
    T = terne(recs)
    i, j, k = T[:, 0], T[:, 1], T[:, 2]
    diretta = prevedi(pred, E, I, i, k, device)
    meta = prevedi(pred, E, I, i, j, device)
    with torch.no_grad():
        due = pred(torch.from_numpy(meta).to(device), caratteristiche(I, j, k).to(device)).cpu().numpy()
    pesci = [recs[n]["fish_id"] for n in i]
    cos = lambda a, b: (a * b).sum(1)
    return {"terne": int(len(T)), "pesci": len(set(pesci)),
            "coseno diretta - due passi": media_ic(pesci, cos(diretta, due)),
            "coseno con la terza cattura": {"senza predittore (prima cattura)": media_ic(pesci, cos(E[i], E[k])),
                                            "previsione diretta 1 -> 3": media_ic(pesci, cos(diretta, E[k])),
                                            "previsione in due passi 1 -> 2 -> 3": media_ic(pesci, cos(due, E[k])),
                                            "seconda cattura": media_ic(pesci, cos(E[j], E[k]))}}


def auroc_ic(y, s, pesci, n=1000, seed=0):
    pesci = np.array(pesci)
    unici = np.unique(pesci)
    per = {p: np.flatnonzero(pesci == p) for p in unici}
    rng = np.random.default_rng(seed)
    stime = []
    for _ in range(n):
        idx = np.concatenate([per[p] for p in rng.choice(unici, len(unici))])
        if len(set(y[idx])) == 2:
            stime.append(roc_auc_score(y[idx], s[idx]))
    return [round(100 * roc_auc_score(y, s), 2), round(100 * np.percentile(stime, 2.5), 2), round(100 * np.percentile(stime, 97.5), 2)]


def mai_visti(pred, E, I, recs, galleria, query, device, seed=0):
    """Pesci con una sola data come query, tolti dalla galleria, contro le query del protocollo."""
    ids = np.array([r["fish_id"] for r in recs])
    date = defaultdict(set)
    for r in recs:
        date[r["fish_id"]].add(r["date"])
    candidati = {}
    for n in galleria:
        r = recs[n]
        if len(date[r["fish_id"]]) == 1:
            candidati.setdefault((r["fish_id"], r["side"]), n)
    chiavi = sorted(candidati)
    rng = np.random.default_rng(seed)
    scelte = [chiavi[x] for x in rng.permutation(len(chiavi))[:len(query)]]
    nuovi = {f for f, _ in scelte}
    q_nuove = [candidati[c] for c in scelte]
    gal = [n for n in galleria if ids[n] not in nuovi]
    qs = list(query) + q_nuove
    y = np.array([1] * len(query) + [0] * len(q_nuove))
    pesci = [ids[n] for n in qs]
    out = {"query di pesci presenti": len(query), "query di pesci mai visti": len(q_nuove), "galleria": len(gal)}
    for nome, p in (("sola invarianza", None), ("world model", pred)):
        s = np.nanmax(punteggi_proiettati(p, E, I, recs, gal, qs, device), axis=1)
        soglia = np.percentile(s[y == 1], 5)
        out[nome] = {"AUROC (%)": auroc_ic(y, s, pesci),
                     "mai visti respinti accettando il 95% dei presenti (%)": round(100 * float(np.mean(s[y == 0] < soglia)), 1)}
    return out


def main():
    ap = argparse.ArgumentParser(description="World model: spostamento, coerenza su piu' passi, pesci mai visti")
    ap.add_argument("--cartella", default=os.path.join("runs", "world_model_controllato_arcface_dinov2_catture_prototipi"))
    ap.add_argument("--omega", type=float, nargs="+", default=[0, 0.5])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device)
    insiemi = carica_split(build_records("head"))
    meta = metadati()
    I_tr = Info(insiemi["train"], meta)
    recs = insiemi["test"]
    I = Info(recs, meta)
    galleria, query = protocollo(recs)
    giorno0 = min(Info(insiemi[n], meta).giorno.min() for n in ("train", "validation", "test"))
    lung_mediana = float(np.median(I_tr.lung))
    Q = np.linalg.qr(np.random.default_rng(0).standard_normal((384, 384)))[0].astype(np.float32)
    E0 = np.load(os.path.join(args.cartella, "embedding.npz"))["test"]

    out, righe = {}, [f"Analisi del world model, test ({len(query)} query), cartella {args.cartella}", ""]
    for omega in args.omega:
        E = ruota(E0, angoli(I, omega, giorno0, lung_mediana), Q)
        pred = Predittore().to(device).eval()
        pred.load_state_dict(torch.load(os.path.join(args.cartella, f"predittore_omega_{omega}.pth"), map_location="cpu",
                                        weights_only=False)["predittore"])
        o = {"spostamento": spostamento(pred, E, I, recs, galleria, query, device),
             "coerenza": coerenza(pred, E, I, recs, device),
             "mai visti": mai_visti(pred, E, I, recs, galleria, query, device)}
        out[str(omega)] = o
        righe.append(f"omega {omega} rad/anno" + (" (dati veri)" if omega == 0 else " (caso controllato)"))
        righe.append("  per intervallo (1 - coseno): spostamento del predittore | cambiamento vero (errore senza predittore) | errore con predittore")
        for n, v in o["spostamento"].items():
            righe.append(f"    {n} ({v['coppie']} coppie): {v['spostamento del predittore']:.4f} | {v['cambiamento vero']:.4f} | "
                         f"{v['errore con predittore']:.4f}")
        c = o["coerenza"]
        righe.append(f"  coerenza su piu' passi: {c['terne']} terne di {c['pesci']} pesci; coseno diretta - due passi "
                     f"{c['coseno diretta - due passi'][0]:.4f} ({c['coseno diretta - due passi'][1]:.4f}-{c['coseno diretta - due passi'][2]:.4f})")
        for k, v in c["coseno con la terza cattura"].items():
            righe.append(f"    coseno con la terza cattura, {k}: {v[0]:.4f} ({v[1]:.4f}-{v[2]:.4f})")
        m = o["mai visti"]
        righe.append(f"  pesci mai visti: {m['query di pesci presenti']} query presenti, {m['query di pesci mai visti']} mai visti, "
                     f"galleria {m['galleria']}")
        for k in ("sola invarianza", "world model"):
            a = m[k]["AUROC (%)"]
            righe.append(f"    {k}: AUROC {a[0]:.2f} ({a[1]:.2f}-{a[2]:.2f}) | mai visti respinti accettando il 95% dei presenti "
                         f"{m[k]['mai visti respinti accettando il 95% dei presenti (%)']:.1f}%")
        righe.append("")
        print("\n".join(righe), flush=True)
    with open(os.path.join(args.cartella, "risultati_analisi.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(righe) + "\n")
    with open(os.path.join(args.cartella, "risultati_analisi.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
