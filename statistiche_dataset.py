"""
Statistiche del dataset (foto di testa utilizzabili; cattura = individuo + data): totali, catture per individuo,
intervalli tra catture, anni e mesi, metadati, sesso e crescita tra catture dello stesso individuo.

    python statistiche_dataset.py      output: runs/dataset/ (statistiche.txt, statistiche.json, intervalli.png)
"""

import os
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import build_records

OUT_DIR = os.path.join("runs", "dataset")
MANCANTE = ("", "NA", "NaN", "nan")
FASCE = [("1-7 giorni", 1, 7), ("8-30 giorni", 8, 30), ("31-90 giorni", 31, 90), ("91-365 giorni", 91, 365),
         ("1-2 anni", 366, 730), ("oltre 2 anni", 731, 10 ** 6)]
CAMPI = ["length", "sex", "validated_sex", "spawning", "tagged", "suspected_tagloss", "sightings", "sightingnumb",
         "cryptocotyle", "scaleloss", "zombie", "area", "trapID", "lat", "lon"]


def giorno(s):
    return datetime.strptime(s, "%d.%m.%Y")


def fasce(giorni):
    g = np.asarray(giorni)
    return {nome: int(((g >= lo) & (g <= hi)).sum()) for nome, lo, hi in FASCE}


def quantili(v):
    v = np.asarray(v, dtype=float)
    return {"n": int(len(v)), "minimo": float(v.min()), "25%": float(np.percentile(v, 25)), "mediana": float(np.median(v)),
            "75%": float(np.percentile(v, 75)), "massimo": float(v.max())}


def main():
    meta = {}
    with open("Melops_metadata.txt", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            meta[row["filename_year"]] = row
    records = build_records("head")
    righe = [meta[r["stem"]] for r in records]

    catture = defaultdict(list)                         # (individuo, data) -> righe dei metadati
    for r in righe:
        catture[(r["ID"], r["date"])].append(r)
    date = defaultdict(list)
    for (ind, d) in catture:
        date[ind].append(giorno(d))
    for ind in date:
        date[ind].sort()

    out = {"totali": {"righe nei metadati": len(meta), "foto di testa utilizzabili": len(righe),
                      "foto escluse perche' mosse": sum(m["blurry"] == "1" for m in meta.values()),
                      "individui": len(date), "catture (individuo + data)": len(catture),
                      "foto per cattura": dict(sorted(Counter(len(v) for v in catture.values()).items())),
                      "catture con entrambi i lati": sum({x["side"] for x in v} == {"left", "right"} for v in catture.values())}}
    n_date = Counter(len(d) for d in date.values())
    multi = [ind for ind, d in date.items() if len(d) >= 2]
    out["catture per individuo"] = {"distribuzione": {str(k): v for k, v in sorted(n_date.items())},
                                    "individui con almeno 2 date": len(multi),
                                    "quota": round(100 * len(multi) / len(date), 1),
                                    "catture successive alla prima": sum(len(date[i]) - 1 for i in multi)}

    consecutivi = [(b - a).days for ind in multi for a, b in zip(date[ind][:-1], date[ind][1:])]
    ultima = [(date[ind][-1] - date[ind][-2]).days for ind in multi]
    estremi = [(date[ind][-1] - date[ind][0]).days for ind in multi]
    stesso_anno = sum(a.year == b.year for ind in multi for a, b in zip(date[ind][:-1], date[ind][1:]))
    out["intervalli (giorni)"] = {
        "tra catture consecutive": {**quantili(consecutivi), "fasce": fasce(consecutivi), "nello stesso anno": stesso_anno},
        "tra l'ultima e la precedente (query e galleria)": {**quantili(ultima), "fasce": fasce(ultima)},
        "tra la prima e l'ultima": {**quantili(estremi), "fasce": fasce(estremi)}}

    anni = Counter(giorno(d).year for (_, d) in catture)
    mesi = Counter(giorno(d).month for (_, d) in catture)
    out["anni e mesi delle catture"] = {"anni": dict(sorted(anni.items())), "mesi": dict(sorted(mesi.items()))}

    campi = {}
    for c in CAMPI:
        valori = [r[c] for r in righe]
        pieni = [v for v in valori if v not in MANCANTE]
        diversi_in_cattura = sum(len({x[c] for x in v if x[c] not in MANCANTE}) > 1 for v in catture.values())
        voce = {"foto con il valore": len(pieni), "quota %": round(100 * len(pieni) / len(valori), 1),
                "catture con valori diversi tra le foto": diversi_in_cattura}
        try:
            num = np.array([float(v) for v in pieni])
            if len(set(pieni)) > 12:
                voce["valori"] = quantili(num)
            else:
                voce["valori"] = dict(Counter(pieni).most_common())
        except ValueError:
            voce["valori"] = dict(Counter(pieni).most_common(12)) if len(set(pieni)) <= 40 else \
                {"valori distinti": len(set(pieni)), "piu' frequenti": dict(Counter(pieni).most_common(5))}
        campi[c] = voce
    out["metadati per foto"] = campi

    def valore(cat, campo):
        v = [x[campo] for x in cat if x[campo] not in MANCANTE]
        return Counter(v).most_common(1)[0][0] if v else None
    sesso_cambia, crescita, crescita_anno = 0, [], []
    passaggi = Counter()
    for ind in multi:
        cat = [catture[(ind, d.strftime("%d.%m.%Y"))] for d in date[ind]]
        sessi = [valore(c, "sex") for c in cat]
        sessi = [s for s in sessi if s]
        if len(set(sessi)) > 1:
            sesso_cambia += 1
            for a, b in zip(sessi[:-1], sessi[1:]):
                if a != b:
                    passaggi[f"{a} -> {b}"] += 1
        lung = [valore(c, "length") for c in cat]
        for (a, la), (b, lb) in zip(list(zip(date[ind], lung))[:-1], list(zip(date[ind], lung))[1:]):
            if la is not None and lb is not None:
                dl, dg = float(lb) - float(la), (b - a).days
                crescita.append(dl)
                if dg >= 180:
                    crescita_anno.append(365 * dl / dg)
    crescita = np.array(crescita)
    out["tra catture dello stesso individuo"] = {
        "individui il cui sesso cambia": sesso_cambia, "cambi di sesso": dict(passaggi),
        "crescita in lunghezza tra catture consecutive": {**quantili(crescita),
                                                         "quota negativa %": round(100 * float((crescita < 0).mean()), 1),
                                                         "quota zero %": round(100 * float((crescita == 0).mean()), 1)},
        "crescita per anno (intervalli di almeno 180 giorni)": quantili(crescita_anno) if crescita_anno else None}

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "statistiche.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

    def scrivi(d, rientro=0):
        lines = []
        for k, v in d.items():
            if isinstance(v, dict):
                lines.append(" " * rientro + f"{k}:")
                lines += scrivi(v, rientro + 2)
            else:
                lines.append(" " * rientro + f"{k}: {v:.1f}" if isinstance(v, float) else " " * rientro + f"{k}: {v}")
        return lines
    text = "\n".join(["Statistiche del dataset: foto di testa utilizzabili (su disco, nei metadati, non mosse)", ""] + scrivi(out))
    print(text)
    with open(os.path.join(OUT_DIR, "statistiche.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    bins = np.logspace(0, np.log10(max(consecutivi) + 1), 40)
    ax[0].hist(consecutivi, bins=bins, color="#1f4e79")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("giorni tra catture consecutive (scala logaritmica)")
    ax[0].set_ylabel("coppie di catture")
    ax[0].set_title("Intervalli tra catture consecutive")
    ax[1].bar([str(k) for k in sorted(n_date)], [n_date[k] for k in sorted(n_date)], color="#1f4e79")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("date diverse per individuo")
    ax[1].set_ylabel("individui (scala logaritmica)")
    ax[1].set_title("Catture per individuo")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "intervalli.png"), dpi=120)
    plt.close()


if __name__ == "__main__":
    main()
