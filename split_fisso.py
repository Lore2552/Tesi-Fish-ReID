"""
Split fisso per individuo: train / validation / test = 70 / 10 / 20% dei pesci, seme 42, stratificato per numero di
date di cattura (1, 2, 3+). Tutte le foto di un pesce stanno nello stesso insieme.

    python split_fisso.py              crea il csv dello split
"""

import os
import csv
import random
from collections import defaultdict

FILE = "split_individui.csv"
QUOTE = (("train", 0.7), ("validation", 0.1), ("test", 0.2))
SEED = 42


def crea(records):
    date = defaultdict(set)
    for r in records:
        date[r["fish_id"]].add(r["date"])
    gruppi = defaultdict(list)
    for f in sorted(date):
        gruppi[min(len(date[f]), 3)].append(f)
    rng = random.Random(SEED)
    insieme = {}
    for g in sorted(gruppi):
        pesci = gruppi[g]
        rng.shuffle(pesci)
        n_train, n_val = round(len(pesci) * QUOTE[0][1]), round(len(pesci) * QUOTE[1][1])
        for i, f in enumerate(pesci):
            insieme[f] = "train" if i < n_train else "validation" if i < n_train + n_val else "test"
    with open(FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fish_id", "insieme"])
        for pesce in sorted(insieme):
            w.writerow([pesce, insieme[pesce]])
    return insieme


def leggi():
    with open(FILE, encoding="utf-8") as f:
        return {r["fish_id"]: r["insieme"] for r in csv.DictReader(f)}


def carica_split(records):
    insieme = leggi()
    mancanti = {r["fish_id"] for r in records} - set(insieme)
    if mancanti:
        raise ValueError(f"{len(mancanti)} pesci non sono in {FILE}, per esempio {sorted(mancanti)[:3]}")
    out = {nome: [] for nome, _ in QUOTE}
    for r in records:
        out[insieme[r["fish_id"]]].append(r)
    return out


def main():
    from dataset import build_records
    from reid import protocollo
    records = build_records("head")
    if os.path.exists(FILE):
        print(f"{FILE} esiste gia': non lo cambio")
    else:
        crea(records)
        print(f"creato {FILE}")
    for nome, recs in carica_split(records).items():
        date = defaultdict(set)
        for r in recs:
            date[r["fish_id"]].add(r["date"])
        galleria, query = protocollo(recs)
        print(f"{nome}: {len(date)} pesci, {len(recs)} foto, {sum(len(d) >= 2 for d in date.values())} con almeno 2 date, "
              f"{len(query)} query, galleria {len(galleria)} foto")


if __name__ == "__main__":
    main()
