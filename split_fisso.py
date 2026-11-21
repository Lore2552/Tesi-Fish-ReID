"""
Split fisso per individuo: train / validation / test = 70 / 10 / 20% dei pesci, seme 42, stratificato per numero di
date di cattura (1, 2, 3+). Tutte le foto di un pesce stanno nello stesso insieme.

    python split_fisso.py                                          crea split_individui.csv (seme 42)
    python split_fisso.py --seme 43 --file split_individui_43.csv  un altro split per individuo
    python split_fisso.py --zona_test B --file split_zona_B.csv     test = i pesci catturati nella zona B, gli altri
                                                                   90 / 10 in train e validation
Gli altri script leggono il file indicato dalla variabile d'ambiente SPLIT_INDIVIDUI (default split_individui.csv).
"""

import os
import csv
import random
from collections import defaultdict

FILE = os.environ.get("SPLIT_INDIVIDUI", "split_individui.csv")
QUOTE = (("train", 0.7), ("validation", 0.1), ("test", 0.2))
SEED = 42


def crea(records, seme=SEED, file=FILE):
    date = defaultdict(set)
    for r in records:
        date[r["fish_id"]].add(r["date"])
    gruppi = defaultdict(list)
    for f in sorted(date):
        gruppi[min(len(date[f]), 3)].append(f)
    rng = random.Random(seme)
    insieme = {}
    for g in sorted(gruppi):
        pesci = gruppi[g]
        rng.shuffle(pesci)
        n_train, n_val = round(len(pesci) * QUOTE[0][1]), round(len(pesci) * QUOTE[1][1])
        for i, f in enumerate(pesci):
            insieme[f] = "train" if i < n_train else "validation" if i < n_train + n_val else "test"
    with open(file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fish_id", "insieme"])
        for pesce in sorted(insieme):
            w.writerow([pesce, insieme[pesce]])
    return insieme


def crea_per_zona(records, zona, seme=SEED, file=FILE):
    """Test = tutti i pesci con almeno una foto nella zona; gli altri in train / validation 90 / 10, stratificati."""
    area = {}
    with open("Melops_metadata.txt", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            area[r["filename_year"]] = r["area"]
    date, in_zona = defaultdict(set), set()
    for r in records:
        date[r["fish_id"]].add(r["date"])
        if area.get(r["stem"]) == zona:
            in_zona.add(r["fish_id"])
    gruppi = defaultdict(list)
    for f in sorted(date):
        if f not in in_zona:
            gruppi[min(len(date[f]), 3)].append(f)
    rng = random.Random(seme)
    insieme = {f: "test" for f in in_zona}
    for g in sorted(gruppi):
        pesci = gruppi[g]
        rng.shuffle(pesci)
        n_train = round(len(pesci) * 0.9)
        for i, f in enumerate(pesci):
            insieme[f] = "train" if i < n_train else "validation"
    with open(file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fish_id", "insieme"])
        for pesce in sorted(insieme):
            w.writerow([pesce, insieme[pesce]])
    return insieme


def leggi(file=None):
    with open(file or FILE, encoding="utf-8") as f:
        return {r["fish_id"]: r["insieme"] for r in csv.DictReader(f)}


def carica_split(records, file=None):
    insieme = leggi(file)
    mancanti = {r["fish_id"] for r in records} - set(insieme)
    if mancanti:
        raise ValueError(f"{len(mancanti)} pesci non sono in {file or FILE}, per esempio {sorted(mancanti)[:3]}")
    out = {nome: [] for nome, _ in QUOTE}
    for r in records:
        out[insieme[r["fish_id"]]].append(r)
    return out


def main():
    import argparse
    from dataset import build_records
    from reid import protocollo
    ap = argparse.ArgumentParser(description="Split per individuo")
    ap.add_argument("--seme", type=int, default=SEED)
    ap.add_argument("--file", default=FILE)
    ap.add_argument("--zona_test", default="", help="zona tenuta fuori come test (S, B, L, Flodevigen)")
    args = ap.parse_args()
    records = build_records("head")
    if os.path.exists(args.file):
        print(f"{args.file} esiste gia': non lo cambio")
    else:
        if args.zona_test:
            crea_per_zona(records, args.zona_test, args.seme, args.file)
        else:
            crea(records, args.seme, args.file)
        print(f"creato {args.file}")
    for nome, recs in carica_split(records, args.file).items():
        date = defaultdict(set)
        for r in recs:
            date[r["fish_id"]].add(r["date"])
        galleria, query = protocollo(recs)
        print(f"{nome}: {len(date)} pesci, {len(recs)} foto, {sum(len(d) >= 2 for d in date.values())} con almeno 2 date, "
              f"{len(query)} query, galleria {len(galleria)} foto")


if __name__ == "__main__":
    main()
