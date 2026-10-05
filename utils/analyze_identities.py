"""
analyze_identities.py — Distribuzione immagini per identita' nel dataset Melops

Esegui con:
    py analyze_identities.py --part head
    py analyze_identities.py --part body
"""

import os
import sys
import argparse
from collections import Counter

# Aggiungi la root del progetto al path per importare i moduli
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dataset import build_images_by_id, load_metadata, split_ids_by_identity


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=str, default='head', choices=['head', 'body', 'full'])
    return parser.parse_args()


def main():
    args = parse_args()

    folder_name = f"Melops_{args.part}"
    data_dir = os.path.join("dataset", folder_name, folder_name)
    filename_to_id = load_metadata("Melops_metadata.txt")
    all_images_by_id, _ = build_images_by_id(data_dir, filename_to_id)
    train_ids, test_ids = split_ids_by_identity(all_images_by_id, test_ratio=0.2, seed=42)

    total_ids = len(all_images_by_id)
    total_images = sum(len(v) for v in all_images_by_id.values())
    img_counts = [len(v) for v in all_images_by_id.values()]

    print(f"\n{'='*55}")
    print(f"  Dataset: Melops_{args.part}")
    print(f"{'='*55}")
    print(f"  Identita' totali (pesci unici) : {total_ids}")
    print(f"  Immagini totali                : {total_images}")
    print(f"  Media immagini per pesce       : {total_images / total_ids:.2f}")
    print(f"  Min immagini per pesce         : {min(img_counts)}")
    print(f"  Max immagini per pesce         : {max(img_counts)}")
    print(f"\n  Split (seed=42, ratio=0.2):")
    print(f"    Train identita' : {len(train_ids)}")
    print(f"    Test  identita' : {len(test_ids)}")

    # Distribuzione per fasce
    print(f"\n{'='*55}")
    print(f"  Distribuzione immagini per identita':")
    print(f"{'='*55}")
    buckets = [
        ("1 immagine  (inutili per training)", lambda n: n == 1),
        ("2 immagini  (minimo utile)         ", lambda n: n == 2),
        ("3-5 immagini                       ", lambda n: 3 <= n <= 5),
        ("6-10 immagini                      ", lambda n: 6 <= n <= 10),
        ("11-20 immagini                     ", lambda n: 11 <= n <= 20),
        ("21-50 immagini                     ", lambda n: 21 <= n <= 50),
        ("Oltre 50 immagini (identita' ricca)", lambda n: n > 50),
    ]
    for label, condition in buckets:
        count = sum(1 for n in img_counts if condition(n))
        perc = count / total_ids * 100
        bar = "█" * int(perc / 2)
        print(f"  {label}: {count:5d} ({perc:5.1f}%)  {bar}")

    # Top 10 identita' con piu' immagini
    print(f"\n{'='*55}")
    print(f"  Top 10 identita' con piu' immagini:")
    print(f"{'='*55}")
    sorted_ids = sorted(all_images_by_id.items(), key=lambda x: len(x[1]), reverse=True)
    for fid, imgs in sorted_ids[:10]:
        split_tag = "TRAIN" if fid in train_ids else "TEST"
        print(f"  ID {fid:<10} : {len(imgs):4d} immagini  [{split_tag}]")

    print(f"\n{'='*55}\n")


if __name__ == '__main__':
    main()
