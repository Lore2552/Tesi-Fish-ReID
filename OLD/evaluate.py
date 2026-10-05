"""
evaluate.py — Valutazione Re-ID per il dataset Melops

Protocollo:
-----------
1. Lo split identita'/train/test viene ricalcolato con lo stesso seed del training
   (seed=42, test_ratio=0.2) per garantire che le identita' di test siano
   SEMPRE le stesse e MAI viste in training.

2. Per ogni identita' di test con >= 2 immagini:
   - Una immagine e' estratta come "query"
   - Le restanti formano la "gallery"

3. Il modello estrae gli embedding di query e gallery.

4. Il confronto avviene via COSINE SIMILARITY (non distanza euclidea):
   similarity(q, g) = dot(norm(q), norm(g))
   Poiche' gli embedding sono gia' normalizzati L2 nel modello,
   questo equivale semplicemente al prodotto scalare.

5. Metriche calcolate:
   - Rank-1 Accuracy: la gallery image piu' simile appartiene alla stessa identita'?
   - Rank-5 Accuracy: almeno una delle top-5 gallery images appartiene alla stessa identita'?
   - mAP (mean Average Precision): media delle AP su tutte le query
   - CMC Curve: salvata come immagine in results/

    python evaluate.py --part head --model_path siamese_vit_head_ep3.pth
"""

import os
import argparse

# Carica il token HF dal file .env
env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_path):
    with open(env_path, 'r') as f:
        for line in f:
            if line.startswith('HF_TOKEN'):
                token = line.split('=')[1].strip()
                os.environ["HF_TOKEN"] = token
                os.environ["HUGGING_FACE_HUB_TOKEN"] = token

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import (GalleryDataset, build_images_by_id,
                     load_metadata, split_ids_by_identity)
from model import SiameseViT
from run_manager import get_run_dir, save_config


def parse_args():
    parser = argparse.ArgumentParser(description="Valutazione Re-ID su Melops (Cosine Similarity)")
    parser.add_argument('--run', type=int, required=True,
                        help="Numero della run da valutare (es. 1 per run_001).")
    parser.add_argument('--epoch', type=int, required=True,
                        help="Epoca del modello da caricare (es. 3 per epoch_03.pth).")
    parser.add_argument('--part', type=str, default='head', choices=['head', 'body', 'full'],
                        help="Parte del pesce (head, body, full).")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size per l'estrazione degli embedding.")
    return parser.parse_args()


def extract_embeddings(model, dataloader, device):
    """
    Estrae gli embedding per tutte le immagini nel dataloader.
    Restituisce:
        embeddings: tensor (N, D) normalizzato L2
        labels:     lista di fish_id (stringa) per ogni sample
    """
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for imgs, fish_ids in tqdm(dataloader, desc="Estrazione embedding"):
            imgs = imgs.to(device)
            emb = model(imgs)  # forward_one -> (B, D) normalizzato L2
            all_embeddings.append(emb.cpu())
            all_labels.extend(fish_ids)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    return all_embeddings, all_labels


def compute_cosine_similarity_matrix(query_emb, gallery_emb):
    """
    Calcola la matrice di similarita' coseno tra query e gallery.
    Dato che gli embedding sono gia' normalizzati L2 nel modello,
    la cosine similarity coincide col prodotto scalare.

    Returns:
        sim_matrix: tensor (N_query, N_gallery), valori in [-1, 1]
    """
    query_emb = F.normalize(query_emb, p=2, dim=1)
    gallery_emb = F.normalize(gallery_emb, p=2, dim=1)
    return torch.mm(query_emb, gallery_emb.T)


def compute_reid_metrics(sim_matrix, query_labels, gallery_labels):
    """
    Calcola Rank-1, Rank-5, mAP e la CMC Curve completa.

    Protocollo:
    - Per ogni query, la gallery e' ordinata per similarita' decrescente.
    - Se la query appare anche nella gallery (stessa immagine), viene esclusa.
    - Il match e' corretto se il fish_id coincide.

    Returns:
        rank1, rank5, mAP: float
        cmc_curve: lista di float (lunghezza = max rank)
    """
    n_query = sim_matrix.shape[0]
    gallery_labels_arr = np.array(gallery_labels)

    all_ap = []
    cmc_accum = np.zeros(len(gallery_labels))

    for q_idx in range(n_query):
        q_label = query_labels[q_idx]
        sims = sim_matrix[q_idx].numpy()

        sorted_indices = np.argsort(-sims)
        sorted_labels = gallery_labels_arr[sorted_indices]

        # Rimuoviamo il caso in cui la query sia essa stessa nella gallery
        mask = np.ones(len(sorted_labels), dtype=bool)

        sorted_labels_filtered = sorted_labels[mask]
        matches = (sorted_labels_filtered == q_label).astype(int)

        if matches.sum() == 0:
            # Identita' non trovata in gallery: saltiamo (non dovrebbe accadere)
            continue

        # CMC: cumulativo dei match corretti
        cmc_accum[:len(matches)] += np.cumsum(matches) > 0

        # Average Precision per questa query
        num_correct = 0
        ap = 0.0
        for rank, is_match in enumerate(matches):
            if is_match:
                num_correct += 1
                precision_at_rank = num_correct / (rank + 1)
                ap += precision_at_rank
        ap /= matches.sum()
        all_ap.append(ap)

    n_valid_queries = len(all_ap)
    cmc_curve = (cmc_accum / n_valid_queries).tolist()

    rank1 = cmc_curve[0] if len(cmc_curve) > 0 else 0.0
    rank5 = cmc_curve[4] if len(cmc_curve) >= 5 else cmc_curve[-1]
    mAP = float(np.mean(all_ap)) if all_ap else 0.0

    return rank1, rank5, mAP, cmc_curve


def plot_cmc_curve(cmc_curve, part, model_path, output_dir="results"):
    """Salva la CMC Curve come immagine PNG."""
    os.makedirs(output_dir, exist_ok=True)
    max_rank = min(len(cmc_curve), 20)  # Plottiamo fino a Rank-20

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_rank + 1), cmc_curve[:max_rank],
             marker='o', linewidth=2, color='steelblue', label=f'CMC ({part})')
    plt.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8)
    plt.xlabel('Rank', fontsize=13)
    plt.ylabel('Identification Rate', fontsize=13)
    plt.title(f'CMC Curve — {part.upper()} | Modello: {os.path.basename(model_path)}', fontsize=14)
    plt.xticks(range(1, max_rank + 1))
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    model_stem = os.path.splitext(os.path.basename(model_path))[0]
    out_path = os.path.join(output_dir, f"cmc_curve_{part}_{model_stem}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"CMC Curve salvata: {out_path}")


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_dir = get_run_dir(args.run)
    results_dir = os.path.join(run_dir, "results")
    model_path = os.path.join(run_dir, "models", f"epoch_{args.epoch:02d}.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modello non trovato: {model_path}")

    print("=" * 60)
    print(f"  Re-ID Evaluation — Parte: {args.part.upper()}")
    print(f"  Run: run_{args.run:03d} | Epoch: {args.epoch}")
    print(f"  Modello: {model_path}")
    print(f"  Device:  {device}")
    print("=" * 60)


    folder_name = f"Melops_{args.part}"
    data_dir = os.path.join("dataset", folder_name, folder_name)
    filename_to_id = load_metadata("Melops_metadata.txt")
    all_images_by_id, _ = build_images_by_id(data_dir, filename_to_id)
    train_ids, test_ids = split_ids_by_identity(all_images_by_id, test_ratio=0.2, seed=42)

    test_ids_valid = {fid for fid in test_ids
                      if fid in all_images_by_id and len(all_images_by_id[fid]) >= 2}

    print(f"\nSplit identita': {len(train_ids)} train / {len(test_ids)} test")
    print(f"Identita' di test valide (>= 2 img): {len(test_ids_valid)}")

    # ------------------------------------------------------------------ #
    # 2. Costruiamo Query e Gallery dalle identita' di test
    #    - Query:   1 immagine per identita' (la prima in ordine)
    #    - Gallery: tutte le restanti immagini delle stesse identita'
    # ------------------------------------------------------------------ #
    query_samples = []   # lista di (path, fish_id)
    gallery_samples = [] # lista di (path, fish_id)

    for fid in sorted(test_ids_valid):
        imgs = sorted(all_images_by_id[fid])
        query_samples.append((imgs[0], fid))    # prima immagine = query
        gallery_samples.extend([(p, fid) for p in imgs[1:]])  # resto = gallery

    print(f"Query:   {len(query_samples)} immagini")
    print(f"Gallery: {len(gallery_samples)} immagini\n")

    # ------------------------------------------------------------------ #
    # 3. Carica il modello
    # ------------------------------------------------------------------ #
    model = SiameseViT(model_name='vit_base_patch16_224', pretrained=False, embedding_dim=256)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Modello caricato da: {model_path}")

    # ------------------------------------------------------------------ #
    # 4. Estrai embedding per query e gallery usando GalleryDataset
    # ------------------------------------------------------------------ #
    query_dataset = GalleryDataset(part=args.part,
                                   id_subset={fid for _, fid in query_samples})
    query_dataset.samples = query_samples

    gallery_dataset = GalleryDataset(part=args.part,
                                     id_subset={fid for _, fid in gallery_samples})
    gallery_dataset.samples = gallery_samples

    query_loader = DataLoader(query_dataset, batch_size=args.batch_size, shuffle=False)
    gallery_loader = DataLoader(gallery_dataset, batch_size=args.batch_size, shuffle=False)

    print("Estrazione embedding query...")
    query_emb, query_labels = extract_embeddings(model, query_loader, device)

    print("Estrazione embedding gallery...")
    gallery_emb, gallery_labels = extract_embeddings(model, gallery_loader, device)

    # ------------------------------------------------------------------ #
    # 5. Calcola la matrice di similarita' (Cosine Similarity)
    # ------------------------------------------------------------------ #
    print("\nCalcolo Cosine Similarity matrix...")
    sim_matrix = compute_cosine_similarity_matrix(query_emb, gallery_emb)

    # ------------------------------------------------------------------ #
    # 6. Calcola le metriche Re-ID
    # ------------------------------------------------------------------ #
    rank1, rank5, mAP, cmc_curve = compute_reid_metrics(
        sim_matrix, query_labels, gallery_labels
    )

    print("\n" + "=" * 60)
    print(f"  RISULTATI VALUTAZIONE — {args.part.upper()}")
    print("=" * 60)
    print(f"  Rank-1 Accuracy : {rank1 * 100:.2f}%")
    print(f"  Rank-5 Accuracy : {rank5 * 100:.2f}%")
    print(f"  mAP             : {mAP * 100:.2f}%")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # 7. Salva CMC Curve e risultati su file
    # ------------------------------------------------------------------ #
    plot_cmc_curve(cmc_curve, args.part, model_path, output_dir=results_dir)

    results_path = os.path.join(results_dir, f"eval_metrics_ep{args.epoch:02d}.txt")
    with open(results_path, 'w') as f:
        f.write(f"Run: run_{args.run:03d}\n")
        f.write(f"Modello: {model_path}\n")
        f.write(f"Parte: {args.part}\n")
        f.write(f"Split: {len(train_ids)} train IDs / {len(test_ids)} test IDs (seed=42)\n")
        f.write(f"Query: {len(query_samples)} | Gallery: {len(gallery_samples)}\n\n")
        f.write(f"Rank-1 Accuracy : {rank1 * 100:.2f}%\n")
        f.write(f"Rank-5 Accuracy : {rank5 * 100:.2f}%\n")
        f.write(f"mAP             : {mAP * 100:.2f}%\n\n")
        f.write("CMC Curve (rank: identification_rate):\n")
        for i, val in enumerate(cmc_curve[:20], 1):
            f.write(f"  Rank-{i:2d}: {val * 100:.2f}%\n")

    print(f"\nRisultati salvati: {results_path}")


if __name__ == '__main__':
    main()
