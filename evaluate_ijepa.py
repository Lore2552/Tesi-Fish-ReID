import os
import argparse

# --- Caricamento chiavi ---
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

from ijepa_model import build_ijepa
from dataset import GalleryDataset, build_images_by_id, load_metadata, split_ids_by_identity
from run_manager import get_run_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Valutazione Re-ID con backbone I-JEPA")
    parser.add_argument('--run', type=int, required=True,
                        help="Numero della run da valutare")
    parser.add_argument('--checkpoint', type=str, default='best',
                        help="Checkpoint: 'best' o numero epoca (es. '5')")
    parser.add_argument('--mode', type=str, default='pretrain', choices=['pretrain', 'probe'],
                        help="'pretrain': usa il backbone I-JEPA diretto. 'probe': usa il modello dopo linear probe.")
    parser.add_argument('--part', type=str, default='head')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def extract_features_pretrain(model, dataloader, device):
    """
    Estrae feature dal backbone I-JEPA puro (context_encoder).
    Usa Global Average Pooling sui patch token.
    """
    model.eval()
    all_embeddings = []
    all_labels = []
    num_patches = model.context_encoder.patch_embed.num_patches

    with torch.no_grad():
        for imgs, fish_ids in tqdm(dataloader, desc="Estrazione feature (pretrain)"):
            imgs = imgs.to(device)

            # Passa tutte le patch (nessuna maschera in valutazione)
            x = model.context_encoder.patch_embed(imgs)
            pos_embed = model.context_encoder.pos_embed[:, 1:, :]
            x = x + pos_embed
            cls_token = model.context_encoder.cls_token.expand(x.shape[0], -1, -1)
            cls_pos = model.context_encoder.pos_embed[:, :1, :]
            cls_token = cls_token + cls_pos
            x = torch.cat([cls_token, x], dim=1)
            x = model.context_encoder.pos_drop(x)
            for blk in model.context_encoder.blocks:
                x = blk(x)
            x = model.context_encoder.norm(x)

            # Global Average Pooling (escluso CLS token)
            features = x[:, 1:, :].mean(dim=1)
            features = F.normalize(features, p=2, dim=1)

            all_embeddings.append(features.cpu())
            all_labels.extend(fish_ids)

    return torch.cat(all_embeddings, dim=0), all_labels


def extract_features_probe(model, dataloader, device):
    """
    Estrae feature dal modello dopo linear probe/fine-tuning.
    Usa il metodo extract_features del modello IJEPAWithHead.
    """
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for imgs, fish_ids in tqdm(dataloader, desc="Estrazione feature (probe)"):
            imgs = imgs.to(device)
            features = model.extract_features(imgs)
            all_embeddings.append(features.cpu())
            all_labels.extend(fish_ids)

    return torch.cat(all_embeddings, dim=0), all_labels


def compute_cosine_similarity_matrix(query_emb, gallery_emb):
    query_emb = F.normalize(query_emb, p=2, dim=1)
    gallery_emb = F.normalize(gallery_emb, p=2, dim=1)
    return torch.mm(query_emb, gallery_emb.T)


def compute_reid_metrics(sim_matrix, query_labels, gallery_labels):
    n_query = sim_matrix.shape[0]
    gallery_labels_arr = np.array(gallery_labels)
    all_ap = []
    cmc_accum = np.zeros(len(gallery_labels))

    for q_idx in range(n_query):
        q_label = query_labels[q_idx]
        sims = sim_matrix[q_idx].numpy()
        sorted_indices = np.argsort(-sims)
        sorted_labels = gallery_labels_arr[sorted_indices]
        matches = (sorted_labels == q_label).astype(int)
        if matches.sum() == 0:
            continue
        cmc_accum[:len(matches)] += np.cumsum(matches) > 0
        num_correct = 0
        ap = 0.0
        for rank, is_match in enumerate(matches):
            if is_match:
                num_correct += 1
                precision_at_rank = num_correct / (rank + 1)
                ap += precision_at_rank
        ap /= matches.sum()
        all_ap.append(ap)

    n_valid = len(all_ap)
    cmc_curve = (cmc_accum / n_valid).tolist() if n_valid > 0 else []
    rank1 = cmc_curve[0] if len(cmc_curve) > 0 else 0.0
    rank5 = cmc_curve[4] if len(cmc_curve) >= 5 else (cmc_curve[-1] if cmc_curve else 0.0)
    mAP = float(np.mean(all_ap)) if all_ap else 0.0
    return rank1, rank5, mAP, cmc_curve


def plot_cmc_curve(cmc_curve, part, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    max_rank = min(len(cmc_curve), 20)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_rank + 1), cmc_curve[:max_rank],
             marker='o', linewidth=2, color='coral', label=f'CMC I-JEPA ({part})')
    plt.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8)
    plt.xlabel('Rank', fontsize=13)
    plt.ylabel('Identification Rate', fontsize=13)
    plt.title(f'CMC Curve I-JEPA - {part.upper()}', fontsize=14)
    plt.xticks(range(1, max_rank + 1))
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"cmc_curve_ijepa_{part}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"CMC Curve salvata: {out_path}")


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_dir = get_run_dir(args.run)
    results_dir = os.path.join(run_dir, "results")

    # --- Carica checkpoint ---
    if args.mode == 'pretrain':
        if args.checkpoint == 'best':
            ckpt_path = os.path.join(run_dir, "models", "ijepa_best.pth")
        else:
            ep = int(args.checkpoint)
            ckpt_path = os.path.join(run_dir, "models", f"ijepa_epoch_{ep:03d}.pth")
    else:  # probe
        if args.checkpoint == 'best':
            ckpt_path = os.path.join(run_dir, "models", "probe_best.pth")
        else:
            ep = int(args.checkpoint)
            ckpt_path = os.path.join(run_dir, "models", f"probe_epoch_{ep:03d}.pth")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint non trovato: {ckpt_path}")

    # --- Split ---
    folder_name = f"Melops_{args.part}"
    data_dir = os.path.join("dataset", folder_name, folder_name)
    filename_to_id = load_metadata("Melops_metadata.txt")
    all_images_by_id, _ = build_images_by_id(data_dir, filename_to_id)
    train_ids, test_ids = split_ids_by_identity(
        all_images_by_id, test_ratio=args.test_ratio, seed=args.seed
    )

    test_ids_valid = {fid for fid in test_ids
                      if fid in all_images_by_id and len(all_images_by_id[fid]) >= 2}

    query_samples = []
    gallery_samples = []
    for fid in sorted(test_ids_valid):
        imgs = sorted(all_images_by_id[fid])
        query_samples.append((imgs[0], fid))
        gallery_samples.extend([(p, fid) for p in imgs[1:]])

    print("=" * 60)
    print(f"  Re-ID Evaluation [{args.mode.upper()}] - Parte: {args.part.upper()}")
    print(f"  Checkpoint: {ckpt_path}")
    print("=" * 60)
    print(f"  Identita' test valide: {len(test_ids_valid)}")
    print(f"  Query: {len(query_samples)} | Gallery: {len(gallery_samples)}")
    print("=" * 60)

    # --- Carica modello ---
    if args.mode == 'pretrain':
        model = build_ijepa()
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device)
        extract_fn = extract_features_pretrain
    else:
        # Per il linear probe, dobbiamo ricreare IJEPAWithHead
        from train_linear_probe import IJEPAWithHead
        ijepa = build_ijepa()
        # Numero di classi = numero identita' di train
        num_classes = len(train_ids)
        model = IJEPAWithHead(ijepa, num_classes=num_classes, freeze_backbone=True)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device)
        extract_fn = extract_features_probe

    print("Modello caricato.")

    # --- Dataset ---
    query_dataset = GalleryDataset(part=args.part, id_subset={fid for _, fid in query_samples})
    query_dataset.samples = query_samples
    gallery_dataset = GalleryDataset(part=args.part, id_subset={fid for _, fid in gallery_samples})
    gallery_dataset.samples = gallery_samples

    query_loader = DataLoader(query_dataset, batch_size=args.batch_size, shuffle=False)
    gallery_loader = DataLoader(gallery_dataset, batch_size=args.batch_size, shuffle=False)

    # --- Estrazione e valutazione ---
    query_emb, query_labels = extract_fn(model, query_loader, device)
    gallery_emb, gallery_labels = extract_fn(model, gallery_loader, device)

    sim_matrix = compute_cosine_similarity_matrix(query_emb, gallery_emb)
    rank1, rank5, mAP, cmc_curve = compute_reid_metrics(sim_matrix, query_labels, gallery_labels)

    print("\n" + "=" * 60)
    print(f"  RISULTATI I-JEPA [{args.mode.upper()}]")
    print("=" * 60)
    print(f"  Rank-1 Accuracy : {rank1 * 100:.2f}%")
    print(f"  Rank-5 Accuracy : {rank5 * 100:.2f}%")
    print(f"  mAP             : {mAP * 100:.2f}%")
    print("=" * 60)

    plot_cmc_curve(cmc_curve, args.part, results_dir)

    results_path = os.path.join(results_dir, f"eval_metrics_ijepa_{args.mode}.txt")
    os.makedirs(results_dir, exist_ok=True)
    with open(results_path, 'w') as f:
        f.write(f"Modello: {ckpt_path} (I-JEPA {args.mode})\n")
        f.write(f"Rank-1 Accuracy : {rank1 * 100:.2f}%\n")
        f.write(f"Rank-5 Accuracy : {rank5 * 100:.2f}%\n")
        f.write(f"mAP             : {mAP * 100:.2f}%\n")
    print(f"Metriche salvate in: {results_path}")


if __name__ == '__main__':
    main()
