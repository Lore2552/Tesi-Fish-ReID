"""
visualize_ijepa.py — Visualizzazione di cosa guarda il modello I-JEPA

Genera 3 tipi di output:
  1. Attention Rollout Heatmaps: mostra dove il modello focalizza l'attenzione
  2. Top-K Matching: per ogni query mostra i match piu' simili nella gallery
  3. Confronto: heatmap di due foto dello stesso pesce per vedere se guarda le stesse zone
"""
import os
import sys
import argparse
import random

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
import cv2
from PIL import Image
import torchvision.transforms as transforms

from ijepa_model import build_ijepa
from attention_rollout import rollout, generate_visualization
from dataset import load_metadata, build_images_by_id, split_ids_by_identity
from run_manager import get_run_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Visualizzazione I-JEPA")
    parser.add_argument('--run', type=int, required=True)
    parser.add_argument('--part', type=str, default='head')
    parser.add_argument('--num_samples', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    return parser.parse_args()


def load_model(run_dir, device):
    """Carica il backbone I-JEPA dal checkpoint."""
    ckpt_path = os.path.join(run_dir, "models", "ijepa_best.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint non trovato: {ckpt_path}")
    
    model = build_ijepa()
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def extract_attention_maps(model, img_tensor, device):
    """
    Estrae le mappe di attenzione da tutti i layer del context encoder.
    Disabilita fused_attn e registra hook su attn_drop.
    """
    attentions = []
    hooks = []
    
    def get_attention(module, input, output):
        attentions.append(output.cpu())
    
    # Disabilita fused attention e registra hook
    for blk in model.context_encoder.blocks:
        blk.attn.fused_attn = False
        h = blk.attn.attn_drop.register_forward_hook(get_attention)
        hooks.append(h)
    
    # Forward pass completo (tutte le patch)
    with torch.no_grad():
        x = model.context_encoder.patch_embed(img_tensor)
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
        features = x[:, 1:, :].mean(dim=1)
        features = F.normalize(features, p=2, dim=1)
    
    # Rimuovi hook
    for h in hooks:
        h.remove()
    
    # Ripristina fused attention
    for blk in model.context_encoder.blocks:
        blk.attn.fused_attn = True
    
    return attentions, features


def extract_features_only(model, img_tensor):
    """Estrae feature senza attention maps (piu' veloce)."""
    with torch.no_grad():
        x = model.context_encoder.patch_embed(img_tensor)
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
        features = x[:, 1:, :].mean(dim=1)
        features = F.normalize(features, p=2, dim=1)
    return features


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_dir = get_run_dir(args.run)
    output_dir = os.path.join(run_dir, "visualizations")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print(f"  Visualizzazione I-JEPA — Run {args.run}")
    print("=" * 60)

    # Carica modello
    model = load_model(run_dir, device)
    print("Modello caricato.")

    # Trasformazione
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    transform_vis = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Split
    folder_name = f"Melops_{args.part}"
    data_dir = os.path.join("dataset", folder_name, folder_name)
    filename_to_id = load_metadata("Melops_metadata.txt")
    all_images_by_id, _ = build_images_by_id(data_dir, filename_to_id)
    _, test_ids = split_ids_by_identity(
        all_images_by_id, test_ratio=args.test_ratio, seed=args.seed
    )
    test_ids_valid = {fid for fid in test_ids
                      if fid in all_images_by_id and len(all_images_by_id[fid]) >= 2}

    # =========================================================
    # 1. ATTENTION ROLLOUT HEATMAPS
    # =========================================================
    print("\n--- 1. Generazione Attention Rollout Heatmaps ---")
    
    sample_ids = random.sample(sorted(test_ids_valid), min(args.num_samples, len(test_ids_valid)))
    
    for i, fid in enumerate(sample_ids):
        imgs = sorted(all_images_by_id[fid])
        img_path = imgs[0]
        
        img_pil = Image.open(img_path).convert('RGB')
        img_tensor = transform(img_pil).unsqueeze(0).to(device)
        img_vis = np.array(img_pil.resize((224, 224))) / 255.0
        
        attentions, _ = extract_attention_maps(model, img_tensor, device)
        
        if len(attentions) > 0:
            mask = rollout(attentions, discard_ratio=0.9, head_fusion="mean")
            cam = generate_visualization(img_vis, mask)
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(img_vis)
            axes[0].set_title(f"Originale\nID: {fid}", fontsize=11)
            axes[0].axis('off')
            
            mask_resized = cv2.resize(mask, (224, 224))
            axes[1].imshow(mask_resized, cmap='hot')
            axes[1].set_title("Mappa di Attenzione", fontsize=11)
            axes[1].axis('off')
            
            axes[2].imshow(cam)
            axes[2].set_title("Attenzione sovrapposta", fontsize=11)
            axes[2].axis('off')
            
            plt.suptitle(f"Attention Rollout — Pesce {fid}", fontsize=13, fontweight='bold')
            plt.tight_layout()
            out_path = os.path.join(output_dir, f"attention_{i+1}_id{fid}.png")
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Salvato: {out_path}")

    # =========================================================
    # 2. TOP-K MATCHING (Query -> Gallery)
    # =========================================================
    print("\n--- 2. Visualizzazione Top-K Matching ---")
    
    # Costruiamo query e gallery
    query_samples = []
    gallery_samples = []
    for fid in sorted(test_ids_valid):
        imgs = sorted(all_images_by_id[fid])
        query_samples.append((imgs[0], fid))
        gallery_samples.extend([(p, fid) for p in imgs[1:]])

    # Estraiamo feature per tutte le gallery
    print("  Estrazione feature gallery...")
    gallery_features = []
    gallery_labels = []
    for path, fid in gallery_samples:
        img_pil = Image.open(path).convert('RGB')
        img_tensor = transform(img_pil).unsqueeze(0).to(device)
        feat = extract_features_only(model, img_tensor)
        gallery_features.append(feat.cpu())
        gallery_labels.append(fid)
    gallery_features = torch.cat(gallery_features, dim=0)

    # Per ogni query campionata, mostra i top-5 match
    sample_queries = random.sample(query_samples, min(args.num_samples, len(query_samples)))
    
    for qi, (q_path, q_id) in enumerate(sample_queries):
        q_pil = Image.open(q_path).convert('RGB')
        q_tensor = transform(q_pil).unsqueeze(0).to(device)
        q_feat = extract_features_only(model, q_tensor).cpu()
        
        # Calcola similarita'
        sims = torch.mm(q_feat, gallery_features.T).squeeze(0)
        top_k = 5
        top_indices = torch.argsort(sims, descending=True)[:top_k]
        
        fig, axes = plt.subplots(1, top_k + 1, figsize=(3 * (top_k + 1), 3.5))
        
        # Query
        q_vis = np.array(q_pil.resize((224, 224))) / 255.0
        axes[0].imshow(q_vis)
        axes[0].set_title(f"QUERY\nID: {q_id}", fontsize=10, fontweight='bold', color='blue')
        axes[0].axis('off')
        axes[0].add_patch(plt.Rectangle((0, 0), 223, 223, fill=False, edgecolor='blue', linewidth=3))
        
        # Top-K gallery matches
        for k, idx in enumerate(top_indices):
            g_path, g_id = gallery_samples[idx.item()]
            g_pil = Image.open(g_path).convert('RGB')
            g_vis = np.array(g_pil.resize((224, 224))) / 255.0
            sim_val = sims[idx].item()
            
            is_correct = (g_id == q_id)
            color = 'green' if is_correct else 'red'
            symbol = 'V' if is_correct else 'X'
            
            axes[k+1].imshow(g_vis)
            axes[k+1].set_title(f"Rank {k+1} [{symbol}]\nID: {g_id}\nSim: {sim_val:.3f}",
                               fontsize=9, color=color, fontweight='bold')
            axes[k+1].axis('off')
            axes[k+1].add_patch(plt.Rectangle((0, 0), 223, 223, fill=False, edgecolor=color, linewidth=3))
        
        plt.suptitle(f"Query vs Gallery — Pesce {q_id}", fontsize=12, fontweight='bold')
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"matching_{qi+1}_id{q_id}.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Salvato: {out_path}")

    # =========================================================
    # 3. CONFRONTO ATTENZIONE STESSO PESCE
    # =========================================================
    print("\n--- 3. Confronto attenzione stesso pesce ---")
    
    # Scegliamo pesci con almeno 3 foto per un confronto significativo
    multi_ids = [fid for fid in test_ids_valid if len(all_images_by_id[fid]) >= 3]
    compare_ids = random.sample(multi_ids, min(4, len(multi_ids)))
    
    for ci, fid in enumerate(compare_ids):
        imgs = sorted(all_images_by_id[fid])[:3]  # max 3 foto
        
        fig, axes = plt.subplots(2, len(imgs), figsize=(5 * len(imgs), 10))
        
        for j, img_path in enumerate(imgs):
            img_pil = Image.open(img_path).convert('RGB')
            img_tensor = transform(img_pil).unsqueeze(0).to(device)
            img_vis = np.array(img_pil.resize((224, 224))) / 255.0
            
            attentions, _ = extract_attention_maps(model, img_tensor, device)
            
            if len(attentions) > 0:
                mask = rollout(attentions, discard_ratio=0.9, head_fusion="mean")
                cam = generate_visualization(img_vis, mask)
                
                axes[0, j].imshow(img_vis)
                axes[0, j].set_title(f"Foto {j+1}", fontsize=11)
                axes[0, j].axis('off')
                
                axes[1, j].imshow(cam)
                axes[1, j].set_title(f"Attenzione", fontsize=11)
                axes[1, j].axis('off')
        
        plt.suptitle(f"Stesso pesce (ID {fid}) — Il modello guarda le stesse zone?",
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"confronto_{ci+1}_id{fid}.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Salvato: {out_path}")

    print(f"\nTutte le visualizzazioni salvate in: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
