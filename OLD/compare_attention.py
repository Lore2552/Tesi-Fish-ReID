"""
compare_attention.py — Confronto visivo tra coppie di pesci con Heatmap e Cosine Similarity

Questo script risponde alla domanda scientifica:
    "Qual e' la feature che il modello usa per distinguere due pesci diversi?"

Per ogni coppia (o per coppie selezionate automaticamente dal test set), produce
una figura a 2 righe x 4 colonne:

    Pesce A (originale) | Heatmap A | Pesce B (originale) | Heatmap B
    ---------------------------------------------------------------
    [zoomed region A]   | [overlay] | [zoomed region B]   | [overlay]

    Cosine Similarity: 0.21 (distanti → pesci diversi ✓)
    oppure
    Cosine Similarity: 0.91 (vicini → stesso pesce ✓)

    # Confronto automatico tra 5 coppie casuali dal test set (pesci DIVERSI):
    python compare_attention.py --part head --model_path siamese_vit_head_ep3.pth

    # Confronto tra due immagini specifiche:
    python compare_attention.py --part head --model_path siamese_vit_head_ep3.pth \\
        --img_a dataset/Melops_head/Melops_head/P7092062_2022.jpg \\
        --img_b dataset/Melops_head/Melops_head/P8290870_2024.jpg

    # Confronto tra due immagini dello STESSO pesce (coppia positiva):
    python compare_attention.py --part head --model_path siamese_vit_head_ep3.pth \\
        --img_a dataset/Melops_head/Melops_head/P7092062_2022.jpg \\
        --img_b dataset/Melops_head/Melops_head/P7092063_2022.jpg \\
        --same_fish
"""

import os
import argparse
import random
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

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
import torchvision.transforms as transforms

from model import SiameseViT
from dataset import build_images_by_id, load_metadata, split_ids_by_identity
from attention_rollout import rollout, generate_visualization
from run_manager import get_run_dir


# ------------------------------------------------------------------ #
# Trasformazione standard (uguale al training)
# ------------------------------------------------------------------ #
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Confronto visivo heatmap tra coppie di pesci"
    )
    parser.add_argument('--part', type=str, default='head',
                        choices=['head', 'body', 'full'])
    parser.add_argument('--run', type=int, required=True,
                        help="Numero della run (es. 1 per run_001).")
    parser.add_argument('--epoch', type=int, required=True,
                        help="Epoca del modello da caricare (es. 3 per epoch_03.pth).")
    parser.add_argument('--num_pairs', type=int, default=5,
                        help="Numero di coppie casuali da generare.")
    parser.add_argument('--img_a', type=str, default=None,
                        help="Percorso immagine A (modalita' manuale).")
    parser.add_argument('--img_b', type=str, default=None,
                        help="Percorso immagine B (modalita' manuale).")
    parser.add_argument('--same_fish', action='store_true',
                        help="Se specificato, le due immagini sono dello stesso pesce.")
    parser.add_argument('--discard_ratio', type=float, default=0.9,
                        help="Percentuale di attenzione da scartare nel rollout (default 0.9).")
    return parser.parse_args()


def load_model(model_path, device):
    model = SiameseViT(model_name='vit_base_patch16_224', pretrained=False, embedding_dim=256)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    for blk in model.backbone.blocks:
        blk.attn.fused_attn = False

    return model


def get_attention_map(model, img_tensor, discard_ratio=0.9):
    """
    Esegue il forward pass con hook su attn_drop e restituisce
    la mappa di attenzione (Attention Rollout) normalizzata.

    Returns:
        mask: numpy array (H, W) con valori in [0, 1]
    """
    attentions = []

    def hook_fn(module, input, output):
        attentions.append(output.detach().cpu())

    handles = []
    for blk in model.backbone.blocks:
        handles.append(blk.attn.attn_drop.register_forward_hook(hook_fn))

    with torch.no_grad():
        _ = model(img_tensor)

    for h in handles:
        h.remove()

    if not attentions:
        raise RuntimeError("Nessuna attenzione estratta. Controlla che fused_attn sia False.")

    mask = rollout(attentions, discard_ratio=discard_ratio, head_fusion="mean")
    return mask


def get_embedding(model, img_tensor):
    """Restituisce l'embedding normalizzato L2 per un'immagine."""
    with torch.no_grad():
        emb = model(img_tensor)
    return F.normalize(emb, p=2, dim=1)


def preprocess_image(path):
    """Carica e trasforma un'immagine. Restituisce anche la versione numpy per la visualizzazione."""
    pil_img = Image.open(path).convert('RGB')
    img_np = np.array(pil_img.resize((224, 224))) / 255.0   # [0,1] per visualizzazione
    img_tensor = TRANSFORM(pil_img).unsqueeze(0)             # [1,3,224,224]
    return img_tensor, img_np, pil_img


def generate_comparison_figure(path_a, path_b, label_a, label_b,
                                model, device, discard_ratio,
                                cosine_sim, is_same_fish, output_path):
    """
    Genera e salva la figura comparativa a 2 righe x 3 colonne:

    Riga 1: [Originale A] [Heatmap A] [Originale B] [Heatmap B]
    Barra info: Cosine Sim, ID pesci, etichetta coppia
    """
    # Preprocessa entrambe le immagini
    tensor_a, np_a, _ = preprocess_image(path_a)
    tensor_b, np_b, _ = preprocess_image(path_b)

    tensor_a = tensor_a.to(device)
    tensor_b = tensor_b.to(device)

    # Heatmap via Attention Rollout
    mask_a = get_attention_map(model, tensor_a, discard_ratio)
    mask_b = get_attention_map(model, tensor_b, discard_ratio)

    # Overlay heatmap sull'immagine originale
    overlay_a = generate_visualization(np_a, mask_a)
    overlay_b = generate_visualization(np_b, mask_b)

    # ---- Costruzione figura ----------------------------------------- #
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.patch.set_facecolor('#1a1a2e')

    titles = [
        f"Pesce A — ID: {label_a}\n(Immagine originale)",
        f"Pesce A — Attention Rollout\n(Dove guarda il modello)",
        f"Pesce B — ID: {label_b}\n(Immagine originale)",
        f"Pesce B — Attention Rollout\n(Dove guarda il modello)",
    ]
    images = [np_a, overlay_a, np_b, overlay_b]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, color='white', fontsize=10, pad=8)
        ax.axis('off')
    for ax, color in zip([axes[0], axes[1]], ['#4cc9f0', '#4cc9f0']):
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
    for ax, color in zip([axes[2], axes[3]], ['#f72585', '#f72585']):
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)

    if is_same_fish:
        pair_label = "Coppia POSITIVA (stesso pesce)"
        bar_color = '#2d6a4f'
        verdict = "Alta similarita' attesa"
    else:
        pair_label = "Coppia NEGATIVA (pesci diversi)"
        bar_color = '#6d1b1b'
        verdict = "Bassa similarita' attesa"

    sim_text = f"Cosine Similarity: {cosine_sim:.4f}   |   {pair_label}   |   {verdict}"
    fig.text(0.5, 0.01, sim_text,
             ha='center', va='bottom', fontsize=11, color='white',
             bbox=dict(facecolor=bar_color, alpha=0.8, boxstyle='round,pad=0.4'))

    plt.suptitle(
        f"Analisi Attention Rollout — {os.path.basename(path_a)}  vs  {os.path.basename(path_b)}",
        color='white', fontsize=12, y=1.01
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Salvato: {output_path}")


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    run_dir = get_run_dir(args.run)
    results_dir = os.path.join(run_dir, "results")
    model_path = os.path.join(run_dir, "models", f"epoch_{args.epoch:02d}.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modello non trovato: {model_path}")

    print("=" * 60)
    print(f"  Heatmap Comparativa — Parte: {args.part.upper()}")
    print(f"  Run: run_{args.run:03d} | Epoch: {args.epoch}")
    print(f"  Modello: {model_path} | Device: {device}")
    print("=" * 60)

    model = load_model(model_path, device)

    # ------------------------------------------------------------------ #
    # MODALITA' MANUALE: due immagini specificate dall'utente
    # ------------------------------------------------------------------ #
    if args.img_a and args.img_b:
        tensor_a, _, _ = preprocess_image(args.img_a)
        tensor_b, _, _ = preprocess_image(args.img_b)

        emb_a = get_embedding(model, tensor_a.to(device))
        emb_b = get_embedding(model, tensor_b.to(device))
        cosine_sim = torch.dot(emb_a[0], emb_b[0]).item()

        label_a = os.path.splitext(os.path.basename(args.img_a))[0]
        label_b = os.path.splitext(os.path.basename(args.img_b))[0]

        out_path = os.path.join(results_dir, f"compare_{label_a}_vs_{label_b}.png")
        generate_comparison_figure(
            args.img_a, args.img_b, label_a, label_b,
            model, device, args.discard_ratio,
            cosine_sim, args.same_fish, out_path
        )
        print(f"\nCosine Similarity: {cosine_sim:.4f}")
        return

    # ------------------------------------------------------------------ #
    # MODALITA' AUTOMATICA: coppie casuali dal test set
    # ------------------------------------------------------------------ #
    # Ricalcola lo split con lo stesso seed del training
    folder_name = f"Melops_{args.part}"
    data_dir = os.path.join("dataset", folder_name, folder_name)
    filename_to_id = load_metadata("Melops_metadata.txt")
    all_images_by_id, _ = build_images_by_id(data_dir, filename_to_id)
    _, test_ids = split_ids_by_identity(all_images_by_id, test_ratio=0.2, seed=42)

    # Solo identita' con almeno 2 immagini
    test_ids_valid = sorted([
        fid for fid in test_ids
        if fid in all_images_by_id and len(all_images_by_id[fid]) >= 2
    ])

    print(f"Identita' di test disponibili: {len(test_ids_valid)}")
    print(f"Generazione di {args.num_pairs} coppie comparative...\n")

    rng = random.Random(99)  # seed fisso per riproducibilita'

    for pair_idx in range(args.num_pairs):
        # Coppia NEGATIVA (due pesci diversi) — piu' informativa per l'explainability
        id_a, id_b = rng.sample(test_ids_valid, 2)
        path_a = rng.choice(all_images_by_id[id_a])
        path_b = rng.choice(all_images_by_id[id_b])
        is_same = False

        tensor_a, _, _ = preprocess_image(path_a)
        tensor_b, _, _ = preprocess_image(path_b)

        emb_a = get_embedding(model, tensor_a.to(device))
        emb_b = get_embedding(model, tensor_b.to(device))
        cosine_sim = torch.dot(emb_a[0], emb_b[0]).item()

        out_name = f"compare_{args.part}_pair{pair_idx+1:02d}_A{id_a}_B{id_b}.png"
        out_path = os.path.join(results_dir, out_name)

        print(f"Coppia {pair_idx+1}/{args.num_pairs}: ID {id_a} vs ID {id_b} | sim={cosine_sim:.4f}")
        generate_comparison_figure(
            path_a, path_b, id_a, id_b,
            model, device, args.discard_ratio,
            cosine_sim, is_same, out_path
        )

    # Genera anche una coppia POSITIVA (stesso pesce, immagini diverse)
    pos_candidates = [fid for fid in test_ids_valid if len(all_images_by_id[fid]) >= 2]
    if pos_candidates:
        id_pos = rng.choice(pos_candidates)
        imgs = all_images_by_id[id_pos]
        path_a, path_b = rng.sample(imgs, 2)

        tensor_a, _, _ = preprocess_image(path_a)
        tensor_b, _, _ = preprocess_image(path_b)
        emb_a = get_embedding(model, tensor_a.to(device))
        emb_b = get_embedding(model, tensor_b.to(device))
        cosine_sim = torch.dot(emb_a[0], emb_b[0]).item()

        out_name = f"compare_{args.part}_POSITIVE_ID{id_pos}.png"
        out_path = os.path.join(results_dir, out_name)
        print(f"\nCoppia POSITIVA: ID {id_pos} vs se stesso | sim={cosine_sim:.4f}")
        generate_comparison_figure(
            path_a, path_b, id_pos, id_pos,
            model, device, args.discard_ratio,
            cosine_sim, True, out_path
        )

    print("\nTutte le figure salvate in results/")


if __name__ == '__main__':
    main()
