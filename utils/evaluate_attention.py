import os
import sys

# Aggiungi la root del progetto al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Leggiamo il token dal file .env se presente
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
if os.path.exists(env_path):
    with open(env_path, 'r') as f:
        for line in f:
            if line.startswith('HF_TOKEN'):
                token = line.split('=')[1].strip()
                os.environ["HF_TOKEN"] = token
                os.environ["HUGGING_FACE_HUB_TOKEN"] = token
                print("Token HuggingFace caricato dal file .env!")

import torch
import random
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import argparse

from model import SiameseViT
from attention_rollout import rollout, generate_visualization

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=str, default='head', choices=['head', 'body', 'full'])
    parser.add_argument('--model_path', type=str, required=True, help="Percorso del modello addestrato (.pth)")
    parser.add_argument('--num_samples', type=int, default=5, help="Numero di immagini da testare")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Caricamento del Modello
    print(f"Caricamento modello da {args.model_path}...")
    model = SiameseViT(model_name='vit_base_patch16_224', pretrained=False)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    # 2. Prepariamo le trasformazioni
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # 3. Selezioniamo immagini a caso dal dataset
    data_dir = os.path.join("dataset", f"Melops_{args.part}", f"Melops_{args.part}")
    if not os.path.exists(data_dir):
        print(f"Cartella non trovata: {data_dir}")
        return
        
    all_files = [f for f in os.listdir(data_dir) if f.endswith(".jpg")]
    if not all_files:
        print("Nessuna immagine trovata!")
        return
        
    sample_files = random.sample(all_files, min(args.num_samples, len(all_files)))
    
    # Creiamo la cartella per i risultati
    os.makedirs("results", exist_ok=True)
    
    # 4. Generiamo le Heatmap
    print(f"Generazione heatmap per {len(sample_files)} immagini...")
    
    # Hook per estrarre le attenzioni dal ViT
    attentions = []
    def get_attention(module, input, output):
        attentions.append(output.cpu())

    # Registriamo l'hook su 'attn_drop' che si trova subito dopo il softmax dell'attenzione.
    # NB: dobbiamo disabilitare 'fused_attn' (FlashAttention/SDPA) per forzare PyTorch a calcolare e restituire
    # esplicitamente le mappe di attenzione, altrimenti l'hook non scatterà mai!
    for blk in model.backbone.blocks:
        blk.attn.fused_attn = False
        blk.attn.attn_drop.register_forward_hook(get_attention)

    for filename in sample_files:
        path = os.path.join(data_dir, filename)
        img_pil = Image.open(path).convert('RGB')
        img_tensor = transform(img_pil).unsqueeze(0).to(device)
        
        attentions.clear() # Svuota le attenzioni precedenti
        
        # Forward pass (per far scattare gli hook)
        with torch.no_grad():
            _ = model(img_tensor)
            
        # L'output dell'hook in timm spesso è (B, H, N, N)
        # Il nostro rollout si aspetta le attenzioni estratte. 
        if len(attentions) > 0:
            # attentions contiene tensori di shape (B, num_heads, N, N)
            mask = rollout(attentions, discard_ratio=0.9, head_fusion="mean")
            
            import numpy as np
            # Immagine originale in numpy per sovrapposizione
            img_np = np.array(img_pil.resize((224, 224))) / 255.0
            
            cam = generate_visualization(img_np, mask)
            
            # Salvataggio
            out_name = f"heatmap_{args.part}_{filename}"
            out_path = os.path.join("results", out_name)
            plt.imsave(out_path, cam)
            print(f"Salvato: {out_path}")
        else:
            print(f"Errore: Nessuna attenzione estratta per {filename}")

    print("Analisi completata! Controlla la cartella 'results'.")

if __name__ == '__main__':
    main()
