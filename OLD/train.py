import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import csv

env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_path):
    with open(env_path, 'r') as f:
        for line in f:
            if line.startswith('HF_TOKEN'):
                token = line.split('=')[1].strip()
                os.environ["HF_TOKEN"] = token
                os.environ["HUGGING_FACE_HUB_TOKEN"] = token
                print("Token HuggingFace caricato dal file .env!")

import argparse
from dataset import RealFishDataset, build_images_by_id, load_metadata, split_ids_by_identity
from model import SiameseViT
from loss import ContrastiveLoss
from run_manager import get_next_run_dir, save_config

def parse_args():
    parser = argparse.ArgumentParser(description="Addestramento Siamese Network su Melops")
    parser.add_argument('--part', type=str, default='head', choices=['head', 'body', 'full'],
                        help="Parte del pesce su cui addestrare (head, body, full).")
    parser.add_argument('--epochs', type=int, default=3, help="Numero di epoche.")
    parser.add_argument('--batch_size', type=int, default=8, help="Dimensione del batch.")
    return parser.parse_args()

def main():
    args = parse_args()

    run_dir = get_next_run_dir()
    models_dir = os.path.join(run_dir, "models")
    results_dir = os.path.join(run_dir, "results")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*50)
    print(f"Run directory: {run_dir}")
    print(f"Avvio Addestramento su: {args.part.upper()}")
    print(f"Device: {device}")
    print("="*50)
    
    num_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = 1e-4

    # Dataset: split per identita' (80% train, 20% test)
    # Le identita' di test non vengono MAI viste durante il training.
    print(f"Caricamento dataset {args.part}...")
    folder_name = f"Melops_{args.part}"
    data_dir = os.path.join("dataset", folder_name, folder_name)
    filename_to_id = load_metadata("Melops_metadata.txt")
    all_images_by_id, _ = build_images_by_id(data_dir, filename_to_id)
    train_ids, test_ids = split_ids_by_identity(all_images_by_id, test_ratio=0.2, seed=42)

    print(f"Split identita': {len(train_ids)} train / {len(test_ids)} test (seed=42, ratio=0.2)")

    train_dataset = RealFishDataset(part=args.part, img_size=224, id_subset=train_ids)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f"Immagini di training: {len(train_dataset)} (identita': {len(train_ids)})")

    # Salva la configurazione della run
    save_config(run_dir, {
        "part": args.part,
        "epochs": args.epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "embedding_dim": 256,
        "contrastive_margin": 1.0,
        "train_identities": len(train_ids),
        "test_identities": len(test_ids),
        "train_images": len(train_dataset),
        "split_seed": 42,
        "split_ratio": "80/20",
        "device": str(device),
    })
    print(f"Config salvata in: {run_dir}/config.txt")

    # 3. Creazione di Modello, Loss e Ottimizzatore
    model = SiameseViT(model_name='vit_base_patch16_224', pretrained=True, embedding_dim=256)
    model.to(device)
    
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    csv_file_path = os.path.join(results_dir, "training_metrics.csv")
    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Epoch", "Loss", "Pos_Dist", "Neg_Dist", "Alignment", "Accuracy"])

        # 4. Loop di Addestramento
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            
            # Variabili per le nuove metriche
            total_pos_dist = 0.0
            total_neg_dist = 0.0
            total_alignment = 0.0
            total_correct = 0
            pos_samples = 0
            neg_samples = 0
            total_samples = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoca {epoch+1}/{num_epochs}")
            for img1, img2, label in progress_bar:
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)
                
                optimizer.zero_grad()
                emb1, emb2 = model(img1, img2)
                
                loss, distances = criterion(emb1, emb2, label)
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                # Calcolo delle metriche avanzate staccandole dal gradiente 
                with torch.no_grad():
                    dist_val = distances.view(-1)
                    lbl_val = label.view(-1)
                    
                    pos_mask = (lbl_val == 1)
                    neg_mask = (lbl_val == 0)
                    
                    if pos_mask.sum() > 0:
                        pos_distances = dist_val[pos_mask]
                        total_pos_dist += pos_distances.sum().item()
                        total_alignment += (pos_distances ** 2).sum().item() # Alignment = E[ ||f(x) - f(x+)||^2 ]
                        pos_samples += pos_mask.sum().item()
                        
                        # Correct se la distanza per i positivi è minore del margine diviso 2
                        total_correct += (pos_distances < criterion.margin / 2).sum().item()
                        
                    if neg_mask.sum() > 0:
                        neg_distances = dist_val[neg_mask]
                        total_neg_dist += neg_distances.sum().item()
                        neg_samples += neg_mask.sum().item()
                        
                        # Correct se la distanza per i negativi è maggiore o uguale al margine diviso 2
                        total_correct += (neg_distances >= criterion.margin / 2).sum().item()
                    
                    total_samples += lbl_val.size(0)
                
                # Calcolo stime parziali per la barra di progresso
                avg_pos = (total_pos_dist / pos_samples) if pos_samples > 0 else 0
                avg_neg = (total_neg_dist / neg_samples) if neg_samples > 0 else 0
                avg_align = (total_alignment / pos_samples) if pos_samples > 0 else 0
                acc = (total_correct / total_samples) * 100 if total_samples > 0 else 0
                
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'p_dist': f"{avg_pos:.4f}",
                    'n_dist': f"{avg_neg:.4f}",
                    'align': f"{avg_align:.4f}",
                    'acc': f"{acc:.1f}%"
                })
                
            # Calcolo medie finali dell'epoca
            epoch_loss = running_loss / len(train_loader)
            epoch_pos = (total_pos_dist / pos_samples) if pos_samples > 0 else 0
            epoch_neg = (total_neg_dist / neg_samples) if neg_samples > 0 else 0
            epoch_align = (total_alignment / pos_samples) if pos_samples > 0 else 0
            epoch_acc = (total_correct / total_samples) * 100 if total_samples > 0 else 0
            
            print(f"\n[EPOCH {epoch+1}/{num_epochs}] -> Loss: {epoch_loss:.4f} | Pos_Dist: {epoch_pos:.4f} | Neg_Dist: {epoch_neg:.4f} | Alignment: {epoch_align:.4f} | Acc: {epoch_acc:.2f}%")
            
            # Scrittura su CSV
            csv_writer.writerow([epoch+1, f"{epoch_loss:.4f}", f"{epoch_pos:.4f}", f"{epoch_neg:.4f}", f"{epoch_align:.4f}", f"{epoch_acc:.2f}"])
            csv_file.flush() 
            
            model_save_path = os.path.join(models_dir, f"epoch_{epoch+1:02d}.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"Modello salvato: {model_save_path}\n")

    print("="*50)
    print(f"Addestramento completato per la parte: {args.part.upper()}")
    print("="*50)

if __name__ == '__main__':
    main()
