import os
import argparse

# --- Caricamento chiavi PRIMA degli import pesanti ---
env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_path):
    with open(env_path, 'r') as f:
        for line in f:
            if line.startswith('HF_TOKEN'):
                token = line.split('=')[1].strip()
                os.environ["HF_TOKEN"] = token
                os.environ["HUGGING_FACE_HUB_TOKEN"] = token
            if line.startswith('WANDB_API_KEY'):
                os.environ["WANDB_API_KEY"] = line.split('=')[1].strip()

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import wandb

from ijepa_model import build_ijepa
from run_manager import get_next_run_dir, save_config
from dataset import load_metadata, build_images_by_id, split_ids_by_identity


class ClassificationDataset(Dataset):
    """
    Dataset per il Linear Probe / Fine-tuning.
    Restituisce (immagine, label_numerica) dove label_numerica e' l'indice
    dell'identita' del pesce nella lista ordinata delle identita' di training.
    """
    def __init__(self, image_paths_with_ids, id_to_label, img_size=224):
        self.samples = image_paths_with_ids  # lista di (path, fish_id)
        self.id_to_label = id_to_label
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, fish_id = self.samples[idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        label = self.id_to_label[fish_id]
        return img, label


class IJEPAWithHead(nn.Module):
    """
    Modello per il fine-tuning: backbone I-JEPA + testa di classificazione.
    
    In modalita' linear probe: il backbone e' congelato, si allena solo la testa.
    In modalita' fine-tune: tutto il modello viene aggiornato.
    """
    def __init__(self, ijepa_model, num_classes, freeze_backbone=True):
        super().__init__()
        self.backbone = ijepa_model.context_encoder
        embed_dim = self.backbone.embed_dim
        
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
    
    def forward(self, x):
        # Passa l'immagine nel backbone ViT
        x = self.backbone.patch_embed(x)
        pos_embed = self.backbone.pos_embed[:, 1:, :]
        x = x + pos_embed
        cls_token = self.backbone.cls_token.expand(x.shape[0], -1, -1)
        cls_pos = self.backbone.pos_embed[:, :1, :]
        cls_token = cls_token + cls_pos
        x = torch.cat([cls_token, x], dim=1)
        x = self.backbone.pos_drop(x)
        for blk in self.backbone.blocks:
            x = blk(x)
        x = self.backbone.norm(x)
        
        # Global Average Pooling sui token delle patch (escluso CLS)
        features = x[:, 1:, :].mean(dim=1)
        
        # Classification head
        logits = self.head(features)
        return logits, features
    
    def extract_features(self, x):
        """Estrae solo le feature (senza classification head), per la valutazione."""
        with torch.no_grad():
            x = self.backbone.patch_embed(x)
            pos_embed = self.backbone.pos_embed[:, 1:, :]
            x = x + pos_embed
            cls_token = self.backbone.cls_token.expand(x.shape[0], -1, -1)
            cls_pos = self.backbone.pos_embed[:, :1, :]
            cls_token = cls_token + cls_pos
            x = torch.cat([cls_token, x], dim=1)
            x = self.backbone.pos_drop(x)
            for blk in self.backbone.blocks:
                x = blk(x)
            x = self.backbone.norm(x)
            features = x[:, 1:, :].mean(dim=1)
            features = F.normalize(features, p=2, dim=1)
        return features


def main():
    parser = argparse.ArgumentParser(description="Linear Probe / Fine-tuning su backbone I-JEPA")
    parser.add_argument('--run', type=int, required=True,
                        help="Numero della run I-JEPA da cui caricare il backbone")
    parser.add_argument('--checkpoint', type=str, default='best',
                        help="Quale checkpoint caricare: 'best' oppure numero epoca (es. '5')")
    parser.add_argument('--part', type=str, default='head', choices=['head', 'body', 'full'])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--freeze_backbone', action='store_true', default=True,
                        help="Congela il backbone (linear probe). Default: True")
    parser.add_argument('--unfreeze_backbone', action='store_true', default=False,
                        help="Scongela il backbone (fine-tuning completo)")
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--wandb_name', type=str, default=None,
                        help="Nome personalizzato per la run WandB")
    args = parser.parse_args()

    freeze = not args.unfreeze_backbone
    mode = "Linear Probe" if freeze else "Fine-tuning"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_dir = get_next_run_dir()
    models_dir = os.path.join(run_dir, "models")

    # --- 1. CARICA IL CHECKPOINT I-JEPA ---
    from run_manager import get_run_dir
    source_run = get_run_dir(args.run)
    if args.checkpoint == 'best':
        ckpt_path = os.path.join(source_run, "models", "ijepa_best.pth")
    else:
        epoch_num = int(args.checkpoint)
        ckpt_path = os.path.join(source_run, "models", f"ijepa_epoch_{epoch_num:03d}.pth")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint non trovato: {ckpt_path}")

    # --- 2. SPLIT PER IDENTITA' ---
    folder_name = f"Melops_{args.part}"
    data_dir = os.path.join("dataset", folder_name, folder_name)
    filename_to_id = load_metadata("Melops_metadata.txt")
    all_images_by_id, _ = build_images_by_id(data_dir, filename_to_id)
    train_ids, test_ids = split_ids_by_identity(
        all_images_by_id, test_ratio=args.test_ratio, seed=args.seed
    )

    # Costruiamo la mappa ID -> label numerica (per cross-entropy)
    sorted_train_ids = sorted(train_ids)
    id_to_label = {fid: i for i, fid in enumerate(sorted_train_ids)}
    num_classes = len(sorted_train_ids)

    # Raccogliamo i campioni di training
    train_samples = []
    for fid in sorted_train_ids:
        if fid in all_images_by_id:
            for path in all_images_by_id[fid]:
                train_samples.append((path, fid))

    print("=" * 60)
    print(f"  {mode} su backbone I-JEPA")
    print("=" * 60)
    print(f"  Device: {device}")
    print(f"  Backbone da: {ckpt_path}")
    print(f"  Identita' TRAIN: {len(train_ids)} ({num_classes} classi)")
    print(f"  Immagini di training: {len(train_samples)}")
    print(f"  Backbone congelato: {freeze}")
    print("=" * 60)

    # --- 3. WANDB ---
    wandb.init(
        project="melops-reid-ijepa",
        name=args.wandb_name if args.wandb_name else os.path.basename(run_dir),
        config={
            "architecture": "I-JEPA + Linear Head",
            "phase": mode.lower().replace(" ", "_"),
            "dataset_part": args.part,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "num_classes": num_classes,
            "freeze_backbone": freeze,
            "source_checkpoint": ckpt_path,
        }
    )

    # --- 4. MODELLO ---
    ijepa = build_ijepa()
    ijepa.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    print("Backbone I-JEPA caricato.")

    model = IJEPAWithHead(ijepa, num_classes=num_classes, freeze_backbone=freeze).to(device)

    # Ottimizzatore: solo parametri trainabili
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"  Parametri trainabili: {sum(p.numel() for p in trainable_params):,}")

    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # --- 5. DATASET ---
    dataset = ClassificationDataset(train_samples, id_to_label)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        drop_last=True, num_workers=2, pin_memory=True
    )

    save_config(run_dir, {
        "type": mode,
        "part": args.part,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "num_classes": num_classes,
        "freeze_backbone": freeze,
        "source_checkpoint": ckpt_path,
    })

    # --- 6. TRAINING LOOP ---
    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(dataloader, desc=f"Epoca {epoch+1}/{args.epochs}")
        for imgs, labels in progress_bar:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits, _ = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            # Accuracy
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            running_loss += loss.item()
            acc = 100.0 * correct / total

            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{acc:.1f}%"
            })

            wandb.log({
                "probe/loss_step": loss.item(),
                "probe/acc_step": acc,
            })

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100.0 * correct / total

        print(f"\n[EPOCH {epoch+1}/{args.epochs}] Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")

        wandb.log({
            "epoch/loss": epoch_loss,
            "epoch/accuracy": epoch_acc,
            "epoch_num": epoch + 1
        })

        # Salva il miglior modello per accuracy
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            save_path = os.path.join(models_dir, "probe_best.pth")
            torch.save(model.state_dict(), save_path)

        if (epoch + 1) == args.epochs:
            save_path = os.path.join(models_dir, f"probe_epoch_{epoch+1:03d}.pth")
            torch.save(model.state_dict(), save_path)

    print(f"\n{mode} completato! Miglior accuracy: {best_acc:.2f}%")
    wandb.finish()


if __name__ == '__main__':
    main()
