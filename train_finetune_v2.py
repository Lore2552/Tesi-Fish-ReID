"""
train_finetune_v2.py” Fine-tuning del backbone I-JEPA

Miglioramenti rispetto alla v1:
  1. ArcFace Loss: margine angolare per feature piu' discriminative 
  2. Data Augmentation: RandomResizedCrop, RandomErasing, ColorJitter
  3. Warmup + Cosine LR Schedule: evita instabilita' iniziale
  4. Label Smoothing: previene overfitting sulle classi con poche immagini
"""
import os
import argparse
import math

# --- Caricamento chiavi ---
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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import wandb

from ijepa_model import build_ijepa
from run_manager import get_next_run_dir, save_config, get_run_dir
from dataset import load_metadata, build_images_by_id, split_ids_by_identity


# =============================================================
# ArcFace Loss Margine angolare per feature discriminative
# =============================================================
class ArcFaceLoss(nn.Module):
    """
    Additive Angular Margin Loss (ArcFace).
    
    e' una loss di classificazione che opera su
    singole immagini. La differenza con la Cross-Entropy standard e':
    - I vettori feature e i pesi del classificatore vengono normalizzati L2
    - Si aggiunge un margine angolare (m) alla classe corretta
    - Si scala il tutto per un fattore (s)
    
    Questo costringe le feature dello stesso pesce a raggrupparsi
    piu' strettamente nello spazio angolare, migliorando la
    separabilita' tra identita' diverse.
    
    Paper: "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
    (Deng et al., CVPR 2019)
    """
    def __init__(self, in_features, num_classes, s=30.0, m=0.50, label_smoothing=0.1):
        super().__init__()
        self.s = s          # fattore di scala (temperatura)
        self.m = m          # margine angolare (in radianti)
        self.label_smoothing = label_smoothing
        
        # Pesi del classificatore (come un nn.Linear ma senza bias)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        # Pre-calcolo dei valori trigonometrici
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        # Soglia per evitare instabilita' numerica
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, features, labels):
        # 1. Normalizza feature e pesi
        features_norm = F.normalize(features, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        
        # 2. Calcola coseno dell'angolo tra feature e pesi
        cosine = F.linear(features_norm, weight_norm)
        
        # 3. Calcola cos(theta + m) per la classe corretta
        sine = torch.sqrt(1.0 - torch.clamp(cosine * cosine, 0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
        
        # Stabilita' numerica: se cos(theta) < threshold, usa formula alternativa
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # 4. Applica il margine solo alla classe corretta
        one_hot = F.one_hot(labels, num_classes=self.weight.shape[0]).float()
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        
        # 5. Scala per il fattore s
        logits *= self.s
        
        # 6. Cross-Entropy con label smoothing
        loss = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
        return loss, logits


# =============================================================
# Modello: Backbone + BN + Projection
# =============================================================
class IJEPAFineTuneModel(nn.Module):
    """
    Modello per il fine-tuning: backbone I-JEPA + BatchNorm + proiezione.
    """
    def __init__(self, ijepa_model, embed_dim=768):
        super().__init__()
        self.backbone = ijepa_model.context_encoder
        self.bn = nn.BatchNorm1d(embed_dim)
    
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
        features = self.bn(features)
        return features
    
    def extract_features(self, x):
        """Per la valutazione: estrae feature normalizzate L2."""
        with torch.no_grad():
            features = self.forward(x)
            features = F.normalize(features, p=2, dim=1)
        return features


# =============================================================
# Dataset con augmentation avanzata
# =============================================================
class AugmentedDataset(Dataset):
    def __init__(self, samples, id_to_label, img_size=224):
        self.samples = samples
        self.id_to_label = id_to_label
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, fish_id = self.samples[idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        label = self.id_to_label[fish_id]
        return img, label


# =============================================================
# Warmup + Cosine Scheduler
# =============================================================
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-7):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
    
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


# =============================================================
# Main
# =============================================================
def main():
    parser = argparse.ArgumentParser(description="Fine-tuning V2: ArcFace + Augmentation")
    parser.add_argument('--run', type=int, required=True)
    parser.add_argument('--checkpoint', type=str, default='best')
    parser.add_argument('--part', type=str, default='head', choices=['head', 'body', 'full'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--warmup_epochs', type=int, default=3)
    parser.add_argument('--arcface_s', type=float, default=30.0, help="ArcFace scale")
    parser.add_argument('--arcface_m', type=float, default=0.50, help="ArcFace margin")
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--wandb_name', type=str, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_dir = get_next_run_dir()
    models_dir = os.path.join(run_dir, "models")

    # --- Checkpoint I-JEPA ---
    source_run = get_run_dir(args.run)
    if args.checkpoint == 'best':
        ckpt_path = os.path.join(source_run, "models", "ijepa_best.pth")
    else:
        ep = int(args.checkpoint)
        ckpt_path = os.path.join(source_run, "models", f"ijepa_epoch_{ep:03d}.pth")
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

    sorted_train_ids = sorted(train_ids)
    id_to_label = {fid: i for i, fid in enumerate(sorted_train_ids)}
    num_classes = len(sorted_train_ids)

    train_samples = []
    for fid in sorted_train_ids:
        if fid in all_images_by_id:
            for path in all_images_by_id[fid]:
                train_samples.append((path, fid))

    print("=" * 60)
    print(f"  Fine-tuning V2 (ArcFace) - Parte: {args.part.upper()}")
    print("=" * 60)
    print(f"  Device: {device}")
    print(f"  Backbone da: {ckpt_path}")
    print(f"  Classi: {num_classes} | Immagini: {len(train_samples)}")
    print(f"  ArcFace: s={args.arcface_s}, m={args.arcface_m}")
    print(f"  LR: {args.lr} | Warmup: {args.warmup_epochs} ep")
    print("=" * 60)

    # --- WandB ---
    wandb_name = args.wandb_name if args.wandb_name else os.path.basename(run_dir)
    wandb.init(
        project="melops-reid-ijepa",
        name=wandb_name,
        config={
            "architecture": "I-JEPA + ArcFace",
            "phase": "finetune_v2_arcface",
            "dataset_part": args.part,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "warmup_epochs": args.warmup_epochs,
            "arcface_s": args.arcface_s,
            "arcface_m": args.arcface_m,
            "label_smoothing": args.label_smoothing,
            "num_classes": num_classes,
            "source_checkpoint": ckpt_path,
        }
    )

    # --- Modello ---
    ijepa = build_ijepa()
    ijepa.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    embed_dim = ijepa.context_encoder.embed_dim
    
    model = IJEPAFineTuneModel(ijepa, embed_dim=embed_dim).to(device)
    arcface = ArcFaceLoss(embed_dim, num_classes,
                          s=args.arcface_s, m=args.arcface_m,
                          label_smoothing=args.label_smoothing).to(device)

    # Tutti i parametri trainabili
    all_params = list(model.parameters()) + list(arcface.parameters())
    optimizer = optim.AdamW(all_params, lr=args.lr, weight_decay=0.01)
    scheduler = WarmupCosineScheduler(optimizer, args.warmup_epochs, args.epochs, args.lr)

    # --- Dataset ---
    dataset = AugmentedDataset(train_samples, id_to_label)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        drop_last=True, num_workers=0, pin_memory=True
    )

    save_config(run_dir, {
        "type": "Fine-tuning V2 ArcFace",
        "part": args.part,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "arcface_s": args.arcface_s,
        "arcface_m": args.arcface_m,
        "label_smoothing": args.label_smoothing,
        "num_classes": num_classes,
        "source_checkpoint": ckpt_path,
    })

    # --- Training ---
    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        arcface.train()
        running_loss = 0.0
        correct = 0
        total = 0

        lr = scheduler.step(epoch)

        progress_bar = tqdm(dataloader, desc=f"Epoca {epoch+1}/{args.epochs}")
        for imgs, labels in progress_bar:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            features = model(imgs)
            loss, logits = arcface(features, labels)
            loss.backward()
            optimizer.step()

            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item()
            acc = 100.0 * correct / total

            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{acc:.1f}%",
                'lr': f"{lr:.2e}"
            })

            wandb.log({
                "finetune/loss_step": loss.item(),
                "finetune/acc_step": acc,
                "finetune/lr": lr,
            })

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100.0 * correct / total

        print(f"\n[EPOCH {epoch+1}/{args.epochs}] Loss: {epoch_loss:.4f} | "
              f"Acc: {epoch_acc:.2f}% | LR: {lr:.2e}")

        wandb.log({
            "epoch/loss": epoch_loss,
            "epoch/accuracy": epoch_acc,
            "epoch/lr": lr,
            "epoch_num": epoch + 1
        })

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            # Salva in formato compatibile con evaluate_ijepa.py --mode probe
            # Salviamo il modello completo come IJEPAWithHead-compatibile
            save_dict = {}
            # Mappa i pesi del backbone nel formato atteso
            for k, v in model.backbone.state_dict().items():
                save_dict[f"backbone.{k}"] = v
            # Aggiungi head (BN come LayerNorm + ArcFace weight come linear)
            save_dict["head.0.weight"] = model.bn.weight.data
            save_dict["head.0.bias"] = model.bn.bias.data
            save_dict["head.0.running_mean"] = model.bn.running_mean
            save_dict["head.0.running_var"] = model.bn.running_var
            save_dict["head.0.num_batches_tracked"] = model.bn.num_batches_tracked
            save_dict["head.1.weight"] = arcface.weight.data
            save_dict["head.1.bias"] = torch.zeros(num_classes)
            
            save_path = os.path.join(models_dir, "probe_best.pth")
            torch.save(save_dict, save_path)

            # Salva anche il backbone puro per evaluate in modalita' pretrain
            ijepa_save = build_ijepa()
            ijepa_save.context_encoder.load_state_dict(model.backbone.state_dict())
            save_path2 = os.path.join(models_dir, "ijepa_best.pth")
            torch.save(ijepa_save.state_dict(), save_path2)

        if (epoch + 1) == args.epochs:
            ijepa_save = build_ijepa()
            ijepa_save.context_encoder.load_state_dict(model.backbone.state_dict())
            save_path = os.path.join(models_dir, f"ijepa_epoch_{epoch+1:03d}.pth")
            torch.save(ijepa_save.state_dict(), save_path)

    print(f"\nFine-tuning V2 completato! Miglior accuracy: {best_acc:.2f}%")
    wandb.finish()


if __name__ == '__main__':
    main()

