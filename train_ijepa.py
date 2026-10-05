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
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import wandb

from ijepa_model import build_ijepa
from run_manager import get_next_run_dir, save_config
from dataset import load_metadata, build_images_by_id, split_ids_by_identity


class TrainImageDataset(Dataset):
    """
    Dataset per il pre-training I-JEPA.
    Carica SOLO le immagini appartenenti alle identita' di TRAIN.
    Non restituisce etichette (self-supervised).
    """
    def __init__(self, image_paths, img_size=224):
        self.image_paths = image_paths
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img


def main():
    parser = argparse.ArgumentParser(description="I-JEPA Pre-training (Self-Supervised)")
    parser.add_argument('--part', type=str, default='head', choices=['head', 'body', 'full'])
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1.5e-4)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--wandb_name', type=str, default=None,
                        help="Nome personalizzato per la run WandB")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_dir = get_next_run_dir()
    models_dir = os.path.join(run_dir, "models")

    # --- 1. SPLIT PER IDENTITA' (Open-Set) ---
    folder_name = f"Melops_{args.part}"
    data_dir = os.path.join("dataset", folder_name, folder_name)
    filename_to_id = load_metadata("Melops_metadata.txt")
    all_images_by_id, _ = build_images_by_id(data_dir, filename_to_id)
    train_ids, test_ids = split_ids_by_identity(
        all_images_by_id, test_ratio=args.test_ratio, seed=args.seed
    )

    # Raccogliamo SOLO le immagini delle identita' di TRAIN
    train_image_paths = []
    for fid in train_ids:
        if fid in all_images_by_id:
            train_image_paths.extend(all_images_by_id[fid])

    print("=" * 60)
    print(f"  I-JEPA Pre-training - Parte: {args.part.upper()}")
    print("=" * 60)
    print(f"  Device: {device}")
    print(f"  Run dir: {run_dir}")
    print(f"  Identita' TRAIN: {len(train_ids)} | Identita' TEST: {len(test_ids)}")
    print(f"  Immagini per il pre-training: {len(train_image_paths)}")
    print(f"  Epoche: {args.epochs} | Batch size: {args.batch_size}")
    print("=" * 60)

    # --- 2. WANDB ---
    wandb.init(
        project="melops-reid-ijepa",
        name=args.wandb_name if args.wandb_name else os.path.basename(run_dir),
        config={
            "architecture": "I-JEPA ViT-B/16",
            "phase": "pretrain",
            "dataset_part": args.part,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "train_ids": len(train_ids),
            "test_ids": len(test_ids),
            "train_images": len(train_image_paths),
            "test_ratio": args.test_ratio,
            "seed": args.seed,
        }
    )

    # --- 3. DATASET E DATALOADER ---
    dataset = TrainImageDataset(train_image_paths)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        drop_last=True, num_workers=2, pin_memory=True
    )

    save_config(run_dir, {
        "type": "I-JEPA Pre-training",
        "part": args.part,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "train_ids": len(train_ids),
        "test_ids": len(test_ids),
        "train_images": len(train_image_paths),
        "split_seed": args.seed,
    })

    # --- 4. MODELLO E OTTIMIZZATORE ---
    model = build_ijepa().to(device)
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.05
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- 5. TRAINING LOOP ---
    best_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_target_std = 0.0
        running_pred_std = 0.0

        progress_bar = tqdm(dataloader, desc=f"Epoca {epoch+1}/{args.epochs}")
        for step, imgs in enumerate(progress_bar):
            imgs = imgs.to(device)

            optimizer.zero_grad()
            loss, preds, targets, _, _ = model(imgs)
            loss.backward()
            optimizer.step()
            model.update_target_encoder()

            with torch.no_grad():
                pred_std = preds[0].std(dim=-1).mean().item()
                target_std = targets[0].std(dim=-1).mean().item()

            running_loss += loss.item()
            running_pred_std += pred_std
            running_target_std += target_std

            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'tgt_std': f"{target_std:.3f}"
            })

            wandb.log({
                "train/loss_step": loss.item(),
                "train/target_std_step": target_std,
                "train/pred_std_step": pred_std,
                "train/lr": optimizer.param_groups[0]['lr'],
                "epoch": epoch + (step / len(dataloader))
            })

        scheduler.step()

        epoch_loss = running_loss / len(dataloader)
        epoch_tgt_std = running_target_std / len(dataloader)
        epoch_pred_std = running_pred_std / len(dataloader)

        print(f"\n[EPOCH {epoch+1}/{args.epochs}] Loss: {epoch_loss:.4f} | "
              f"Target Std: {epoch_tgt_std:.4f} | Pred Std: {epoch_pred_std:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        wandb.log({
            "epoch/loss": epoch_loss,
            "epoch/target_std": epoch_tgt_std,
            "epoch/pred_std": epoch_pred_std,
            "epoch/lr": optimizer.param_groups[0]['lr'],
            "epoch_num": epoch + 1
        })

        # Salva il miglior modello
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_path = os.path.join(models_dir, "ijepa_best.pth")
            torch.save(model.state_dict(), save_path)

        # Salva checkpoint ogni 5 epoche e all'ultima
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
            save_path = os.path.join(models_dir, f"ijepa_epoch_{epoch+1:03d}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  Checkpoint salvato: {save_path}")

    print(f"\nTraining completato! Miglior loss: {best_loss:.4f}")
    wandb.finish()


if __name__ == '__main__':
    main()
