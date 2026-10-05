"""
run_domani.py â€” Esperimenti per domani

Strategia: Fine-tuning con ArcFace Loss (margine angolare)
  - ArcFace e' la loss standard nel face recognition e Re-ID
  - NON e' contrastiva: opera su singole immagini + etichette
  - Aggiunge un margine angolare che forza le feature dello stesso
    pesce a raggrupparsi piu' strettamente nello spazio
  - In combinazione con augmentation aggressiva e warmup scheduler

Usa il backbone gia' pre-addestrato dalla run_008 (5 epoche I-JEPA).
"""
import subprocess
import sys
import os
import glob

def get_latest_run_number():
    run_dirs = sorted(glob.glob(os.path.join("runs", "run_*")))
    if not run_dirs:
        return 0
    return int(os.path.basename(run_dirs[-1]).split("_")[1])

def main():
    python = sys.executable

    print("\n" + "=" * 60)
    print("  ESPERIMENTO: ArcFace Fine-tuning")
    print("  Backbone: run_008 (I-JEPA 5 epoche)")
    print("  Loss: ArcFace (s=30, m=0.5) + Label Smoothing 0.1")
    print("  Augmentation: RandomResizedCrop + RandomErasing")
    print("  Scheduler: Warmup 3 ep + Cosine Decay")
    print("=" * 60)

    # =============================================
    # Fine-tuning ArcFace (30 epoche)
    # =============================================
    print("\n>>> Avvio Fine-tuning ArcFace...")
    result = subprocess.run([
        python, "train_finetune_v2.py",
        "--run", "8",
        "--part", "head",
        "--epochs", "30",
        "--batch_size", "32",
        "--lr", "5e-5",
        "--warmup_epochs", "3",
        "--arcface_s", "30.0",
        "--arcface_m", "0.50",
        "--wandb_name", "ArcFace FT 30ep (s30 m0.5)",
    ])
    if result.returncode != 0:
        print("ERRORE nel fine-tuning!")
        sys.exit(1)

    ft_run = get_latest_run_number()
    print(f"\nFine-tuning completato! Run: run_{ft_run:03d}")

    # =============================================
    # Valutazione
    # =============================================
    print("\n>>> Valutazione Re-ID...")
    result = subprocess.run([
        python, "evaluate_ijepa.py",
        "--run", str(ft_run),
        "--mode", "pretrain",
        "--part", "head",
    ])

    print("\n" + "=" * 60)
    print("  ESPERIMENTO COMPLETATO!")
    print(f"  Risultati in: runs/run_{ft_run:03d}/results/")
    print("=" * 60)


if __name__ == '__main__':
    main()

