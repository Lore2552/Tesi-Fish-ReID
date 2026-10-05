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
    print("  ESPERIMENTO: ArcFace 'Hard Margin'")
    print("  Obiettivo: Risolvere gli 'hard negatives' (pesci sosia)")
    print("  Backbone: run_008 (I-JEPA 5 epoche)")
    print("  Loss: ArcFace (s=64.0, m=0.70) + Label Smoothing 0.1")
    print("  Epoche: 40 ")
    print("=" * 60)

    # =============================================
    # Fine-tuning ArcFace Aggressivo
    # =============================================
    print("\n>>> Avvio Fine-tuning ArcFace Hard Margin...")
    result = subprocess.run([
        python, "train_finetune_v2.py",
        "--run", "8",
        "--part", "head",
        "--epochs", "40",
        "--batch_size", "32",
        "--lr", "5e-5",
        "--warmup_epochs", "4",
        "--arcface_s", "64.0",
        "--arcface_m", "0.70",
        "--wandb_name", "ArcFace FT 40ep (s64 m0.70)",
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
