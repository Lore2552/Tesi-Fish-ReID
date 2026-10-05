"""
run_pipeline.py — Lancia pre-training I-JEPA + fine-tuning in sequenza.
Basta un solo comando per far partire tutto.
"""
import os
import argparse
import subprocess
import sys
import glob

def get_latest_run_number():
    """Trova il numero della run piu' recente in runs/"""
    run_dirs = sorted(glob.glob(os.path.join("runs", "run_*")))
    if not run_dirs:
        return 0
    last = os.path.basename(run_dirs[-1])
    return int(last.split("_")[1])

def main():
    parser = argparse.ArgumentParser(description="Pipeline completa: I-JEPA Pretrain + Fine-tuning")
    parser.add_argument('--part', type=str, default='head', choices=['head', 'body', 'full'])
    parser.add_argument('--pretrain_epochs', type=int, default=10)
    parser.add_argument('--finetune_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--pretrain_lr', type=float, default=1.5e-4)
    parser.add_argument('--finetune_lr', type=float, default=1e-5)
    parser.add_argument('--pretrain_name', type=str, default=None,
                        help="Nome personalizzato per la run WandB di pre-training")
    parser.add_argument('--finetune_name', type=str, default=None,
                        help="Nome personalizzato per la run WandB di fine-tuning")
    args = parser.parse_args()

    python = sys.executable

    # =============================================
    # FASE 1: Pre-training I-JEPA
    # =============================================
    print("\n" + "=" * 60)
    print("  FASE 1: Pre-training I-JEPA")
    print("=" * 60)

    pretrain_cmd = [
        python, "train_ijepa.py",
        "--part", args.part,
        "--epochs", str(args.pretrain_epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.pretrain_lr),
    ]
    if args.pretrain_name:
        pretrain_cmd += ["--wandb_name", args.pretrain_name]

    result = subprocess.run(pretrain_cmd)
    if result.returncode != 0:
        print("ERRORE: Pre-training fallito!")
        sys.exit(1)

    # Recupera il numero della run appena creata
    pretrain_run = get_latest_run_number()
    print(f"\nPre-training completato! Run creata: run_{pretrain_run:03d}")

    # =============================================
    # FASE 2: Fine-tuning con backbone sbloccato
    # =============================================
    print("\n" + "=" * 60)
    print("  FASE 2: Fine-tuning (backbone sbloccato)")
    print("=" * 60)

    finetune_cmd = [
        python, "train_linear_probe.py",
        "--run", str(pretrain_run),
        "--part", args.part,
        "--epochs", str(args.finetune_epochs),
        "--lr", str(args.finetune_lr),
        "--unfreeze_backbone",
    ]
    if args.finetune_name:
        finetune_cmd += ["--wandb_name", args.finetune_name]

    result = subprocess.run(finetune_cmd)
    if result.returncode != 0:
        print("ERRORE: Fine-tuning fallito!")
        sys.exit(1)

    finetune_run = get_latest_run_number()
    print(f"\nFine-tuning completato! Run creata: run_{finetune_run:03d}")

    # =============================================
    # FASE 3: Valutazione Re-ID
    # =============================================
    print("\n" + "=" * 60)
    print("  FASE 3: Valutazione Re-ID")
    print("=" * 60)

    eval_cmd = [
        python, "evaluate_ijepa.py",
        "--run", str(finetune_run),
        "--mode", "probe",
        "--part", args.part,
    ]

    result = subprocess.run(eval_cmd)
    if result.returncode != 0:
        print("ERRORE: Valutazione fallita!")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETATA!")
    print(f"  Pretrain: run_{pretrain_run:03d}")
    print(f"  Fine-tune: run_{finetune_run:03d}")
    print(f"  Risultati in: runs/run_{finetune_run:03d}/results/")
    print("=" * 60)


if __name__ == '__main__':
    main()
