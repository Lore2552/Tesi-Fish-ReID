"""
run_manager.py — Gestione delle cartelle di run

Struttura generata:
    runs/
    ├── run_001/
    │   ├── config.txt          (iperparametri e informazioni del training)
    │   ├── models/             (checkpoint .pth per ogni epoca)
    │   │   ├── epoch_01.pth
    │   │   └── epoch_02.pth
    │   └── results/            (metriche, heatmap, CSV, CMC curve)
    │       ├── training_metrics.csv
    │       ├── eval_metrics.txt
    │       └── *.png
    ├── run_002/
    │   └── ...
    └── ...
"""

import os
import json
from datetime import datetime


def get_next_run_dir(base_dir="runs"):
    """
    Trova il prossimo numero di run disponibile e crea la cartella.
    Restituisce il percorso della nuova run directory.
    """
    os.makedirs(base_dir, exist_ok=True)

    existing = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("run_")
    ]

    run_numbers = []
    for d in existing:
        try:
            run_numbers.append(int(d.split("_")[1]))
        except (IndexError, ValueError):
            pass

    next_num = max(run_numbers, default=0) + 1
    run_dir = os.path.join(base_dir, f"run_{next_num:03d}")

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "results"), exist_ok=True)

    return run_dir


def get_run_dir(run_id, base_dir="runs"):
    """
    Restituisce il percorso di una run esistente dato il suo ID numerico.
    Utile per evaluate.py e compare_attention.py.
    """
    run_dir = os.path.join(base_dir, f"run_{run_id:03d}")
    if not os.path.exists(run_dir):
        raise FileNotFoundError(
            f"Run non trovata: {run_dir}\n"
            f"Esegui prima il training con train.py."
        )
    return run_dir


def list_runs(base_dir="runs"):
    """Stampa un riepilogo di tutte le run disponibili."""
    if not os.path.exists(base_dir):
        print("Nessuna run trovata. Esegui prima train.py.")
        return

    runs = sorted([
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("run_")
    ])

    if not runs:
        print("Nessuna run trovata.")
        return

    print(f"\n{'Run':<12} {'Config':<40} {'Modelli':<10} {'Risultati'}")
    print("-" * 75)
    for run in runs:
        run_path = os.path.join(base_dir, run)
        config_path = os.path.join(run_path, "config.txt")
        models_dir = os.path.join(run_path, "models")
        results_dir = os.path.join(run_path, "results")

        config_summary = ""
        if os.path.exists(config_path):
            with open(config_path) as f:
                lines = f.readlines()
            # Cerca le righe chiave
            for line in lines:
                if "part:" in line or "epochs:" in line:
                    config_summary += line.strip().replace("part:", "p=").replace("epochs:", "ep=") + " "

        n_models = len([f for f in os.listdir(models_dir) if f.endswith(".pth")]) if os.path.exists(models_dir) else 0
        n_results = len(os.listdir(results_dir)) if os.path.exists(results_dir) else 0

        print(f"{run:<12} {config_summary:<40} {n_models:<10} {n_results}")
    print()


def save_config(run_dir, config_dict):
    """
    Salva un file config.txt con tutti i parametri del training/evaluation.
    """
    config_path = os.path.join(run_dir, "config.txt")
    with open(config_path, "w") as f:
        f.write(f"timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("-" * 40 + "\n")
        for key, value in config_dict.items():
            f.write(f"{key}: {value}\n")
    return config_path


if __name__ == "__main__":
    list_runs()
