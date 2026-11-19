import os
from datetime import datetime


def get_next_run_dir(base_dir="runs"):
    """Crea la cartella della prossima run libera e ne restituisce il percorso."""
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
    """Percorso di una run esistente dato il suo numero."""
    run_dir = os.path.join(base_dir, f"run_{run_id:03d}")
    if not os.path.exists(run_dir):
        raise FileNotFoundError(
            f"Run non trovata: {run_dir}\n"
            f"Esegui prima il training con train_ijepa.py."
        )
    return run_dir


def save_config(run_dir, config_dict):
    """Scrive config.txt con i parametri della run."""
    config_path = os.path.join(run_dir, "config.txt")
    with open(config_path, "w") as f:
        f.write(f"timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("-" * 40 + "\n")
        for key, value in config_dict.items():
            f.write(f"{key}: {value}\n")
    return config_path

