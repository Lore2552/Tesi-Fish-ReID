"""
    python analisi_collasso.py --runs 4 6 7         output: runs/collasso/risultati.txt
"""

import os
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset import build_records, split_fish, subset, FishImages, normalize_uint8, build_image_cache
from ijepa_model import load_checkpoint
from run_manager import get_run_dir

EPS = 1e-7
OUT_DIR = os.path.join("runs", "collasso")


def carica_foto():
    """Foto dei pesci tenuti fuori dal training."""
    records = build_records("head")
    _, test_ids = split_fish(records, test_frac=0.1, seed=42)
    return FishImages(subset(records, test_ids), img_size=(224, 224), cache_path=build_image_cache("head", (224, 224)))


def gruppi_di_foto(n_foto, n_gruppi, size, seed=0):
    perm = np.random.default_rng(seed).permutation(n_foto)
    return [perm[g * size:(g + 1) * size].tolist() for g in range(n_gruppi)]


def encoder_run(run, checkpoint, device):
    model, cfg = load_checkpoint(os.path.join(get_run_dir(run), "models", f"{checkpoint}.pth"), device)
    return f"run {run} (patch {cfg['patch_size']})", model.encoder


class DinoV2:
    """Token di patch di DINOv2 ViT-S/14 su foto 224x224, senza il token di classe."""

    def __init__(self, device):
        import timm
        self.net = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, img_size=224).to(device).eval()

    def __call__(self, x):
        return self.net.forward_features(x)[:, self.net.num_prefix_tokens:]


@torch.no_grad()
def estrai_token(encoder, ds, indici, device, batch_size=32):
    """Token normalizzati di tutte le foto: [foto, patch, D]."""
    out = []
    for i in range(0, len(indici), batch_size):
        x = normalize_uint8(torch.stack([ds[j][0] for j in indici[i:i + batch_size]]).to(device))
        t = encoder(x).float()
        out.append(F.layer_norm(t, (t.size(-1),)).cpu())
    return torch.cat(out)


def rankme(z, device):
    """RankMe di z [N, D]; valori singolari dagli autovalori di z^T z."""
    g = torch.zeros(z.shape[1], z.shape[1], dtype=torch.float64, device=device)
    for i in range(0, len(z), 65536):
        c = z[i:i + 65536].to(device).double()
        g += c.T @ c
    s = torch.linalg.eigvalsh(g).clamp_min(0).sqrt()
    p = s / s.sum() + EPS
    return torch.exp(-(p * p.log()).sum()).item()


def rapporto_varianza(t, device):
    """t [foto, patch, D]: varianza tra foto a posizione fissa / varianza totale dei token."""
    t = t.to(device).double()
    return (t.var(dim=0).mean() / t.flatten(0, 1).var(dim=0).mean()).item()


def main():
    ap = argparse.ArgumentParser(description="RankMe e rapporto di varianza sui token di patch")
    ap.add_argument("--runs", type=int, nargs="+", default=[4, 6, 7])
    ap.add_argument("--checkpoint", default="epoch_030")
    ap.add_argument("--foto", type=int, default=500, help="foto per gruppo")
    ap.add_argument("--gruppi", type=int, default=3, help="gruppi disgiunti di foto")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = carica_foto()
    gruppi = gruppi_di_foto(len(ds), args.gruppi, args.foto)
    modelli = [encoder_run(r, args.checkpoint, device) for r in args.runs] + [("DINOv2 ViT-S/14", DinoV2(device))]

    lines = [f"Token di patch di foto di test intere, senza maschere: {args.gruppi} gruppi disgiunti da {args.foto} foto, "
             f"D = 384, {args.checkpoint}", "valore = media dei gruppi (minimo-massimo)", ""]
    for nome, encoder in modelli:
        rm, rv = [], []
        for ids in tqdm(gruppi, desc=nome):
            t = estrai_token(encoder, ds, ids, device)
            rm.append(rankme(t.flatten(0, 1), device))
            rv.append(rapporto_varianza(t, device))
        lines.append(f"{nome}: {t.shape[1]} patch per foto | RankMe {np.mean(rm):.1f} ({min(rm):.1f}-{max(rm):.1f}), "
                     f"{100 * np.mean(rm) / 384:.0f}% di D | rapporto di varianza {np.mean(rv):.3f} ({min(rv):.3f}-{max(rv):.3f})")
    text = "\n".join(lines)
    print(text)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "risultati.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
