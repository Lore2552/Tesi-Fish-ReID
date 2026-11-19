"""
Riferimenti senza addestramento (MegaDescriptor-L-384, DINOv2 ViT-S/14 a 224 e 448 px) e funzioni di valutazione con un
embedding per foto, usate anche dai modelli allenati: coseno, protocollo di reid.py, stesso lato.

    python riferimenti_globali.py [--insieme test]     output: runs/riferimenti/
"""

import os
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset import build_records, normalize_uint8, FishImages, build_image_cache
from split_fisso import carica_split
from reid import protocollo, caso
from reid_locale import valuta_punteggi, sveglio
from reid_aliked import misure, per_fascia

OUT_DIR = os.path.join("runs", "riferimenti")


def foto(records, lato):
    """Foto uint8 [3, lato, lato], dalla cache se esiste."""
    cache = os.path.join("cache", f"head_{lato}x{lato}.u8")
    if os.path.exists(cache) and os.path.exists(cache + ".txt"):
        return FishImages(records, img_size=(lato, lato), cache_path=cache)
    return FishImages(records, img_size=(lato, lato), cache_path=None)


@torch.no_grad()
def embedding(fn, ds, device, normalizza=normalize_uint8, batch=32):
    """Embedding normalizzati L2 [foto, D] (numpy); fn mappa immagini normalizzate [B, 3, H, W] in [B, D]."""
    out = []
    for i in tqdm(range(0, len(ds), batch), desc="embedding", leave=False):
        u8 = torch.stack([ds[j][0] for j in range(i, min(i + batch, len(ds)))]).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            e = fn(normalizza(u8)).float()
        out.append(F.normalize(e, dim=-1).cpu())
    return torch.cat(out).numpy()


def punteggi(emb, galleria, query):
    """Coseno query x galleria."""
    return emb[query] @ emb[galleria].T


def valuta(emb, records, galleria, query):
    righe = valuta_punteggi(punteggi(emb, galleria, query), records, galleria, query)
    return righe, misure(righe), per_fascia(righe)


def normalizza_mezzo(u8):
    return (u8.float() / 255.0 - 0.5) / 0.5


def main():
    ap = argparse.ArgumentParser(description="Riferimenti senza addestramento sul nuovo split")
    ap.add_argument("--insieme", default="test", choices=["validation", "test"])
    args = ap.parse_args()
    sveglio()
    device = torch.device("cuda")
    import timm
    records = carica_split(build_records("head"))[args.insieme]
    galleria, query = protocollo(records)

    modelli = []
    mega = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True, num_classes=0).to(device).eval()
    modelli.append(("MegaDescriptor-L-384", lambda x: mega(x), 384, normalizza_mezzo))
    for lato in (224, 448):
        dino = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, img_size=lato, num_classes=0).to(device).eval()
        modelli.append((f"DINOv2 ViT-S/14 a {lato} px, token di classe", lambda x, m=dino: m(x), lato, normalize_uint8))
        modelli.append((f"DINOv2 ViT-S/14 a {lato} px, media dei token",
                        lambda x, m=dino: m.forward_features(x)[:, m.num_prefix_tokens:].mean(1), lato, normalize_uint8))

    out = {"insieme": args.insieme, "query": len(query), "galleria": len(galleria)}
    lines = [f"Riferimenti senza addestramento, {args.insieme}: {len(query)} query, galleria {len(galleria)} foto, stesso lato", ""]
    for nome, fn, lato, norm in modelli:
        emb = embedding(fn, foto(records, lato), device, norm)
        righe, m, pf = valuta(emb, records, galleria, query)
        out[nome] = {"misure": m, "per fascia": pf}
        lines.append(f"{nome}: " + " | ".join(f"{k} {v[0]:.2f} ({v[1]:.2f}-{v[2]:.2f})" for k, v in m.items()))
        lines += [f"    {f}: {v}" for f, v in pf.items()]
        print(lines[-len(pf) - 1], flush=True)
    c = caso(righe)
    out["classifica casuale"] = [round(float(x), 2) for x in c]
    lines.append(f"classifica casuale: rank-1 {c[0]:.2f} | rank-5 {c[1]:.2f} | mAP {c[2]:.2f}")
    text = "\n".join(lines)
    print(text)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"risultati_{args.insieme}.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    with open(os.path.join(OUT_DIR, f"risultati_{args.insieme}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False, default=float)


if __name__ == "__main__":
    main()
