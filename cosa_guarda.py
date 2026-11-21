"""
Dove guarda il modello, sul test:
  occlusione media : su N query riconosciute, calo del coseno con la foto giusta quando un quadrato grigio 32x32
                     copre ogni posizione; media sulle query e quota dell'importanza che cade sul pesce
  attenzione       : attenzione del token di classe sulle patch nell'ultimo blocco del ViT, media sulle teste

    python cosa_guarda.py                    output: runs/cosa_guarda/
"""

import os
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import build_records, normalize_uint8
from split_fisso import carica_split
from reid import protocollo
from reid_locale import FONDO_MIN
from riferimenti_globali import embedding, foto, valuta
from controllo_scorciatoie import carica, mappa_occlusione, MODELLI
from lejepa_multivista import LATO_GLOBALE

OUT_DIR = os.path.join("runs", "cosa_guarda")
NOMI = {"5c": "Invarianza + contrastiva", "ArcFace": "ArcFace", "DINOv2": "DINOv2 senza training"}


@torch.no_grad()
def attenzione_cls(backbone, x, device):
    """Attenzione del token di classe sulle patch nell'ultimo blocco, media sulle teste -> [16, 16]."""
    catturato = {}
    gancio = backbone.blocks[-1].attn.register_forward_hook(lambda m, i, o: catturato.setdefault("x", i[0]))
    backbone(normalize_uint8(x[None].to(device)))
    gancio.remove()
    att = backbone.blocks[-1].attn
    t = catturato["x"].float()
    B, N, C = t.shape
    qkv = att.qkv(t).reshape(B, N, 3, att.num_heads, C // att.num_heads).permute(2, 0, 3, 1, 4)
    q, k = qkv[0], qkv[1]
    if hasattr(att, "q_norm"):
        q, k = att.q_norm(q), att.k_norm(k)
    a = ((q * att.scale) @ k.transpose(-2, -1)).softmax(-1)[0, :, 0, 1:].mean(0)
    lato = int(round((N - 1) ** 0.5))
    return a.reshape(lato, lato).cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description="Dove guarda il modello")
    ap.add_argument("--n", type=int, default=200, help="query riconosciute per la mappa media")
    ap.add_argument("--seme", type=int, default=0)
    args = ap.parse_args()
    device = torch.device("cuda")
    os.makedirs(OUT_DIR, exist_ok=True)
    recs = carica_split(build_records("head"))["test"]
    galleria, query = protocollo(recs)
    ds = foto(recs, LATO_GLOBALE)
    rng = np.random.default_rng(args.seme)
    esempi = rng.choice(query, 6, replace=False)

    out, medie, att = {}, {}, {}
    pesce_medio = np.mean([(ds[q][0].min(0).values <= FONDO_MIN).float().numpy() for q in query], 0)
    for nome in ("5c", "ArcFace", "DINOv2"):
        backbone = carica(nome, device)
        att[nome] = [attenzione_cls(backbone, ds[q][0], device) for q in esempi]
        if nome != "DINOv2":
            E = embedding(lambda x: backbone(x), ds, device)
            righe, _, _ = valuta(E, recs, galleria, query)
            giuste = [r for r in righe if r["r1"] == 1]
            scelte = [giuste[k] for k in rng.choice(len(giuste), min(args.n, len(giuste)), replace=False)]
            somma, quota = 0, []
            for r in scelte:
                q, g = r["q"], r["primi"][0]
                m, _ = mappa_occlusione(backbone, ds[q][0], torch.from_numpy(E[g]).to(device), device)
                m = np.clip(m, 0, None)
                somma = somma + m / max(m.sum(), 1e-9)
                pesce = (ds[q][0].min(0).values <= FONDO_MIN).numpy()
                quota.append(float((m * pesce).sum() / max(m.sum(), 1e-9)))
            medie[nome] = somma / len(scelte)
            out[nome] = {"query usate": len(scelte), "importanza sul pesce (%)": round(100 * float(np.mean(quota)), 1),
                         "area del pesce nella foto (%)": round(100 * float(pesce_medio.mean()), 1)}
            print(nome, out[nome], flush=True)
        del backbone
        torch.cuda.empty_cache()

    fondo = np.mean([ds[q][0].permute(1, 2, 0).numpy().astype(float) for q in query[:300]], 0) / 255
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    axes[0].imshow(fondo)
    axes[0].set_title("Foto media del test", fontsize=10)
    for a, nome in zip(axes[1:], ("5c", "ArcFace")):
        a.imshow(fondo)
        a.imshow(medie[nome], cmap="magma", alpha=0.65)
        a.set_title(f"{NOMI[nome]}\nimportanza sul pesce {out[nome]['importanza sul pesce (%)']}%", fontsize=10)
    for a in axes:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"Zone che fanno riconoscere il pesce, media su {args.n} query riconosciute (più chiaro = più importante)", fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig(os.path.join(OUT_DIR, "occlusione_media.png"), dpi=120)
    plt.close()

    fig, axes = plt.subplots(len(esempi), 4, figsize=(10, 2.6 * len(esempi)))
    for i, q in enumerate(esempi):
        img = ds[q][0].permute(1, 2, 0).numpy()
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"pesce {recs[q]['fish_id']}, {recs[q]['date']}", fontsize=8)
        for j, nome in enumerate(("5c", "ArcFace", "DINOv2"), start=1):
            m = F.interpolate(torch.from_numpy(att[nome][i])[None, None], size=img.shape[:2], mode="bilinear")[0, 0].numpy()
            axes[i, j].imshow(img)
            axes[i, j].imshow(m, cmap="magma", alpha=0.6)
            if i == 0:
                axes[i, j].set_title(NOMI[nome], fontsize=9)
        for a in axes[i]:
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Attenzione del token di classe nell'ultimo blocco (più chiaro = più attenzione)", fontsize=11, y=0.997)
    plt.tight_layout(rect=(0, 0, 1, 0.985))
    plt.savefig(os.path.join(OUT_DIR, "attenzione.png"), dpi=110)
    plt.close()
    with open(os.path.join(OUT_DIR, "risultati.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
