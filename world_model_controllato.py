"""
World model in un caso controllato: agli embedding veri di un backbone congelato si aggiunge un cambiamento sintetico
noto, una rotazione in tutti i 192 piani di una base ortonormale casuale con angolo
    theta = omega * v(sesso) * anni dall'inizio del dataset + (omega / 2) * (lunghezza - mediana) in cm,
v = 1 per i maschi nidificanti, 1,5 per le femmine, 0,5 per i maschi sneaker. Tra due catture la rotazione relativa
dipende solo da intervallo, sesso e crescita, che il predittore riceve: il world model deve recuperarla.
omega = 0 e' il world model sui dati veri con lo stesso backbone.

    python world_model_controllato.py --omega 0 0.25 0.5 1          output: runs/world_model_controllato_<run>/
"""

import os
import json
import math
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from dataset import build_records
from split_fisso import carica_split
from reid import protocollo
from reid_locale import valuta_punteggi, sveglio
from reid_aliked import metadati, misure, differenza, per_fascia, FASCE
from riferimenti_globali import embedding, foto
from lejepa_multivista import Modello, LATO_GLOBALE
from lejepa_world_model import Info, caratteristiche, coppie_train, Predittore, embedding_train, punteggi_proiettati

VELOCITA_SESSO = np.array([1.0, 1.5, 0.5])         # codici di Info.sesso: m, f, s


def ruota(E, theta, Q):
    """Ruota ogni embedding [..., N, D] del proprio angolo theta [N] in tutti i piani della base Q [D, D]."""
    c = E @ Q
    a, b = c[..., 0::2], c[..., 1::2]
    co, si = np.cos(theta)[:, None], np.sin(theta)[:, None]
    r = np.empty_like(c)
    r[..., 0::2], r[..., 1::2] = a * co - b * si, a * si + b * co
    return (r @ Q.T).astype(np.float32)


def angoli(I, omega, giorno0, lung_mediana):
    anni = (I.giorno - giorno0) / 365.25
    return omega * VELOCITA_SESSO[I.sesso] * anni + omega / 2 * (I.lung - lung_mediana) / 10


def allena(E_tr, C_tr, s_tr, t_tr, val, device, epochs=200, batch=512, lr=1e-3, ogni=10, seed=0):
    """Predittore di lejepa_world_model, stesso addestramento; restituisce quello migliore sulla validation."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    pred = Predittore().to(device)
    opt = torch.optim.AdamW(pred.parameters(), lr=lr, weight_decay=0.05)
    E_dev, C_dev = torch.from_numpy(E_tr).to(device), C_tr.to(device)
    totale, passo, migliore, stato = epochs * math.ceil(len(s_tr) / batch), 0, (-1.0, 0), None
    for epoca in range(1, epochs + 1):
        perm = rng.permutation(len(s_tr))
        for a in range(0, len(perm), batch):
            idx = perm[a:a + batch]
            for g in opt.param_groups:
                g["lr"] = lr * 0.5 * (1 + math.cos(math.pi * passo / totale))
            vs, vt = rng.integers(0, len(E_tr), len(idx)), rng.integers(0, len(E_tr), len(idx))
            loss = (1 - (pred(E_dev[vs, s_tr[idx]], C_dev[idx]) * E_dev[vt, t_tr[idx]]).sum(-1)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            passo += 1
        if epoca % ogni == 0 or epoca == epochs:
            pred.eval()
            m = np.mean([r["ap"] for r in valuta(pred, *val, device)])
            pred.train()
            if m > migliore[0]:
                migliore, stato = (m, epoca), {k: v.clone() for k, v in pred.state_dict().items()}
    pred.load_state_dict(stato)
    return pred.eval(), migliore[1]


@torch.no_grad()
def valuta(pred, recs, galleria, query, I, emb, device):
    return valuta_punteggi(punteggi_proiettati(pred, emb, I, recs, galleria, query, device), recs, galleria, query)


def main():
    ap = argparse.ArgumentParser(description="World model in un caso controllato")
    ap.add_argument("--backbone", default=os.path.join("runs", "arcface_dinov2_catture_prototipi", "migliore.pth"))
    ap.add_argument("--omega", type=float, nargs="+", default=[0, 0.25, 0.5, 1.0], help="radianti per anno")
    ap.add_argument("--epochs", type=int, default=200)
    args = ap.parse_args()
    sveglio()
    device = torch.device("cuda")
    out_dir = os.path.join("runs", "world_model_controllato_" + os.path.basename(os.path.dirname(args.backbone)))
    os.makedirs(out_dir, exist_ok=True)

    insiemi = carica_split(build_records("head"))
    meta = metadati()
    tr = insiemi["train"]
    cache = os.path.join(out_dir, "embedding.npz")
    if os.path.exists(cache):
        npz = np.load(cache)
        E = {k: npz[k] for k in ("train", "validation", "test")}
    else:
        modello = Modello("zero").to(device).eval()
        modello.backbone.load_state_dict(torch.load(args.backbone, map_location="cpu", weights_only=False)["backbone"])
        E = {"train": embedding_train(modello.backbone, tr, device).float().numpy()}
        for nome in ("validation", "test"):
            E[nome] = embedding(lambda x: modello.backbone(x), foto(insiemi[nome], LATO_GLOBALE), device)
        np.savez(cache, **E)
        del modello
        torch.cuda.empty_cache()

    s_tr, t_tr = coppie_train(tr)
    I_tr = Info(tr, meta)
    C_tr = caratteristiche(I_tr, s_tr, t_tr)
    dati = {n: (insiemi[n], *protocollo(insiemi[n]), Info(insiemi[n], meta)) for n in ("validation", "test")}
    giorno0 = min(I.giorno.min() for I in [I_tr] + [d[3] for d in dati.values()])
    lung_mediana = float(np.median(I_tr.lung))
    Q = np.linalg.qr(np.random.default_rng(0).standard_normal((384, 384)))[0].astype(np.float32)
    print(f"{len(s_tr)} coppie di train", flush=True)

    out = {"backbone": args.backbone, "coppie di train": int(len(s_tr)), "omega": {}}
    righe_testo = [f"World model controllato, backbone {args.backbone} (congelato)", ""]
    for omega in args.omega:
        Ed = {"train": ruota(E["train"], angoli(I_tr, omega, giorno0, lung_mediana), Q)}
        for n, (recs, g, q, I) in dati.items():
            Ed[n] = ruota(E[n], angoli(I, omega, giorno0, lung_mediana), Q)
        val = (*dati["validation"], Ed["validation"])
        pred, epoca = allena(Ed["train"], C_tr, s_tr, t_tr, val, device, epochs=args.epochs)
        torch.save({"predittore": pred.state_dict(), "epoca": epoca, "omega": omega},
                   os.path.join(out_dir, f"predittore_omega_{omega}.pth"))
        recs, galleria, query, I = dati["test"]
        inv = valuta(None, recs, galleria, query, I, Ed["test"], device)
        wm = valuta(pred, recs, galleria, query, I, Ed["test"], device)
        orig = valuta(None, recs, galleria, query, I, E["test"], device)
        out["omega"][str(omega)] = {
            "epoca scelta (validation)": epoca,
            "senza cambiamento sintetico": {"misure": misure(orig), "per fascia": per_fascia(orig)},
            "sola invarianza": {"misure": misure(inv), "per fascia": per_fascia(inv)},
            "world model": {"misure": misure(wm), "per fascia": per_fascia(wm)},
            "differenza world model - invarianza": differenza(wm, inv)}
        o = out["omega"][str(omega)]
        righe_testo.append(f"omega {omega} rad/anno (predittore scelto all'epoca {epoca}), test {len(query)} query")
        for k in ("senza cambiamento sintetico", "sola invarianza", "world model"):
            righe_testo.append(f"  {k}: " + " | ".join(f"{a} {v[0]:.2f} ({v[1]:.2f}-{v[2]:.2f})" for a, v in o[k]["misure"].items()))
        righe_testo.append("  differenza world model - invarianza: " + " | ".join(
            f"{a} {v[0]:+.2f} ({v[1]:+.2f} .. {v[2]:+.2f})" for a, v in o["differenza world model - invarianza"].items()))
        for n, _, _ in FASCE:
            if n in o["sola invarianza"]["per fascia"]:
                a, b = o["sola invarianza"]["per fascia"][n], o["world model"]["per fascia"][n]
                righe_testo.append(f"    {n}: {a['query']} query | rank-1 invarianza {a['rank-1']:.2f}, world model {b['rank-1']:.2f}")
        print("\n".join(righe_testo[-(len(FASCE) + 5):]), flush=True)
    testo = "\n".join(righe_testo)
    with open(os.path.join(out_dir, "risultati.txt"), "w", encoding="utf-8") as f:
        f.write(testo + "\n")
    with open(os.path.join(out_dir, "risultati.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False, default=float)


if __name__ == "__main__":
    main()
