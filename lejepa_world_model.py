"""
World model con backbone congelato: il backbone dell'invarianza tra catture resta fermo e si allena un predittore che
porta l'embedding di una foto alla data e allo stato di un'altra cattura dello stesso pesce.
  ingressi    : embedding (384) + 18 caratteristiche del passaggio (intervallo con segno e assoluto, lunghezze e crescita,
                sesso di partenza, stato riproduttivo e mese di partenza e arrivo)
  predittore  : MLP residuo (384 + 18) -> 1024 -> 1024 -> 384, ultimo strato inizializzato a zero
  loss        : 1 - coseno tra embedding previsto e vero
  valutazione : la galleria viene portata alla data della query; confronto con la sola invarianza e con l'invarianza
                riordinata sui metadati (pesi scelti sulla validation)

    python lejepa_world_model.py --da dinov2_puro          output: runs/world_model_<da>/
"""

import os
import csv
import json
import math
import time
import argparse
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from dataset import build_records, normalize_uint8
from split_fisso import carica_split
from reid import protocollo, caso
from reid_locale import valuta_punteggi, galleria_per_query, sveglio
from reid_aliked import metadati, misure, differenza, per_fascia, crescita_annua_train, FASCE
from riferimenti_globali import embedding, foto
from lejepa_multivista import FotoCache, Viste, Modello, LATO_GLOBALE

N_VERSIONI = 4
N_CARATT = 18


class Info:
    """Per ogni foto di un insieme: giorno (ordinale), lunghezza, sesso, stato riproduttivo, mese."""

    def __init__(self, records, meta):
        self.giorno = np.array([datetime.strptime(r["date"], "%d.%m.%Y").toordinal() for r in records], dtype=np.float64)
        self.mese = np.array([int(r["date"].split(".")[1]) for r in records], dtype=np.float64)
        lung = np.array([meta[r["stem"]]["length"] for r in records], dtype=np.float64)
        self.lung = np.where(np.isnan(lung), np.nanmedian(lung), lung)
        self.sesso = np.array([{"m": 0, "f": 1, "s": 2}.get(meta[r["stem"]]["sex"], 0) for r in records])
        self.ripr = np.array([int(meta[r["stem"]]["spawning"]) if meta[r["stem"]]["spawning"] in ("0", "1", "2") else 0
                              for r in records])


def caratteristiche(I, s, t):
    """Passaggio dalla foto s alla data e allo stato della foto t (array di indici) -> [n, 18]."""
    dt = (I.giorno[t] - I.giorno[s]) / 365.25
    occhio = lambda v, k: np.eye(k)[v]
    ang = lambda m: np.stack([np.sin(2 * math.pi * m / 12), np.cos(2 * math.pi * m / 12)], 1)
    f = np.concatenate([dt[:, None], np.abs(dt)[:, None], I.lung[s, None] / 100, I.lung[t, None] / 100,
                        (I.lung[t] - I.lung[s])[:, None] / 20, occhio(I.sesso[s], 3), occhio(I.ripr[s], 3),
                        occhio(I.ripr[t], 3), ang(I.mese[s]), ang(I.mese[t])], 1)
    return torch.from_numpy(f).float()


def coppie_train(records):
    """Tutte le coppie ordinate (s, t) di foto in date diverse dello stesso pesce e lato."""
    gruppi = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(records):
        gruppi[(r["fish_id"], r["side"])][r["date"]].append(i)
    s, t = [], []
    for date in gruppi.values():
        if len(date) < 2:
            continue
        foto_date = list(date.values())
        for a in range(len(foto_date)):
            for b in range(len(foto_date)):
                if a != b:
                    for i in foto_date[a]:
                        for j in foto_date[b]:
                            s.append(i)
                            t.append(j)
    return np.array(s), np.array(t)


class Predittore(nn.Module):
    def __init__(self, d=384, k=N_CARATT, h=1024):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d + k), nn.Linear(d + k, h), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(h, h), nn.GELU(), nn.Dropout(0.1), nn.Linear(h, d))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, e, c):
        return F.normalize(e + self.net(torch.cat([e, c], -1)), dim=-1)


@torch.no_grad()
def embedding_train(backbone, records, device, batch=64):
    """Embedding [N_VERSIONI, foto, 384]: foto intera + ritagli globali aumentati."""
    out = [torch.from_numpy(embedding(lambda x: backbone(x), foto(records, LATO_GLOBALE), device))]
    viste = Viste().to(device)
    cache = FotoCache(records)
    for v in range(N_VERSIONI - 1):
        torch.manual_seed(100 + v)
        e = []
        for i in tqdm(range(0, len(cache), batch), desc=f"embedding, versione {v + 2}", leave=False):
            u8 = torch.stack([cache[j][0] for j in range(i, min(i + batch, len(cache)))]).to(device)
            x = viste.globale(u8.float() / 255).clamp(0, 1)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                e.append(F.normalize(backbone((x - viste.mean) / viste.std).float(), dim=-1).cpu())
        out.append(torch.cat(e))
    return torch.stack(out).half()


@torch.no_grad()
def punteggi_proiettati(pred, emb, I, records, galleria, query, device, batch=4096):
    """Coseno tra query e galleria (stesso lato) portata alla data della query; pred = None: sola invarianza."""
    E = torch.from_numpy(emb).to(device)
    sel = galleria_per_query(records, galleria, query)
    g_all = np.array(galleria)
    P = np.full((len(query), len(galleria)), np.nan)
    for k, q in enumerate(query):
        gi = g_all[sel[k]]
        if pred is None:
            s = E[gi] @ E[q]
        else:
            s = torch.cat([pred(E[gi[a:a + batch]], caratteristiche(I, gi[a:a + batch], np.full(len(gi[a:a + batch]), q)).to(device))
                           @ E[q] for a in range(0, len(gi), batch)])
        P[k, sel[k]] = s.float().cpu().numpy()
    return P


def main():
    ap = argparse.ArgumentParser(description="World model tra catture (backbone congelato)")
    ap.add_argument("--da", choices=["dinov2", "zero", "dinov2_puro"], required=True,
                    help="backbone del passo 2 (invarianza tra catture), runs/lejepa_catture_<da>/migliore.pth")
    ap.add_argument("--backbone", default=None, help="percorso di un altro checkpoint del backbone")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ogni", type=int, default=10)
    ap.add_argument("--prova", type=int, default=0, help="prova veloce: solo N query di validation e test")
    args = ap.parse_args()
    sveglio()
    torch.manual_seed(0)
    device = torch.device("cuda")
    out_dir = os.path.join("runs", f"world_model_{args.da}" + ("_prova" if args.prova else ""))
    os.makedirs(out_dir, exist_ok=True)

    insiemi = carica_split(build_records("head"))
    meta = metadati()
    modello = Modello("zero").to(device).eval()
    percorso = args.backbone or os.path.join("runs", f"lejepa_catture_{args.da}", "migliore.pth")
    ck = torch.load(percorso, map_location="cpu", weights_only=False)
    modello.backbone.load_state_dict(ck["backbone"])
    backbone = modello.backbone
    print(f"backbone: {percorso}", flush=True)

    t0 = time.time()
    tr = insiemi["train"]
    if args.prova:
        tenuti = set(sorted({r["fish_id"] for r in tr})[:600])
        tr = [r for r in tr if r["fish_id"] in tenuti]
    s_tr, t_tr = coppie_train(tr)
    I_tr = Info(tr, meta)
    E_tr = embedding_train(backbone, tr, device)
    print(f"{len(s_tr)} coppie di train, embedding in {time.time() - t0:.0f} s", flush=True)
    valutazione = {}
    for nome in ("validation", "test"):
        recs = insiemi[nome]
        galleria, query = protocollo(recs)
        if args.prova:
            query = query[:args.prova]
        valutazione[nome] = (recs, galleria, query, Info(recs, meta),
                             embedding(lambda x: backbone(x), foto(recs, LATO_GLOBALE), device))
    del modello, backbone
    torch.cuda.empty_cache()

    pred = Predittore().to(device)
    opt = torch.optim.AdamW(pred.parameters(), lr=args.lr, weight_decay=0.05)
    C_tr = caratteristiche(I_tr, s_tr, t_tr)
    E_dev = E_tr.to(device)
    passi = math.ceil(len(s_tr) / args.batch)
    totale = args.epochs * passi
    rng = np.random.default_rng(0)

    def valuta_val():
        recs, galleria, query, I, emb = valutazione["validation"]
        pred.eval()
        righe = valuta_punteggi(punteggi_proiettati(pred, emb, I, recs, galleria, query, device), recs, galleria, query)
        pred.train()
        return misure(righe)

    log = os.path.join(out_dir, "train_log.csv")
    with open(log, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoca", "loss", "val_rank1", "val_mAP", "minuti"])
    migliore, passo = -1.0, 0
    for epoca in range(1, args.epochs + 1):
        perm = rng.permutation(len(s_tr))
        somma = 0.0
        for a in range(0, len(perm), args.batch):
            idx = perm[a:a + args.batch]
            for g in opt.param_groups:
                g["lr"] = args.lr * 0.5 * (1 + math.cos(math.pi * passo / totale))
            vs, vt = rng.integers(0, N_VERSIONI, len(idx)), rng.integers(0, N_VERSIONI, len(idx))
            es = E_dev[vs, s_tr[idx]].float()
            et = E_dev[vt, t_tr[idx]].float()
            loss = (1 - (pred(es, C_tr[idx].to(device)) * et).sum(-1)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            somma, passo = somma + loss.item() * len(idx), passo + 1
        riga = [epoca, f"{somma / len(perm):.5f}", "", "", f"{(time.time() - t0) / 60:.1f}"]
        if epoca % args.ogni == 0 or epoca == args.epochs:
            m = valuta_val()
            riga[2:4] = [m["rank-1"][0], m["mAP"][0]]
            print(f"[epoca {epoca}/{args.epochs}] loss {somma / len(perm):.4f} | validation rank-1 {m['rank-1'][0]:.2f} "
                  f"mAP {m['mAP'][0]:.2f}", flush=True)
            if m["mAP"][0] > migliore:
                migliore = m["mAP"][0]
                torch.save({"predittore": pred.state_dict(), "epoca": epoca, "val": m, "args": vars(args)},
                           os.path.join(out_dir, "predittore.pth"))
        with open(log, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(riga)

    ck = torch.load(os.path.join(out_dir, "predittore.pth"), map_location="cpu", weights_only=False)
    pred.load_state_dict(ck["predittore"])
    pred.eval()

    # riferimento: invarianza riordinata sui metadati (lunghezza e sesso)
    r_anno, _ = crescita_annua_train(tr, meta)

    def con_metadati(nome, a, b):
        recs, galleria, query, I, emb = valutazione[nome]
        P = punteggi_proiettati(None, emb, I, recs, galleria, query, device)
        g_all = np.array(galleria)
        for k, q in enumerate(query):
            ok = ~np.isnan(P[k])
            gi = g_all[ok]
            residuo = (I.lung[q] - I.lung[gi]) - r_anno * (I.giorno[q] - I.giorno[gi]) / 365
            P[k, ok] = P[k, ok] - a * np.abs(residuo) / 10 - b * (I.sesso[gi] != I.sesso[q])
        return valuta_punteggi(P, recs, galleria, query)

    scelta = max(((np.mean([r["ap"] for r in con_metadati("validation", a, b)]), a, b)
                  for a in (0, 0.005, 0.01, 0.02, 0.05, 0.1) for b in (0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)))
    out = {"backbone": percorso, "epoca scelta (validation)": ck["epoca"], "coppie di train": int(len(s_tr)),
           "riordino con i metadati": {"a (lunghezza)": scelta[1], "b (sesso)": scelta[2], "validation mAP": round(100 * scelta[0], 2)}}
    lines = [f"World model, backbone {percorso} (congelato); predittore scelto all'epoca {ck['epoca']} sulla validation",
             f"riferimento con i metadati: a = {scelta[1]}, b = {scelta[2]} (scelti sulla validation)", ""]
    for nome in ("validation", "test"):
        recs, galleria, query, I, emb = valutazione[nome]
        inv = valuta_punteggi(punteggi_proiettati(None, emb, I, recs, galleria, query, device), recs, galleria, query)
        invm = con_metadati(nome, scelta[1], scelta[2])
        wm = valuta_punteggi(punteggi_proiettati(pred, emb, I, recs, galleria, query, device), recs, galleria, query)
        out[nome] = {"query": len(query), "sola invarianza": {"misure": misure(inv), "per fascia": per_fascia(inv)},
                     "invarianza + metadati": {"misure": misure(invm), "per fascia": per_fascia(invm)},
                     "world model": {"misure": misure(wm), "per fascia": per_fascia(wm)},
                     "differenza world model - invarianza": differenza(wm, inv),
                     "differenza world model - invarianza con metadati": differenza(wm, invm),
                     "differenza per fascia": {n: differenza([a for a, b in zip(wm, inv) if lo <= b["giorni"] <= hi],
                                                             [b for b in inv if lo <= b["giorni"] <= hi])
                                               for n, lo, hi in FASCE if any(lo <= b["giorni"] <= hi for b in inv)}}
        if nome == "test":
            out[nome]["classifica casuale"] = [round(float(x), 2) for x in caso(inv)]
        lines.append(f"{nome.upper()}: {len(query)} query")
        for k in ("sola invarianza", "invarianza + metadati", "world model"):
            lines.append(f"  {k}: " + " | ".join(f"{a} {v[0]:.2f} ({v[1]:.2f}-{v[2]:.2f})" for a, v in out[nome][k]["misure"].items()))
        for k in ("differenza world model - invarianza", "differenza world model - invarianza con metadati"):
            lines.append(f"  {k}: " + " | ".join(f"{a} {v[0]:+.2f} ({v[1]:+.2f} .. {v[2]:+.2f})" for a, v in out[nome][k].items()))
        for n, _, _ in FASCE:
            if n in out[nome]["differenza per fascia"]:
                a, b = out[nome]["sola invarianza"]["per fascia"][n], out[nome]["world model"]["per fascia"][n]
                d = out[nome]["differenza per fascia"][n]["rank-1"]
                lines.append(f"    {n}: {a['query']} query | rank-1 invarianza {a['rank-1']:.2f}, world model {b['rank-1']:.2f} "
                             f"({d[0]:+.2f}, {d[1]:+.2f} .. {d[2]:+.2f})")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(out_dir, "risultati.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    with open(os.path.join(out_dir, "risultati.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False, default=float)


if __name__ == "__main__":
    main()
