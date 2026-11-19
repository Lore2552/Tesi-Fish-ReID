"""
World model congiunto: encoder, proiettore e predittore condizionato allenati insieme con la loss di LeJEPA.
Coppie, viste e ottimizzazione come in lejepa_catture.py.
  predittore : P(e, c) = e + MLP([LN(e), c]) sul token di classe, c = caratteristiche del passaggio (lejepa_world_model.py);
               ultimo strato a zero, quindi all'inizio coincide con l'invarianza
  loss       : invarianza calcolata nel riferimento di A (viste di B portate alla data di A) e in quello di B, mediata;
               SIGReg su tutte le proiezioni usate dalla loss (grezze e nei due riferimenti), lambda = 0,05
  test       : galleria portata alla data della query con P; accanto, lo stesso backbone senza P e con i metadati

    python lejepa_world_model_congiunto.py --da dinov2_puro --epochs 60 --checkpointing
    output: runs/world_model_congiunto_<da>/
"""

import os
import csv
import json
import time
import random
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import build_records, normalize_uint8
from split_fisso import carica_split
from reid import protocollo, caso
from reid_locale import valuta_punteggi, galleria_per_query, sveglio
from reid_aliked import metadati, misure, differenza, per_fascia, crescita_annua_train, FASCE
from riferimenti_globali import foto
from ijepa_model import SIGReg
from lejepa_multivista import FotoCache, Viste, Modello, lr_al_passo, LATO_GLOBALE
from lejepa_catture import viste_coppia
from lejepa_world_model import Info, caratteristiche, N_CARATT

VISTE_A, VISTE_B = [0, 2, 3, 4], [1, 5, 6, 7]   # posizioni delle viste di A e di B nell'ordine di viste_coppia


class CoppieIndici(torch.utils.data.Dataset):
    """Come lejepa_catture.CoppieCatture, piu' gli indici delle due foto (per i metadati)."""

    def __init__(self, records, seed=0):
        self.foto = FotoCache(records)
        per_chiave = defaultdict(lambda: defaultdict(list))
        for i, r in enumerate(records):
            per_chiave[(r["fish_id"], r["side"])][r["date"]].append(i)
        self.chiavi = [list(d.values()) for d in per_chiave.values() if len(d) >= 2]
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.chiavi)

    def __getitem__(self, k):
        date = self.chiavi[k]
        a, b = self.rng.sample(range(len(date)), 2)
        ia, ib = self.rng.choice(date[a]), self.rng.choice(date[b])
        return self.foto[ia][0], self.foto[ib][0], ia, ib


class Predittore(nn.Module):
    """P(e, c) = e + MLP([LN(e), c]); all'inizio e' l'identita'."""

    def __init__(self, d=384, k=N_CARATT, h=1024):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.net = nn.Sequential(nn.Linear(d + k, h), nn.GELU(), nn.Linear(h, d))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, e, c):
        return e + self.net(torch.cat([self.ln(e), c.to(e.dtype)], -1))


class ModelloWM(nn.Module):
    def __init__(self, init):
        super().__init__()
        self.base = Modello(init)
        self.pred = Predittore()

    def forward(self, g, l, c_ab, c_ba):
        """Proiezioni grezze z [8, B, K] e nei riferimenti di A e di B (zA, zB)."""
        B = g.shape[0] // 2
        E = torch.cat([self.base.backbone(g), self.base.backbone(l)]).view(8, B, -1)
        pA = self.pred(E[VISTE_A], c_ab.expand(4, B, -1))   # viste di A portate alla data di B
        pB = self.pred(E[VISTE_B], c_ba.expand(4, B, -1))   # viste di B portate alla data di A
        EB = torch.stack([pA[0], E[1], pA[1], pA[2], pA[3], E[5], E[6], E[7]])
        EA = torch.stack([E[0], pB[0], E[2], E[3], E[4], pB[1], pB[2], pB[3]])
        proietta = lambda X: self.base.proj(X.reshape(8 * B, -1)).float().view(8, B, -1)
        return proietta(E), proietta(EA), proietta(EB)


def loss_congiunta(z, zA, zB, sigreg, passo, lam):
    """SIGReg su tutte le proiezioni usate dall'invarianza"""
    simA = (zA[:2].mean(0) - zA).square().mean()
    simB = (zB[:2].mean(0) - zB).square().mean()
    sim = (simA + simB) / 2
    sig = torch.stack([sigreg(t[v], seed=(passo * 3 + j) * len(z) + v)
                       for j, t in enumerate((z, zA, zB)) for v in range(len(z))]).mean()
    return (1 - lam) * sim + lam * sig, sim.detach(), sig.detach()


@torch.no_grad()
def cls_grezze(backbone, records, device, batch=32):
    """Token di classe non normalizzato [N, 384], sulla GPU."""
    ds = foto(records, LATO_GLOBALE)
    out = []
    for i in range(0, len(ds), batch):
        u8 = torch.stack([ds[j][0] for j in range(i, min(i + batch, len(ds)))]).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out.append(backbone(normalize_uint8(u8)).float())
    return torch.cat(out)


@torch.no_grad()
def punteggi(pred, E, I, records, galleria, query, device, batch=4096):
    """Coseno query-galleria (galleria portata alla data della query; pred = None: senza P) e coseno medio tra embedding
    proiettato e originale."""
    En = F.normalize(E, dim=-1)
    sel = galleria_per_query(records, galleria, query)
    g_all = np.array(galleria)
    P = np.full((len(query), len(galleria)), np.nan)
    spost = []
    for k, q in enumerate(query):
        gi = g_all[sel[k]]
        if pred is None:
            s = En[gi] @ En[q]
        else:
            parti, mossi = [], []
            for a in range(0, len(gi), batch):
                idx = gi[a:a + batch]
                p = pred(E[idx], caratteristiche(I, idx, np.full(len(idx), q)).to(device))
                parti.append(F.normalize(p, dim=-1) @ En[q])
                mossi.append((F.normalize(p, dim=-1) * En[idx]).sum(-1))
            s = torch.cat(parti)
            spost.append(torch.cat(mossi).mean().item())
        P[k, sel[k]] = s.cpu().numpy()
    return P, (float(np.mean(spost)) if spost else 1.0)


def valuta_insieme(modello, pred, recs, I, galleria, query, device):
    modello.eval()
    E = cls_grezze(modello.base.backbone, recs, device)
    modello.train()
    P_inv, _ = punteggi(None, E, I, recs, galleria, query, device)
    P_wm, spost = punteggi(pred, E, I, recs, galleria, query, device)
    inv = valuta_punteggi(P_inv, recs, galleria, query)
    wm = valuta_punteggi(P_wm, recs, galleria, query)
    return E, inv, wm, spost


def main():
    ap = argparse.ArgumentParser(description="World model JEPA tra catture, encoder e predittore allenati insieme")
    ap.add_argument("--da", choices=["dinov2", "zero", "dinov2_puro"], required=True,
                    help="backbone di partenza: passo 1 da DINOv2 / da zero, oppure i pesi di DINOv2 (dinov2_puro)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4, help="proiettore e predittore")
    ap.add_argument("--lr_backbone", type=float, default=5e-5)
    ap.add_argument("--lam", type=float, default=0.05)
    ap.add_argument("--ogni", type=int, default=5)
    ap.add_argument("--checkpointing", action="store_true")
    ap.add_argument("--riprendi", action="store_true")
    ap.add_argument("--passi_prova", type=int, default=0)
    args = ap.parse_args()
    sveglio()
    torch.manual_seed(0)
    device = torch.device("cuda")
    out_dir = os.path.join("runs", f"world_model_congiunto_{args.da}")
    os.makedirs(out_dir, exist_ok=True)

    insiemi = carica_split(build_records("head"))
    meta = metadati()
    tr = insiemi["train"]
    I_tr = Info(tr, meta)
    dati = CoppieIndici(tr)
    loader = torch.utils.data.DataLoader(dati, batch_size=args.batch, shuffle=True, drop_last=True, num_workers=0, pin_memory=True)
    viste = Viste().to(device)
    modello = ModelloWM("dinov2" if args.da == "dinov2_puro" else "zero").to(device)
    if args.da != "dinov2_puro":
        partenza = os.path.join("runs", f"lejepa_multivista_{args.da}", "migliore.pth")
        ck = torch.load(partenza, map_location="cpu", weights_only=False)
        modello.base.backbone.load_state_dict(ck["backbone"])
        print(f"backbone dal passo 1: {partenza}, epoca {ck['epoca']}", flush=True)
    else:
        print("backbone: pesi di DINOv2", flush=True)
    if args.checkpointing:
        modello.base.backbone.set_grad_checkpointing(True)
    sigreg = SIGReg().to(device)
    opt = torch.optim.AdamW([{"params": modello.base.backbone.parameters(), "lr": args.lr_backbone, "base": args.lr_backbone},
                             {"params": list(modello.base.proj.parameters()) + list(modello.pred.parameters()), "lr": args.lr, "base": args.lr}],
                            weight_decay=0.05)
    totale = args.epochs * len(loader)
    warmup = max(1, round(0.05 * totale))
    valutazione = {}
    for nome in ("validation", "test"):
        recs = insiemi[nome]
        galleria, query = protocollo(recs)
        valutazione[nome] = (recs, Info(recs, meta), galleria, query)
    print(f"{len(dati)} coppie (pesce, lato) con almeno due date, {len(loader)} passi per epoca", flush=True)

    def val():
        recs, I, galleria, query = valutazione["validation"]
        _, inv, wm, spost = valuta_insieme(modello, modello.pred, recs, I, galleria, query, device)
        return misure(inv), misure(wm), spost

    epoca0, passo, migliore = 1, 0, -1.0
    log = os.path.join(out_dir, "train_log.csv")
    if args.riprendi and os.path.exists(os.path.join(out_dir, "ultimo.pth")):
        ck = torch.load(os.path.join(out_dir, "ultimo.pth"), map_location="cpu", weights_only=False)
        modello.load_state_dict(ck["modello"])
        opt.load_state_dict(ck["opt"])
        epoca0, passo, migliore = ck["epoca"] + 1, ck["passo"], ck["migliore"]
        print(f"riprendo dall'epoca {epoca0}", flush=True)
    elif not args.passi_prova:
        with open(log, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["epoca", "loss", "invarianza", "sigreg", "lr", "val_inv_rank1", "val_inv_mAP",
                                    "val_wm_rank1", "val_wm_mAP", "spostamento", "minuti"])
        m_inv, m_wm, spost = val()
        print(f"[prima del training] validation: invarianza rank-1 {m_inv['rank-1'][0]:.2f} mAP {m_inv['mAP'][0]:.2f} | "
              f"world model rank-1 {m_wm['rank-1'][0]:.2f} mAP {m_wm['mAP'][0]:.2f}", flush=True)
        with open(log, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([0, "", "", "", "", m_inv["rank-1"][0], m_inv["mAP"][0], m_wm["rank-1"][0], m_wm["mAP"][0], spost, 0])

    t0 = time.time()
    modello.train()
    for epoca in range(epoca0, args.epochs + 1):
        somme, n = np.zeros(3), 0
        for ua, ub, ia, ib in loader:
            for gr in opt.param_groups:
                gr["lr"] = lr_al_passo(passo, totale, warmup, gr["base"])
            g, l = viste_coppia(viste, ua.to(device, non_blocking=True), ub.to(device, non_blocking=True))
            ia, ib = ia.numpy(), ib.numpy()
            c_ab, c_ba = caratteristiche(I_tr, ia, ib).to(device), caratteristiche(I_tr, ib, ia).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                z, zA, zB = modello(g, l, c_ab, c_ba)
            loss, sim, sig = loss_congiunta(z, zA, zB, sigreg, passo, args.lam)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            somme += [loss.item(), sim.item(), sig.item()]
            n, passo = n + 1, passo + 1
            if args.passi_prova and passo >= args.passi_prova:
                print(f"[prova] {passo} passi | loss {loss.item():.4f} (invarianza {sim.item():.4f}, sigreg {sig.item():.3f}) | "
                      f"{(time.time() - t0) / passo:.2f} s/passo | memoria max {torch.cuda.max_memory_allocated() / 2**30:.1f} GB",
                      flush=True)
                return
        medie = somme / max(n, 1)
        minuti = (time.time() - t0) / 60
        riga = [epoca, f"{medie[0]:.5f}", f"{medie[1]:.5f}", f"{medie[2]:.4f}", f"{opt.param_groups[1]['lr']:.2e}", "", "", "", "", "", f"{minuti:.1f}"]
        testo = f"[epoca {epoca}/{args.epochs}] loss {medie[0]:.4f} | invarianza {medie[1]:.4f} | sigreg {medie[2]:.3f} | {minuti:.0f} min"
        if epoca % args.ogni == 0 or epoca == args.epochs:
            m_inv, m_wm, spost = val()
            riga[5:10] = [m_inv["rank-1"][0], m_inv["mAP"][0], m_wm["rank-1"][0], m_wm["mAP"][0], round(spost, 4)]
            testo += (f" | validation rank-1 {m_wm['rank-1'][0]:.2f} mAP {m_wm['mAP'][0]:.2f} (world model) | "
                      f"sola invarianza rank-1 {m_inv['rank-1'][0]:.2f} mAP {m_inv['mAP'][0]:.2f} | spostamento {spost:.3f}")
            if m_wm["mAP"][0] > migliore:
                migliore = m_wm["mAP"][0]
                torch.save({"backbone": modello.base.backbone.state_dict(), "pred": modello.pred.state_dict(), "epoca": epoca,
                            "val": m_wm, "val invarianza": m_inv, "args": vars(args)}, os.path.join(out_dir, "migliore.pth"))
            torch.save({"modello": modello.state_dict(), "opt": opt.state_dict(), "epoca": epoca, "passo": passo,
                        "migliore": migliore}, os.path.join(out_dir, "ultimo.pth"))
        print(testo, flush=True)
        with open(log, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(riga)

    # test con il checkpoint scelto sulla validation
    ck = torch.load(os.path.join(out_dir, "migliore.pth"), map_location="cpu", weights_only=False)
    modello.base.backbone.load_state_dict(ck["backbone"])
    modello.pred.load_state_dict(ck["pred"])
    modello.eval()
    r_anno, _ = crescita_annua_train(tr, meta)
    risultati = {}
    for nome in ("validation", "test"):
        recs, I, galleria, query = valutazione[nome]
        E, inv, wm, spost = valuta_insieme(modello, modello.pred, recs, I, galleria, query, device)
        P_inv, _ = punteggi(None, E, I, recs, galleria, query, device)
        risultati[nome] = (recs, I, galleria, query, P_inv, inv, wm, spost)

    def con_metadati(nome, a, b):
        recs, I, galleria, query, P_inv, _, _, _ = risultati[nome]
        P = P_inv.copy()
        g_all = np.array(galleria)
        for k, q in enumerate(query):
            ok = ~np.isnan(P[k])
            gi = g_all[ok]
            residuo = (I.lung[q] - I.lung[gi]) - r_anno * (I.giorno[q] - I.giorno[gi]) / 365
            P[k, ok] = P[k, ok] - a * np.abs(residuo) / 10 - b * (I.sesso[gi] != I.sesso[q])
        return valuta_punteggi(P, recs, galleria, query)

    scelta = max(((np.mean([r["ap"] for r in con_metadati("validation", a, b)]), a, b)
                  for a in (0, 0.005, 0.01, 0.02, 0.05, 0.1) for b in (0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)))
    out = {"da": args.da, "epoca scelta (validation)": ck["epoca"], "coppie (pesce, lato)": len(dati),
           "riordino con i metadati": {"a (lunghezza)": scelta[1], "b (sesso)": scelta[2], "validation mAP": round(100 * scelta[0], 2)}}
    lines = [f"World model congiunto (encoder + predittore + SIGReg), da {args.da}; epoca scelta {ck['epoca']} sulla validation",
             f"riferimento con i metadati: a = {scelta[1]}, b = {scelta[2]} (scelti sulla validation)", ""]
    for nome in ("validation", "test"):
        recs, I, galleria, query, _, inv, wm, spost = risultati[nome]
        invm = con_metadati(nome, scelta[1], scelta[2])
        out[nome] = {"query": len(query), "spostamento medio del predittore (coseno proiettato-originale)": round(spost, 4),
                     "sola invarianza": {"misure": misure(inv), "per fascia": per_fascia(inv)},
                     "invarianza + metadati": {"misure": misure(invm), "per fascia": per_fascia(invm)},
                     "world model": {"misure": misure(wm), "per fascia": per_fascia(wm)},
                     "differenza world model - invarianza": differenza(wm, inv),
                     "differenza world model - invarianza con metadati": differenza(wm, invm),
                     "differenza per fascia": {n: differenza([a for a, b in zip(wm, inv) if lo <= b["giorni"] <= hi],
                                                             [b for b in inv if lo <= b["giorni"] <= hi])
                                               for n, lo, hi in FASCE if any(lo <= b["giorni"] <= hi for b in inv)}}
        if nome == "test":
            out[nome]["classifica casuale"] = [round(float(x), 2) for x in caso(inv)]
        lines.append(f"{nome.upper()}: {len(query)} query | spostamento medio del predittore {spost:.3f}")
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
    print(text, flush=True)
    with open(os.path.join(out_dir, "risultati.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    with open(os.path.join(out_dir, "risultati.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False, default=float)
    print(f"FINE: epoca scelta {ck['epoca']} | test world model {out['test']['world model']['misure']} | "
          f"sola invarianza {out['test']['sola invarianza']['misure']}", flush=True)


if __name__ == "__main__":
    main()
