"""
Prova "pesci gia' visti": il modello migliore (invarianza + contrastiva) viene riallenato aggiungendo al training le
catture vecchie di meta' dei pesci di test con almeno tre date (gruppo A). In test la query di ogni pesce e' la sua
ultima cattura, mai vista; la galleria contiene le catture precedenti (per il gruppo A, proprio quelle viste in
training). Si confronta il gruppo A con l'altra meta' (gruppo B, mai vista) e con il modello originale.

    python prova_pesci_visti.py                      output: runs/prova_pesci_visti/
"""

import os
import csv
import json
import time
import argparse
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import build_records
from split_fisso import carica_split
from reid import protocollo
from reid_aliked import misure, differenza
from reid_locale import sveglio
from riferimenti_globali import embedding, foto, valuta
from ijepa_model import SIGReg
from lejepa_multivista import Viste, Modello, loss_lejepa, lr_al_passo, valuta_backbone, LATO_GLOBALE
from lejepa_catture import CoppieCatture, viste_coppia_arcface, repulsione

OUT_DIR = os.path.join("runs", "prova_pesci_visti")
ORIGINALE = os.path.join("runs", "lejepa_catture_dinov2_puro_viste_arcface_contrastiva_cls", "migliore.pth")


def gruppi_test(test, seme):
    """Pesci di test con almeno tre date, divisi a caso in A (catture vecchie nel training) e B (fuori)."""
    date = defaultdict(set)
    for r in test:
        date[r["fish_id"]].add(r["date"])
    pesci = sorted(f for f, d in date.items() if len(d) >= 3)
    rng = np.random.default_rng(seme)
    rng.shuffle(pesci)
    return set(pesci[: len(pesci) // 2]), set(pesci[len(pesci) // 2:])


def catture_vecchie(test, gruppo):
    """Foto dei pesci del gruppo scattate prima della loro ultima data."""
    ultima = defaultdict(int)
    giorno = lambda r: int(r["date"][6:] + r["date"][3:5] + r["date"][:2])
    for r in test:
        if r["fish_id"] in gruppo:
            ultima[r["fish_id"]] = max(ultima[r["fish_id"]], giorno(r))
    return [r for r in test if r["fish_id"] in gruppo and giorno(r) < ultima[r["fish_id"]]]


def allena(train, validation, device, out_dir, epochs=60, batch=128, ogni=5):
    """Stessa ricetta del modello migliore: un ritaglio ampio per cattura, LeJEPA + SIGReg sul proiettore,
    contrastiva (s 10, m 0) sul token di classe."""
    import kornia.augmentation as K
    torch.manual_seed(0)
    dati = CoppieCatture(train)
    loader = DataLoader(dati, batch_size=batch, shuffle=True, drop_last=True, num_workers=0, pin_memory=True)
    viste = Viste().to(device)
    modello = Modello("dinov2").to(device)
    modello.backbone.set_grad_checkpointing(True)
    ritaglio = K.RandomResizedCrop((LATO_GLOBALE, LATO_GLOBALE), scale=(0.6, 1.0)).to(device)
    fotometriche = viste.globale[1:]
    sigreg = SIGReg().to(device)
    opt = torch.optim.AdamW([{"params": modello.backbone.parameters(), "lr": 5e-5, "base": 5e-5},
                             {"params": list(modello.proj.parameters()), "lr": 5e-4, "base": 5e-4}], weight_decay=0.05)
    totale = epochs * len(loader)
    warmup = max(1, round(0.05 * totale))
    print(f"{len(dati.chiavi)} (pesce, lato) con almeno due date nel training, {len(loader)} passi per epoca", flush=True)
    log = os.path.join(out_dir, "train_log.csv")
    with open(log, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoca", "loss", "val_rank1", "val_mAP", "minuti"])
    t0, passo, migliore = time.time(), 0, -1.0
    modello.train()
    for epoca in range(1, epochs + 1):
        somma, n = 0.0, 0
        for ua, ub in loader:
            for gr in opt.param_groups:
                gr["lr"] = lr_al_passo(passo, totale, warmup, gr["base"])
            g, _ = viste_coppia_arcface(ritaglio, fotometriche, viste, ua.to(device), ub.to(device))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                e = modello.backbone(g)
                z = modello.proj(e).float().view(2, len(ua), -1)
            loss, _, _ = loss_lejepa(z, sigreg, passo, 0.05)
            loss = loss + repulsione(e.float().view(2, len(ua), -1), s=10.0, m=0.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            somma, n, passo = somma + loss.item(), n + 1, passo + 1
        riga = [epoca, f"{somma / n:.5f}", "", "", f"{(time.time() - t0) / 60:.1f}"]
        if epoca % ogni == 0 or epoca == epochs:
            _, m, _ = valuta_backbone(modello.backbone, validation, device)
            riga[2:4] = [m["rank-1"][0], m["mAP"][0]]
            print(f"[epoca {epoca}/{epochs}] loss {somma / n:.4f} | validation rank-1 {m['rank-1'][0]:.2f} mAP {m['mAP'][0]:.2f}",
                  flush=True)
            if m["mAP"][0] > migliore:
                migliore = m["mAP"][0]
                torch.save({"backbone": modello.backbone.state_dict(), "epoca": epoca, "val": m},
                           os.path.join(out_dir, "migliore.pth"))
        with open(log, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(riga)


@torch.no_grad()
def righe_test(percorso, test, device):
    import timm
    b = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=False, img_size=LATO_GLOBALE, num_classes=0).to(device).eval()
    b.load_state_dict(torch.load(percorso, map_location="cpu", weights_only=False)["backbone"])
    galleria, query = protocollo(test)
    E = embedding(lambda x: b(x), foto(test, LATO_GLOBALE), device)
    return valuta(E, test, galleria, query)[0]


def main():
    ap = argparse.ArgumentParser(description="Prova con pesci gia' visti in training")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seme", type=int, default=0, help="seme della divisione in gruppo A e B")
    args = ap.parse_args()
    sveglio()
    device = torch.device("cuda")
    os.makedirs(OUT_DIR, exist_ok=True)
    insiemi = carica_split(build_records("head"))
    test = insiemi["test"]
    A, B = gruppi_test(test, args.seme)
    aggiunte = catture_vecchie(test, A)
    print(f"gruppo A: {len(A)} pesci, {len(aggiunte)} foto vecchie aggiunte al training; gruppo B: {len(B)} pesci", flush=True)
    with open(os.path.join(OUT_DIR, "gruppi.json"), "w", encoding="utf-8") as f:
        json.dump({"A (catture vecchie nel training)": sorted(A), "B (mai visti)": sorted(B),
                   "foto aggiunte al training": [r["stem"] for r in aggiunte]}, f, indent=1)

    if not os.path.exists(os.path.join(OUT_DIR, "migliore.pth")):
        allena(insiemi["train"] + aggiunte, insiemi["validation"], device, OUT_DIR, epochs=args.epochs)

    out, testo = {}, []
    righe = {"modello con il gruppo A nel training": righe_test(os.path.join(OUT_DIR, "migliore.pth"), test, device),
             "modello originale (nessuno dei due gruppi visto)": righe_test(ORIGINALE, test, device)}
    for nome_m, rr in righe.items():
        out[nome_m] = {}
        testo.append(nome_m)
        for nome_g, sel in (("gruppo A (gia' visto in training)", lambda r: r["fish"] in A),
                            ("gruppo B (mai visto)", lambda r: r["fish"] in B),
                            ("gruppo A, query a piu' di un anno dalle catture viste", lambda r: r["fish"] in A and r["giorni"] > 365),
                            ("gruppo B, query a piu' di un anno", lambda r: r["fish"] in B and r["giorni"] > 365),
                            ("tutte le query del test", lambda r: True)):
            s = [r for r in rr if sel(r)]
            m = misure(s)
            out[nome_m][nome_g] = {"query": len(s), **m}
            testo.append(f"  {nome_g}: {len(s)} query | " + " | ".join(f"{k} {v[0]:.2f} ({v[1]:.2f}-{v[2]:.2f})" for k, v in m.items()))
    for nome_g, sel in (("gruppo A", lambda r: r["fish"] in A), ("gruppo B", lambda r: r["fish"] in B)):
        nuovo = [r for r in righe["modello con il gruppo A nel training"] if sel(r)]
        vecchio = [r for r in righe["modello originale (nessuno dei due gruppi visto)"] if sel(r)]
        d = differenza(nuovo, vecchio)
        out[f"differenza nuovo - originale, {nome_g}"] = d
        testo.append(f"differenza nuovo - originale, {nome_g}: " + " | ".join(f"{k} {v[0]:+.2f} ({v[1]:+.2f} .. {v[2]:+.2f})" for k, v in d.items()))
    print("\n".join(testo), flush=True)
    with open(os.path.join(OUT_DIR, "risultati.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(testo) + "\n")
    with open(os.path.join(OUT_DIR, "risultati.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False, default=float)


if __name__ == "__main__":
    main()
