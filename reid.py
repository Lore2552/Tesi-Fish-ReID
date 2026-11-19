"""
Protocollo di re-identificazione
  query    : per ogni pesce con almeno due date, una foto per lato dell'ultima cattura
  galleria : catture precedenti di quei pesci + tutte le foto dei pesci con una sola data (fungono da distrattori)
  misure   : coseno, rank-1, rank-5, mAP, stesso lato; intervalli al 95% con bootstrap sui pesci

    python reid.py --runs 6 4 7        output: runs/reid/
"""

import os
import csv
import argparse
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import normalize_uint8
from ijepa_model import build_model
from analisi_collasso import carica_foto, encoder_run, DinoV2

OUT_DIR = os.path.join("runs", "reid")
GIUSTO, SBAGLIATO = "#1a7f37", "#c2362b"
FASCE = [("fino a 30 giorni", 0, 30), ("da 31 a 365 giorni", 31, 365), ("oltre 365 giorni", 366, 10 ** 6)]


def data(r):
    return datetime.strptime(r["date"], "%d.%m.%Y")


def protocollo(records):
    """Indici delle foto in galleria e delle query. Query: una per (pesce, lato) nell'ultima cattura."""
    date = defaultdict(set)
    for r in records:
        date[r["fish_id"]].add(data(r))
    ultima = {f: max(d) for f, d in date.items() if len(d) >= 2}
    galleria = [i for i, r in enumerate(records) if r["fish_id"] not in ultima or data(r) < ultima[r["fish_id"]]]
    in_galleria = {(records[i]["fish_id"], records[i]["side"]) for i in galleria}
    scelte = {}
    for i, r in enumerate(records):
        chiave = (r["fish_id"], r["side"])
        if r["fish_id"] in ultima and data(r) == ultima[r["fish_id"]] and chiave in in_galleria:
            if chiave not in scelte or (r["dayseq"], r["stem"]) > (records[scelte[chiave]]["dayseq"], records[scelte[chiave]]["stem"]):
                scelte[chiave] = i
    return galleria, sorted(scelte.values())


@torch.no_grad()
def embedding(encoder, ds, device, batch_size=32):
    out = []
    for i in tqdm(range(0, len(ds), batch_size), desc="Embedding", leave=False):
        x = normalize_uint8(torch.stack([ds[j][0] for j in range(i, min(i + batch_size, len(ds)))]).to(device))
        out.append(F.normalize(encoder(x).float().mean(dim=1), dim=-1).cpu())
    return torch.cat(out).numpy()


def valuta(emb, records, galleria, query, stesso_lato=True):
    """Una riga per query: rank-1, rank-5, AP, giorni dalla cattura precedente, classifica dei migliori 5."""
    ids = np.array([r["fish_id"] for r in records])
    lati = np.array([r["side"] for r in records])
    g_all = np.array(galleria)
    righe = []
    for q in query:
        g = g_all[lati[g_all] == lati[q]] if stesso_lato else g_all
        ordine = np.argsort(-(emb[g] @ emb[q]), kind="stable")
        giusto = ids[g][ordine] == ids[q]
        pos = np.flatnonzero(giusto)
        ap = float(np.mean(np.arange(1, len(pos) + 1) / (pos + 1)))
        pos_stesso_lato = [i for i in g_all if ids[i] == ids[q] and lati[i] == lati[q]]
        giorni = min((data(records[q]) - data(records[i])).days for i in pos_stesso_lato)
        righe.append({"q": q, "fish": ids[q], "r1": float(giusto[0]), "r5": float(giusto[:5].any()), "ap": ap,
                      "giorni": giorni, "n_gal": len(g), "n_pos": len(pos), "primi": g[ordine[:5]].tolist()})
    return righe


def bootstrap(fish, valori, n=2000, seed=0):
    """Intervallo al 95% ricampionando i pesci."""
    per_pesce = defaultdict(list)
    for f, v in zip(fish, valori):
        per_pesce[f].append(v)
    medie = np.array([np.mean(v) for v in per_pesce.values()])
    pesi = np.array([len(v) for v in per_pesce.values()], dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(medie), (n, len(medie)))
    stime = (medie[idx] * pesi[idx]).sum(1) / pesi[idx].sum(1)
    return np.percentile(stime, 2.5), np.percentile(stime, 97.5)


def caso(righe, seed=0, n=200):
    """Valori attesi con una classifica casuale."""
    rng = np.random.default_rng(seed)
    r1, r5, ap = [], [], []
    for r in righe:
        for _ in range(n):
            pos = np.sort(rng.choice(r["n_gal"], r["n_pos"], replace=False))
            r1.append(pos[0] == 0)
            r5.append(pos[0] < 5)
            ap.append(np.mean(np.arange(1, len(pos) + 1) / (pos + 1)))
    return 100 * np.mean(r1), 100 * np.mean(r5), 100 * np.mean(ap)


def riga_modello(nome, righe):
    fish = [r["fish"] for r in righe]
    pezzi = []
    for chiave, etichetta in (("r1", "rank-1"), ("r5", "rank-5"), ("ap", "mAP")):
        v = [r[chiave] for r in righe]
        lo, hi = bootstrap(fish, v)
        pezzi.append(f"{etichetta} {100 * np.mean(v):.2f} ({100 * lo:.2f}-{100 * hi:.2f})")
    return f"{nome}: " + " | ".join(pezzi)


def differenza(nome, a, b):
    """a - b, query per query, con intervallo al 95% sui pesci."""
    fish = [r["fish"] for r in a]
    pezzi = []
    for chiave, etichetta in (("r1", "rank-1"), ("r5", "rank-5"), ("ap", "mAP")):
        d = [x[chiave] - y[chiave] for x, y in zip(a, b)]
        lo, hi = bootstrap(fish, d)
        pezzi.append(f"{etichetta} {100 * np.mean(d):+.2f} ({100 * lo:+.2f} .. {100 * hi:+.2f})")
    return f"{nome}: " + " | ".join(pezzi)


def figura_esempi(ds, records, righe, path, titolo, n=5, seed=1):
    """Query scelte a caso: la query e le prime 5 foto della classifica."""
    scelte = np.random.default_rng(seed).choice(len(righe), n, replace=False)
    fig, axes = plt.subplots(n, 6, figsize=(12.4, 2.75 * n))
    for row, k in enumerate(scelte):
        r = righe[k]
        axes[row, 0].imshow(ds[r["q"]][0].permute(1, 2, 0).numpy())
        axes[row, 0].set_title(f"query: pesce {r['fish']}\n{records[r['q']]['date']}", fontsize=8)
        for c, i in enumerate(r["primi"], start=1):
            giusto = records[i]["fish_id"] == r["fish"]
            axes[row, c].imshow(ds[i][0].permute(1, 2, 0).numpy())
            axes[row, c].set_title(f"{c}°: {'stesso pesce' if giusto else 'altro pesce'}\n{records[i]['date']}", fontsize=8,
                                   color=GIUSTO if giusto else SBAGLIATO)
            for s in axes[row, c].spines.values():
                s.set_edgecolor(GIUSTO if giusto else SBAGLIATO)
                s.set_linewidth(3 if giusto else 1)
        for a in axes[row]:
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle(titolo, fontsize=14, y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.975))
    plt.savefig(path, dpi=110)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Re-identificazione con encoder congelato")
    ap.add_argument("--runs", type=int, nargs="+", default=[6, 4, 7])
    ap.add_argument("--checkpoint", default="epoch_030")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = carica_foto()
    records = ds.records
    galleria, query = protocollo(records)
    ids = {r["fish_id"] for r in records}
    pesci_q = {records[q]["fish_id"] for q in query}

    modelli = [encoder_run(r, args.checkpoint, device) for r in args.runs]
    modelli.append(("DINOv2 ViT-S/14", DinoV2(device)))
    for seme in range(3):
        torch.manual_seed(seme)
        modelli.append((f"encoder casuale (seme {seme})",
                        build_model({"img_size": [224, 224], "patch_size": 16, "lam": 0.05, "num_slices": 1024}).encoder.to(device).eval()))

    emb = {}
    for nome, encoder in modelli:
        print(nome, flush=True)
        emb[nome] = embedding(encoder, ds, device)

    lines = [f"Re-identificazione, foto dei pesci esclusi dal training: {len(records)} foto, {len(ids)} pesci",
             f"Query: {len(query)} foto di {len(pesci_q)} pesci con catture in almeno due date | Galleria: {len(galleria)} foto",
             "Valori in percentuale; tra parentesi intervallo al 95% sui pesci", ""]
    os.makedirs(OUT_DIR, exist_ok=True)
    tutte = {}
    with open(os.path.join(OUT_DIR, "risultati.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "modello", "query", "pesce", "lato", "giorni", "rank1", "rank5", "ap"])
        for scenario, stesso in (("stesso lato", True), ("qualunque lato", False)):
            lines.append(f"=== {scenario.upper()} ===")
            for nome in emb:
                righe = valuta(emb[nome], records, galleria, query, stesso)
                tutte[(scenario, nome)] = righe
                lines.append(riga_modello(nome, righe))
                for r in righe:
                    q = records[r["q"]]
                    w.writerow([scenario, nome, q["stem"], r["fish"], q["side"], r["giorni"], r["r1"], r["r5"], f"{r['ap']:.5f}"])
            base = next(iter(emb))
            casuali = [n for n in emb if n.startswith("encoder casuale")]
            medio = {k: np.mean([[r[k] for r in tutte[(scenario, n)]] for n in casuali]) for k in ("r1", "r5", "ap")}
            lines.append(f"media dei 3 encoder casuali: rank-1 {100 * medio['r1']:.2f} | rank-5 {100 * medio['r5']:.2f} | mAP {100 * medio['ap']:.2f}")
            c = caso(tutte[(scenario, base)])
            lines.append(f"classifica casuale: rank-1 {c[0]:.2f} | rank-5 {c[1]:.2f} | mAP {c[2]:.2f}")
            lines.append("Differenze (query per query):")
            lines.append(differenza(f"{base} - DINOv2", tutte[(scenario, base)], tutte[(scenario, 'DINOv2 ViT-S/14')]))
            lines.append(differenza(f"{base} - encoder casuale (seme 0)", tutte[(scenario, base)], tutte[(scenario, casuali[0])]))
            lines.append("")

    lines.append("=== STESSO LATO, PER DISTANZA DALLA CATTURA PRECEDENTE ===")
    scelti = [n for n in emb if n.startswith(("run 6", "run 8"))] + ["DINOv2 ViT-S/14", "encoder casuale (seme 0)"]
    for etichetta, lo, hi in FASCE:
        lines.append(f"{etichetta}:")
        for nome in scelti:
            sel = [r for r in tutte[("stesso lato", nome)] if lo <= r["giorni"] <= hi]
            lines.append(f"  {nome}: {len(sel)} query | rank-1 {100 * np.mean([r['r1'] for r in sel]):.2f} | "
                         f"rank-5 {100 * np.mean([r['r5'] for r in sel]):.2f} | mAP {100 * np.mean([r['ap'] for r in sel]):.2f}")
    lines +=["", "=== STESSO LATO, PRIMO RISULTATO SBAGLIATO: SCATTATO ENTRO 7 GIORNI DALLA QUERY? ==="]
    lati = np.array([r["side"] for r in records])
    date_g = {i: data(records[i]) for i in galleria}
    attesa = np.mean([np.mean([abs((date_g[i] - data(records[q])).days) <= 7 for i in galleria if lati[i] == lati[q]])
                      for q in query])
    for nome in scelti:
        sbagliati = [r for r in tutte[("stesso lato", nome)] if not r["r1"]]
        vicini = np.mean([abs((date_g[r["primi"][0]] - data(records[r["q"]])).days) <= 7 for r in sbagliati])
        lines.append(f"{nome}: {100 * vicini:.1f}% su {len(sbagliati)} query")
    lines.append(f"attesa scegliendo a caso nella galleria: {100 * attesa:.1f}%")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(OUT_DIR, "risultati.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")

    primo = next(iter(emb))
    figura_esempi(ds, records, tutte[("stesso lato", primo)], os.path.join(OUT_DIR, f"esempi_run{args.runs[0]}.png"),
                  "Run Patch 16")
    figura_esempi(ds, records, tutte[("stesso lato", "DINOv2 ViT-S/14")], os.path.join(OUT_DIR, "esempi_dinov2.png"),
                  "DinoV2")
    run8 = [n for n in emb if n.startswith("run 8")]
    if run8:
        figura_esempi(ds, records, tutte[("stesso lato", run8[0])], os.path.join(OUT_DIR, "esempi_run8.png"), "Run Patch 8")
    print(f"Risultati in {OUT_DIR}")


if __name__ == "__main__":
    main()
