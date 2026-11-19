"""
ALIKED + LightGlue sullo split fisso (protocollo di reid.py) e analisi degli errori. Contiene anche le funzioni di
misura (misure, per_fascia, differenza) e i metadati usati dagli altri script.
  aliked --insieme test|validation [--maschera]   punteggi query x galleria; con --maschera solo punti sul pesce
  analisi                                         misure per intervallo, dove sbaglia (intervallo, crescita, sesso,
                                                  stato riproduttivo) e riordino con i metadati:
                                                  log(1 + inlier) - a * |crescita inattesa| / 10 - b * [sesso diverso],
                                                  a e b scelti sulla validation

    python reid_aliked.py aliked --insieme test --maschera        output: runs/reid_aliked/
"""

import os
import csv
import json
import time
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import build_records
from split_fisso import carica_split
from reid import protocollo, bootstrap, caso
from reid_locale import foto_originale, inlier, galleria_per_query, valuta_punteggi, sveglio, N_PUNTI, FONDO_MIN

OUT_DIR = os.path.join("runs", "reid_aliked")
EROSIONE = 8
FASCE = [("fino a 30 giorni", 0, 30), ("31-90 giorni", 31, 90), ("91-365 giorni", 91, 365), ("1-2 anni", 366, 730),
         ("oltre 2 anni", 731, 10 ** 6)]
GRIGLIA_A = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
GRIGLIA_B = [0, 0.25, 0.5, 1.0, 2.0, 4.0, 100.0]


def nome_file(insieme, maschera):
    return os.path.join(OUT_DIR, f"punteggi_{insieme}{'_maschera' if maschera else ''}.npz")


# --- ALIKED + LightGlue

def maschera_pesce(im, device):
    t = torch.from_numpy(im).to(device)
    fg = (t.min(dim=-1).values <= FONDO_MIN).float()[None, None]
    return (-F.max_pool2d(-fg, 2 * EROSIONE + 1, stride=1, padding=EROSIONE))[0, 0] > 0.5


@torch.no_grad()
def estrai(estr, records, device, maschera):
    n = len(records)
    kp = torch.zeros(n, N_PUNTI, 2, device=device)
    de = torch.zeros(n, N_PUNTI, 128, device=device, dtype=torch.half)
    dim = torch.zeros(n, 2, device=device)
    dentro = np.ones((n, N_PUNTI), bool)
    area = []
    for i, r in enumerate(tqdm(records, desc="ALIKED")):
        im = foto_originale(r["path"])
        x = torch.from_numpy(im).permute(2, 0, 1).float().div(255)[None].to(device)
        H, W = im.shape[:2]
        if maschera:
            m = maschera_pesce(im, device)
            fmap, smap = estr.extract_dense_map(x)
            kpts, _, _ = estr.dkd(smap * m[None, None].float())
            d, _ = estr.desc_head(fmap, kpts)
            k = torch.tensor([W - 1, H - 1], device=device) * (kpts[0] + 1) / 2 + estr.coords_offset
            d = d[0]
            xi, yi = k[:, 0].round().clamp(0, W - 1).long(), k[:, 1].round().clamp(0, H - 1).long()
            dentro[i] = m[yi, xi].cpu().numpy()
            area.append(m.float().mean().item())
        else:
            f = estr({"image": x})
            k, d = f["keypoints"][0], f["descriptors"][0]
        kp[i], de[i] = k, d.half()
        dim[i] = torch.tensor([W, H], device=device)
    return kp, de, dim, dentro, area


def esegui_aliked(insieme, maschera, device, batch=64, limite=0):
    from lightglue import ALIKED, LightGlue
    records = carica_split(build_records("head"))[insieme]
    galleria, query = protocollo(records)
    if limite:
        query = query[:limite]
    path = nome_file(insieme, maschera)
    parziale = path.replace(".npz", "_parziale.npz")
    P = np.full((len(query), len(galleria)), np.nan)
    fatte = 0
    if os.path.exists(parziale):
        z = np.load(parziale)
        P, fatte = z["P"], int(z["fatte"])
        print(f"riprendo da {fatte} query su {len(query)}", flush=True)
    t0 = time.time()
    estr = ALIKED(max_num_keypoints=N_PUNTI, detection_threshold=-1).eval().to(device)
    kp, de, dim, dentro, area = estrai(estr, records, device, maschera)
    del estr
    torch.cuda.empty_cache()
    info = {"insieme": insieme, "maschera": maschera, "foto": len(records), "query": len(query), "galleria": len(galleria)}
    if maschera:
        info["area della maschera (mediana, % della foto)"] = round(100 * float(np.median(area)), 1)
        info["punti dentro la maschera (%)"] = round(100 * float(dentro.mean()), 2)
        info["foto con meno di 1024 punti dentro la maschera"] = int((dentro.sum(1) < N_PUNTI).sum())
    print(info, f"estrazione {time.time() - t0:.0f} s", flush=True)
    matcher = LightGlue(features="aliked", depth_confidence=-1, width_confidence=-1, mp=True).eval().to(device)
    kp_np = kp.cpu().numpy()
    sel = galleria_per_query(records, galleria, query)
    g_np = np.array(galleria)
    os.makedirs(OUT_DIR, exist_ok=True)
    with ThreadPoolExecutor(os.cpu_count()) as pool, torch.no_grad():
        for k in tqdm(range(fatte, len(query)), desc=f"LightGlue {insieme}{' maschera' if maschera else ''}",
                      initial=fatte, total=len(query)):
            q = query[k]
            gi = g_np[sel[k]]
            lav = []
            for c in range(0, len(gi), batch):
                b = torch.from_numpy(gi[c:c + batch]).to(device)
                B = len(b)
                out = matcher({"image0": {"keypoints": kp[q].expand(B, -1, -1), "descriptors": de[q].expand(B, -1, -1),
                                          "image_size": dim[q].expand(B, -1)},
                               "image1": {"keypoints": kp[b], "descriptors": de[b], "image_size": dim[b]}})
                for r, g in enumerate(gi[c:c + batch]):
                    m = out["matches"][r].cpu().numpy()
                    if maschera:
                        m = m[dentro[q, m[:, 0]] & dentro[g, m[:, 1]]]
                    lav.append(pool.submit(inlier, kp_np[q, m[:, 0]], kp_np[g, m[:, 1]]))
            P[k, sel[k]] = [l.result() for l in lav]
            if (k + 1) % 25 == 0:
                np.savez(parziale, P=P, fatte=k + 1)
    np.savez_compressed(path, P=P, info=json.dumps(info))
    if os.path.exists(parziale):
        os.remove(parziale)
    print(f"{insieme}{' maschera' if maschera else ''}: {len(query)} query in {time.time() - t0:.0f} s -> {path}", flush=True)


# --- analisi

def metadati():
    out = {}
    with open("Melops_metadata.txt", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            out[r["filename_year"]] = {"length": float(r["length"]) if r["length"] not in ("", "NA") else np.nan,
                                       "sex": r["sex"], "spawning": r["spawning"]}
    return out


def giorno(s):
    return datetime.strptime(s, "%d.%m.%Y")


def misure(righe, chiavi=("r1", "r5", "ap")):
    fish = [r["fish"] for r in righe]
    out = {}
    for k, nome in zip(chiavi, ("rank-1", "rank-5", "mAP")):
        v = [r[k] for r in righe]
        lo, hi = bootstrap(fish, v)
        out[nome] = [round(100 * float(np.mean(v)), 2), round(100 * lo, 2), round(100 * hi, 2)]
    return out


def differenza(a, b):
    fish = [r["fish"] for r in a]
    out = {}
    for k, nome in (("r1", "rank-1"), ("r5", "rank-5"), ("ap", "mAP")):
        d = [x[k] - y[k] for x, y in zip(a, b)]
        lo, hi = bootstrap(fish, d)
        out[nome] = [round(100 * float(np.mean(d)), 2), round(100 * lo, 2), round(100 * hi, 2)]
    return out


def per_fascia(righe):
    out = {}
    for nome, lo, hi in FASCE:
        sel = [r for r in righe if lo <= r["giorni"] <= hi]
        if sel:
            out[nome] = {"query": len(sel), **{k: v[0] for k, v in misure(sel).items()}}
    return out


class Coppie:
    """Per un insieme: per ogni coppia query-galleria dello stesso lato, inlier e metadati."""

    def __init__(self, records, galleria, query, P, meta):
        self.records, self.galleria, self.query, self.P = records, galleria, query, P
        self.sel = galleria_per_query(records, galleria, query)
        g = np.array(galleria)
        self.ids = np.array([r["fish_id"] for r in records])
        self.data = np.array([giorno(r["date"]).toordinal() for r in records])
        self.lung = np.array([meta[r["stem"]]["length"] for r in records])
        self.sesso = np.array([meta[r["stem"]]["sex"] for r in records])
        self.ripr = np.array([meta[r["stem"]]["spawning"] for r in records])
        self.gal = [g[s] for s in self.sel]

    def punteggi(self, crescita_annua, a=0.0, b=0.0):
        """Matrice dei punteggi riordinati (NaN per l'altro lato)."""
        out = np.full_like(self.P, np.nan)
        for k, q in enumerate(self.query):
            gi = self.gal[k]
            s = np.log1p(self.P[k, self.sel[k]])
            if a:
                attesa = crescita_annua * (self.data[q] - self.data[gi]) / 365
                s = s - a * np.abs((self.lung[q] - self.lung[gi]) - attesa) / 10
            if b:
                s = s - b * (self.sesso[gi] != self.sesso[q])
            out[k, self.sel[k]] = s
        return out


def crescita_annua_train(records, meta):
    date = defaultdict(dict)
    for r in records:
        date[r["fish_id"]].setdefault(giorno(r["date"]), meta[r["stem"]]["length"])
    v = []
    for d in date.values():
        ordine = sorted(d)
        for x, y in zip(ordine[:-1], ordine[1:]):
            giorni = (y - x).days
            if giorni >= 180 and not np.isnan(d[x]) and not np.isnan(d[y]):
                v.append(365 * (d[y] - d[x]) / giorni)
    return float(np.median(v)), len(v)


def dove_sbaglia(C, righe):
    """Per query: coppia vera piu' vicina nel tempo (stesso lato) e coppia falsa con piu' inlier."""
    tab = []
    for k, (q, r) in enumerate(zip(C.query, righe)):
        gi = C.gal[k]
        p = C.P[k, C.sel[k]]
        vere = C.ids[gi] == C.ids[q]
        j = gi[vere][np.argmin(C.data[q] - C.data[gi[vere]])]
        tab.append({"r1": r["r1"], "r5": r["r5"], "ap": r["ap"], "fish": r["fish"],
                    "giorni": int(C.data[q] - C.data[j]), "crescita": C.lung[q] - C.lung[j],
                    "sesso": C.sesso[q], "riproduttivo": C.ripr[q],
                    "cambio riproduttivo": "uguale" if C.ripr[q] == C.ripr[j] else "diverso",
                    "inlier vera": float(p[gi == j][0]), "inlier falsa max": float(p[~vere].max())})
    gruppi = {
        "intervallo": [(n, lambda t, lo=lo, hi=hi: lo <= t["giorni"] <= hi) for n, lo, hi in FASCE],
        "crescita (mm)": [("in calo", lambda t: t["crescita"] < 0), ("0-10", lambda t: 0 <= t["crescita"] <= 10),
                          ("11-20", lambda t: 10 < t["crescita"] <= 20), ("21-40", lambda t: 20 < t["crescita"] <= 40),
                          ("oltre 40", lambda t: t["crescita"] > 40)],
        "sesso": [(s, lambda t, s=s: t["sesso"] == s) for s in ("m", "f", "s")],
        "stato riproduttivo (query)": [(s, lambda t, s=s: t["riproduttivo"] == s) for s in ("0", "1", "2")],
        "stato riproduttivo tra le catture": [(s, lambda t, s=s: t["cambio riproduttivo"] == s) for s in ("uguale", "diverso")],
        "sesso e crescita (mm)": [(f"{s}, {n}", lambda t, s=s, lo=lo, hi=hi: t["sesso"] == s and lo <= max(t["crescita"], 0) <= hi)
                                  for s in ("m", "f") for n, lo, hi in (("fino a 10", 0, 10), ("11-20", 10.01, 20),
                                                                         ("21-40", 20.01, 40), ("oltre 40", 40.01, 1e9))],
    }
    out = {}
    for gruppo, voci in gruppi.items():
        out[gruppo] = {}
        for nome, filtro in voci:
            sel = [t for t in tab if filtro(t)]
            if not sel:
                continue
            out[gruppo][nome] = {"query": len(sel), "rank-1": round(100 * np.mean([t["r1"] for t in sel]), 2),
                                 "mAP": round(100 * np.mean([t["ap"] for t in sel]), 2),
                                 "inlier coppia vera (mediana)": float(np.median([t["inlier vera"] for t in sel])),
                                 "inlier miglior coppia falsa (mediana)": float(np.median([t["inlier falsa max"] for t in sel]))}
    return out, tab


def auc_coppie(C):
    """Probabilita' che una coppia vera abbia piu' inlier di una falsa della stessa query (pari = meta')."""
    v = []
    for k, q in enumerate(C.query):
        gi = C.gal[k]
        p = C.P[k, C.sel[k]]
        vere = C.ids[gi] == C.ids[q]
        f = np.sort(p[~vere])
        for x in p[vere]:
            v.append((np.searchsorted(f, x, "left") + np.searchsorted(f, x, "right")) / 2 / len(f))
    return float(np.mean(v))


def figura_inlier(C, path):
    vere, false = [], []
    for k, q in enumerate(C.query):
        gi = C.gal[k]
        p = C.P[k, C.sel[k]]
        m = C.ids[gi] == C.ids[q]
        vere += p[m].tolist()
        false += p[~m].tolist()
    bins = np.arange(0, max(max(vere), 60) + 5, 5)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(false, bins=bins, color="#b8c0c8", density=True, histtype="stepfilled",
            label=f"coppie di pesci diversi ({len(false)})", log=True)
    ax.hist(vere, bins=bins, color="#1f4e79", density=True, histtype="step", linewidth=2,
            label=f"coppie dello stesso pesce ({len(vere)})", log=True)
    ax.set_xlabel("inlier")
    ax.set_ylabel("frequenza (scala logaritmica)")
    ax.set_title("ALIKED + LightGlue: inlier delle coppie vere e false")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def analisi(limite=0):
    tutti = build_records("head")
    insiemi = carica_split(tutti)
    meta = metadati()
    r_anno, n_coppie = crescita_annua_train(insiemi["train"], meta)
    out = {"crescita annua mediana (train, mm/anno)": round(r_anno, 2), "coppie di catture usate": n_coppie}
    C, righe = {}, {}
    for insieme in ("validation", "test"):
        p = nome_file(insieme, False)
        if not os.path.exists(p):
            continue
        recs = insiemi[insieme]
        galleria, query = protocollo(recs)
        if limite:
            query = query[:limite]
        C[insieme] = Coppie(recs, galleria, query, np.load(p)["P"], meta)
        righe[insieme] = valuta_punteggi(C[insieme].P, recs, galleria, query)
    lines = [f"ALIKED + LightGlue + RANSAC sul nuovo split. Crescita annua mediana nei pesci di train: {r_anno:.1f} mm/anno "
             f"({n_coppie} coppie di catture)", ""]
    if "test" in C:
        T = C["test"]
        rt = righe["test"]
        out["test"] = {"query": len(T.query), "galleria": len(T.galleria), "misure": misure(rt), "per fascia": per_fascia(rt),
                       "classifica casuale": [round(x, 2) for x in caso(rt)], "AUC coppie vere contro false": round(auc_coppie(T), 4)}
        lines += [f"TEST: {len(T.query)} query, galleria {len(T.galleria)} foto",
                  "ALIKED + LightGlue: " + " | ".join(f"{k} {v[0]:.2f} ({v[1]:.2f}-{v[2]:.2f})" for k, v in out["test"]["misure"].items()),
                  f"classifica casuale: rank-1 {out['test']['classifica casuale'][0]:.2f} | mAP {out['test']['classifica casuale'][2]:.2f}",
                  f"AUC coppie vere contro false: {out['test']['AUC coppie vere contro false']:.4f}", "per fascia di intervallo:"]
        lines += [f"  {n}: {v}" for n, v in out["test"]["per fascia"].items()]
        tab, _ = dove_sbaglia(T, rt)
        out["dove sbaglia"] = tab
        lines += ["", "DOVE SBAGLIA (test): per query, coppia vera piu' vicina nel tempo e miglior coppia falsa"]
        for gruppo, voci in tab.items():
            lines.append(f"  {gruppo}:")
            lines += [f"    {n}: {v}" for n, v in voci.items()]
        os.makedirs(OUT_DIR, exist_ok=True)
        figura_inlier(T, os.path.join(OUT_DIR, "inlier_vere_false.png"))

        pm = nome_file("test", True)
        if os.path.exists(pm):
            Pm = np.load(pm)["P"]
            info = json.loads(str(np.load(pm)["info"]))
            rm = valuta_punteggi(Pm, T.records, T.galleria, T.query)
            out["maschera"] = {"info": info, "misure": misure(rm), "per fascia": per_fascia(rm), "differenza": differenza(rm, rt),
                               "differenza per fascia": {n: differenza([a for a, b in zip(rm, rt) if lo <= b["giorni"] <= hi],
                                                                       [b for b in rt if lo <= b["giorni"] <= hi])
                                                         for n, lo, hi in FASCE}}
            lines += ["", f"MASCHERA DEL PESCE (test): {info}",
                      "ALIKED nella maschera: " + " | ".join(f"{k} {v[0]:.2f} ({v[1]:.2f}-{v[2]:.2f})" for k, v in out["maschera"]["misure"].items()),
                      "differenza maschera - senza: " + " | ".join(f"{k} {v[0]:+.2f} ({v[1]:+.2f} .. {v[2]:+.2f})" for k, v in out["maschera"]["differenza"].items()),
                      "per fascia:"]
            lines += [f"  {n}: {out['maschera']['per fascia'].get(n)} | differenza {out['maschera']['differenza per fascia'][n]}" for n, _, _ in FASCE]

    if "validation" in C and "test" in C:
        V, T = C["validation"], C["test"]
        scelte = {}
        for nome, ga, gb in (("lunghezza", GRIGLIA_A, [0]), ("sesso", [0], GRIGLIA_B), ("lunghezza e sesso", GRIGLIA_A, GRIGLIA_B)):
            migliore = None
            for a in ga:
                for b in gb:
                    mv = np.mean([r["ap"] for r in valuta_punteggi(V.punteggi(r_anno, a, b), V.records, V.galleria, V.query)])
                    if migliore is None or mv > migliore[0]:
                        migliore = (mv, a, b)
            scelte[nome] = migliore
        base_val = np.mean([r["ap"] for r in righe["validation"]])
        out["metadati"] = {"validation: mAP senza metadati": round(100 * base_val, 2)}
        lines += ["", f"METADATI SOPRA ALIKED: pesi scelti sulla validation (mAP senza metadati {100 * base_val:.2f})"]
        for nome, (mv, a, b) in scelte.items():
            rr = valuta_punteggi(T.punteggi(r_anno, a, b), T.records, T.galleria, T.query)
            voce = {"a (lunghezza)": a, "b (sesso)": b, "validation mAP": round(100 * mv, 2), "test": misure(rr),
                    "differenza con ALIKED da solo": differenza(rr, righe["test"]), "per fascia": per_fascia(rr)}
            out["metadati"][nome] = voce
            lines += [f"{nome}: a = {a}, b = {b} | validation mAP {100 * mv:.2f}",
                      "  test: " + " | ".join(f"{k} {v[0]:.2f} ({v[1]:.2f}-{v[2]:.2f})" for k, v in voce["test"].items()),
                      "  differenza con ALIKED da solo: " + " | ".join(f"{k} {v[0]:+.2f} ({v[1]:+.2f} .. {v[2]:+.2f})"
                                                                     for k, v in voce["differenza con ALIKED da solo"].items())]
    text = "\n".join(lines)
    print(text)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "risultati.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    with open(os.path.join(OUT_DIR, "risultati.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False, default=float)


def main():
    ap = argparse.ArgumentParser(description="ALIKED + LightGlue sul nuovo split")
    ap.add_argument("prova", choices=["aliked", "analisi"])
    ap.add_argument("--insieme", choices=["validation", "test"], default="test")
    ap.add_argument("--maschera", action="store_true")
    ap.add_argument("--prova_veloce", type=int, default=0, help="solo le prime N query, risultati in *_veloce")
    args = ap.parse_args()
    sveglio()
    global OUT_DIR
    if args.prova_veloce:
        OUT_DIR += "_veloce"
    if args.prova == "analisi":
        analisi(args.prova_veloce)
        return
    esegui_aliked(args.insieme, args.maschera, torch.device("cuda"), limite=args.prova_veloce)


if __name__ == "__main__":
    main()
