"""
Re-identificazione con confronto locale, stesso protocollo di reid.py; cambia il punteggio di una coppia di foto:
  sift   : SIFT (1024 punti) + rapporto di Lowe 0,8 + omografia RANSAC (MAGSAC++, 4 px); punteggio = inlier
  aliked : ALIKED (1024 punti) + LightGlue + stesso RANSAC
  token  : vicini reciproci tra i token di patch dell'encoder (tutti, o solo quelli del pesce)
Contiene anche funzioni di valutazione usate dagli altri script (valuta_punteggi, galleria_per_query, sveglio).

    python reid_locale.py aliked | sift | token --runs 4 6 7 8 | riepilogo        output: runs/reid_locale/
"""

import os
import re
import csv
import time
import ctypes
import argparse
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from dataset import normalize_uint8
from ijepa_model import build_model
from analisi_collasso import carica_foto, encoder_run, DinoV2
from reid import protocollo, data, bootstrap, caso, figura_esempi, FASCE

OUT_DIR = os.path.join("runs", "reid_locale")
LATO_MAX = 640          # lato lungo massimo per le feature locali
N_PUNTI = 1024          # punti per foto, SIFT e ALIKED
SOGLIA_SIFT = 0.0067   # contrasto minimo dei punti SIFT, come in LightGlue
RAPPORTO_LOWE = 0.8
SOGLIA_RANSAC = 4.0     # pixel
FONDO_MIN = 200         # sfondo: tutti e tre i canali oltre 200
SEED = 0

''' Utility per non far andare il pc in sospensione'''
def sveglio():
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


def nome_file(nome):
    return re.sub(r"[^a-z0-9]+", "_", nome.lower()).strip("_")


def galleria_per_query(records, galleria, query):
    """Per ogni query, le posizioni (nella lista galleria) delle foto dello stesso lato."""
    lati = np.array([r["side"] for r in records])
    g_all = np.array(galleria)
    return [np.flatnonzero(lati[g_all] == lati[q]) for q in query]


def salva(path, punteggi):
    np.savez_compressed(path, **{nome_file(k): v for k, v in punteggi.items()}, nomi=np.array(list(punteggi)))


def carica(path):
    z = np.load(path)
    return {str(n): z[nome_file(str(n))] for n in z["nomi"]}


# --- valutazione

def valuta_punteggi(P, records, galleria, query, seed=SEED):
    """P [query, galleria], NaN per l'altro lato."""
    ids = np.array([r["fish_id"] for r in records])
    lati = np.array([r["side"] for r in records])
    g_all = np.array(galleria)
    rng = np.random.default_rng(seed)
    righe = []
    for k, q in enumerate(query):
        sel = lati[g_all] == lati[q]
        g, s = g_all[sel], P[k, sel]
        ordine = np.lexsort((rng.random(len(g)), -s))
        giusto = ids[g][ordine] == ids[q]
        pos = np.flatnonzero(giusto)
        ap = float(np.mean(np.arange(1, len(pos) + 1) / (pos + 1)))
        giorni = min((data(records[q]) - data(records[i])).days for i in g[ids[g] == ids[q]])
        righe.append({"q": q, "fish": ids[q], "r1": float(giusto[0]), "r5": float(giusto[:5].any()), "ap": ap,
                      "giorni": giorni, "n_gal": len(g), "n_pos": len(pos), "primi": g[ordine[:5]].tolist(),
                      "punteggio_1": float(s[ordine[0]]), "punteggio_giusto": float(s[ids[g] == ids[q]].max())})
    return righe


# --- vicini reciproci tra token

@torch.no_grad()
def token_e_maschere(encoder, ds, device, patch, batch_size=32):
    """Token normalizzati [foto, patch, D], maschera del pesce [foto, patch], media dei token."""
    tok, mas, media = [], [], []
    for i in tqdm(range(0, len(ds), batch_size), desc="token", leave=False):
        u8 = torch.stack([ds[j][0] for j in range(i, min(i + batch_size, len(ds)))]).to(device)
        t = encoder(normalize_uint8(u8)).float()
        media.append(F.normalize(t.mean(dim=1), dim=-1).cpu())
        tok.append(F.normalize(t, dim=-1).half())
        fg = (u8.min(dim=1, keepdim=True).values <= FONDO_MIN).float()
        mas.append((F.avg_pool2d(fg, patch) >= 0.5).flatten(1))
    tok, mas = torch.cat(tok), torch.cat(mas)
    assert tok.shape[1] == mas.shape[1], (tok.shape, mas.shape)
    return tok, mas, torch.cat(media).numpy()


@torch.no_grad()
def vicini_reciproci(tq, mq, tg, mg):
    """tq [n, D], mq [n]; tg [B, m, D], mg [B, m] -> [B] coppie di vicini reciproci."""
    s = torch.einsum("nd,bmd->bnm", tq, tg).float()
    s = s.masked_fill(~mq[None, :, None], float("-inf")).masked_fill(~mg[:, None, :], float("-inf"))
    nn12 = s.argmax(dim=2)                                    # [B, n]
    nn21 = s.argmax(dim=1)                                    # [B, m]
    reciproco = nn21.gather(1, nn12) == torch.arange(s.shape[1], device=s.device)[None]
    valido = mq[None] & mg.gather(1, nn12)
    return (reciproco & valido).sum(dim=1)


def esegui_token(args, ds, records, galleria, query, device):
    modelli = [encoder_run(r, args.checkpoint, device) for r in args.runs]
    modelli.append(("DINOv2 ViT-S/14", DinoV2(device)))
    for seme in range(3):
        torch.manual_seed(seme)
        modelli.append((f"encoder casuale (seme {seme})",
                        build_model({"img_size": [224, 224], "patch_size": 16, "lam": 0.05, "num_slices": 1024}).encoder.to(device).eval()))
    sel = galleria_per_query(records, galleria, query)
    g_all = torch.tensor(galleria, device=device)
    punteggi = {}
    for nome, encoder in modelli:
        patch = 14 if nome.startswith("DINOv2") else int(re.search(r"patch (\d+)", nome).group(1)) if "patch" in nome else 16
        t0 = time.time()
        tok, mas, media = token_e_maschere(encoder, ds, device, patch)
        del encoder
        tutti = torch.ones_like(mas)
        P = {v: np.full((len(query), len(galleria)), np.nan) for v in ("tutti", "pesce")}
        Pm = np.full((len(query), len(galleria)), np.nan)
        for k, q in enumerate(tqdm(query, desc=nome, leave=False)):
            gi = g_all[torch.from_numpy(sel[k]).to(device)]
            for v, m in (("tutti", tutti), ("pesce", mas)):
                out = [vicini_reciproci(tok[q], m[q], tok[gi[c:c + 128]], m[gi[c:c + 128]]) for c in range(0, len(gi), 128)]
                P[v][k, sel[k]] = torch.cat(out).cpu().numpy()
            Pm[k, sel[k]] = media[np.array(galleria)[sel[k]]] @ media[q]
        punteggi[f"{nome} | media dei token"] = Pm
        punteggi[f"{nome} | vicini reciproci, tutti i token"] = P["tutti"]
        punteggi[f"{nome} | vicini reciproci, solo pesce"] = P["pesce"]
        print(f"{nome}: {tok.shape[1]} token per foto, in media {mas.float().sum(1).mean():.0f} del pesce, "
              f"{time.time() - t0:.0f} s", flush=True)
        del tok, mas
        torch.cuda.empty_cache()
    return punteggi


# --- feature locali

def foto_originale(path):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(1.0, LATO_MAX / max(w, h))
    if s < 1:
        img = img.resize((round(w * s), round(h * s)), Image.BICUBIC)
    return np.asarray(img)


def inlier(p0, p1):
    if len(p0) < 4:
        return 0
    H, m = cv2.findHomography(p0, p1, cv2.USAC_MAGSAC, SOGLIA_RANSAC, maxIters=10000, confidence=0.999)
    return 0 if H is None or m is None else int(m.sum())


def estrai_sift(records):
    sift = cv2.SIFT_create(nfeatures=N_PUNTI, contrastThreshold=SOGLIA_SIFT)
    pts = np.zeros((len(records), N_PUNTI, 2), np.float32)
    des = np.zeros((len(records), N_PUNTI, 128), np.float32)
    n = np.zeros(len(records), int)
    for i, r in enumerate(tqdm(records, desc="SIFT")):
        kp, d = sift.detectAndCompute(cv2.cvtColor(foto_originale(r["path"]), cv2.COLOR_RGB2GRAY), None)
        if d is None:
            continue
        k = min(len(kp), N_PUNTI)
        pts[i, :k] = [p.pt for p in kp[:k]]
        des[i, :k] = d[:k]
        n[i] = k
    return pts, des, n


@torch.no_grad()
def coppie_sift(dq, nq, dg, ng):
    """Indice del corrispondente in galleria per ogni punto della query, -1 se scartato dal test di Lowe."""
    d2 = (dq * dq).sum(-1)[None, :, None] + (dg * dg).sum(-1)[:, None, :] - 2 * torch.einsum("nd,bmd->bnm", dq, dg)
    m = dg.shape[1]
    d2 = d2.masked_fill(torch.arange(m, device=dg.device)[None, None] >= ng[:, None, None], float("inf"))
    v, j = d2.clamp_min(0).topk(2, dim=2, largest=False)
    ok = v[..., 0].sqrt() < RAPPORTO_LOWE * v[..., 1].sqrt()
    ok &= torch.arange(dq.shape[0], device=dq.device)[None] < nq
    return torch.where(ok, j[..., 0], -1)


def esegui_sift(records, galleria, query, device):
    t0 = time.time()
    pts, des, n = estrai_sift(records)
    print(f"SIFT: {n.mean():.0f} punti per foto in media (minimo {n.min()}), {time.time() - t0:.0f} s", flush=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    D = torch.from_numpy(des).to(device)
    N = torch.from_numpy(n).to(device)
    sel = galleria_per_query(records, galleria, query)
    g_np = np.array(galleria)
    P = np.full((len(query), len(galleria)), np.nan)
    n_coppie = []
    with ThreadPoolExecutor(os.cpu_count()) as pool:
        for k, q in enumerate(tqdm(query, desc="SIFT, coppie e RANSAC")):
            gi = g_np[sel[k]]
            lavori = []
            for c in range(0, len(gi), 128):
                b = torch.from_numpy(gi[c:c + 128]).to(device)
                j = coppie_sift(D[q], N[q], D[b], N[b]).cpu().numpy()
                for r, g in enumerate(gi[c:c + 128]):
                    i0 = np.flatnonzero(j[r] >= 0)
                    n_coppie.append(len(i0))
                    lavori.append(pool.submit(inlier, pts[q, i0], pts[g, j[r, i0]]))
            P[k, sel[k]] = [l.result() for l in lavori]
    print(f"SIFT: in media {np.mean(n_coppie):.1f} coppie dopo il rapporto di Lowe, {time.time() - t0:.0f} s", flush=True)
    return {"SIFT + RANSAC": P}


def esegui_aliked(records, galleria, query, device, batch=64):
    from lightglue import ALIKED, LightGlue
    t0 = time.time()
    estrattore = ALIKED(max_num_keypoints=N_PUNTI, detection_threshold=-1).eval().to(device)
    kp = torch.zeros(len(records), N_PUNTI, 2, device=device)
    de = torch.zeros(len(records), N_PUNTI, 128, device=device)
    dim = torch.zeros(len(records), 2, device=device)
    with torch.no_grad():
        for i, r in enumerate(tqdm(records, desc="ALIKED")):
            img = torch.from_numpy(foto_originale(r["path"])).permute(2, 0, 1).float().div(255)[None].to(device)
            f = estrattore({"image": img})
            kp[i], de[i] = f["keypoints"][0], f["descriptors"][0]
            dim[i] = torch.tensor([img.shape[-1], img.shape[-2]], device=device)
    del estrattore
    print(f"ALIKED: {N_PUNTI} punti per foto, {time.time() - t0:.0f} s", flush=True)
    matcher = LightGlue(features="aliked", depth_confidence=-1, width_confidence=-1, mp=True).eval().to(device)
    kp_np = kp.cpu().numpy()
    sel = galleria_per_query(records, galleria, query)
    g_np = np.array(galleria)
    P = np.full((len(query), len(galleria)), np.nan)
    n_coppie = []
    with ThreadPoolExecutor(os.cpu_count()) as pool, torch.no_grad():
        for k, q in enumerate(tqdm(query, desc="LightGlue e RANSAC")):
            gi = g_np[sel[k]]
            lavori = []
            for c in range(0, len(gi), batch):
                b = torch.from_numpy(gi[c:c + batch]).to(device)
                B = len(b)
                out = matcher({"image0": {"keypoints": kp[q].expand(B, -1, -1), "descriptors": de[q].expand(B, -1, -1),
                                          "image_size": dim[q].expand(B, -1)},
                               "image1": {"keypoints": kp[b], "descriptors": de[b], "image_size": dim[b]}})
                for r, g in enumerate(gi[c:c + batch]):
                    m = out["matches"][r].cpu().numpy()
                    n_coppie.append(len(m))
                    lavori.append(pool.submit(inlier, kp_np[q, m[:, 0]], kp_np[g, m[:, 1]]))
            P[k, sel[k]] = [l.result() for l in lavori]
    print(f"LightGlue: in media {np.mean(n_coppie):.1f} coppie, {time.time() - t0:.0f} s", flush=True)
    return {"ALIKED + LightGlue + RANSAC": P}


# --- riepilogo

def riga(nome, righe):
    fish = [r["fish"] for r in righe]
    pezzi = []
    for chiave, etichetta in (("r1", "rank-1"), ("r5", "rank-5"), ("ap", "mAP")):
        v = [r[chiave] for r in righe]
        lo, hi = bootstrap(fish, v)
        pezzi.append(f"{etichetta} {100 * np.mean(v):.2f} ({100 * lo:.2f}-{100 * hi:.2f})")
    return f"{nome}: " + " | ".join(pezzi)


def differenza(nome, a, b):
    fish = [r["fish"] for r in a]
    pezzi = []
    for chiave, etichetta in (("r1", "rank-1"), ("r5", "rank-5"), ("ap", "mAP")):
        d = [x[chiave] - y[chiave] for x, y in zip(a, b)]
        lo, hi = bootstrap(fish, d)
        pezzi.append(f"{etichetta} {100 * np.mean(d):+.2f} ({100 * lo:+.2f} .. {100 * hi:+.2f})")
    return f"{nome}: " + " | ".join(pezzi)


def riepilogo(ds, records, galleria, query):
    punteggi = {}
    for f in ("punteggi_sift.npz", "punteggi_aliked.npz", "punteggi_token.npz"):
        if os.path.exists(os.path.join(OUT_DIR, f)):
            punteggi.update(carica(os.path.join(OUT_DIR, f)))
    tutte = {nome: valuta_punteggi(P, records, galleria, query) for nome, P in punteggi.items()}
    pesci_q = {records[q]["fish_id"] for q in query}
    lines = [f"Re-identificazione con confronto locale: {len(query)} query di {len(pesci_q)} pesci, galleria dello stesso lato "
             f"(in media {np.mean([r['n_gal'] for r in next(iter(tutte.values()))]):.0f} foto)",
             "Valori in percentuale; tra parentesi intervallo al 95% sui pesci", ""]
    for nome, righe in tutte.items():
        lines.append(riga(nome, righe))
    casuali = [n for n in tutte if n.startswith("encoder casuale")]
    for versione in ("media dei token", "vicini reciproci, tutti i token", "vicini reciproci, solo pesce"):
        nomi = [n for n in casuali if n.endswith(versione)]
        if nomi:
            medio = {k: np.mean([[r[k] for r in tutte[n]] for n in nomi]) for k in ("r1", "r5", "ap")}
            lines.append(f"media dei 3 encoder casuali | {versione}: rank-1 {100 * medio['r1']:.2f} | "
                         f"rank-5 {100 * medio['r5']:.2f} | mAP {100 * medio['ap']:.2f}")
    c = caso(next(iter(tutte.values())))
    lines += [f"classifica casuale: rank-1 {c[0]:.2f} | rank-5 {c[1]:.2f} | mAP {c[2]:.2f}", "", "Differenze (query per query):"]
    for nome in tutte:
        if nome.endswith("vicini reciproci, tutti i token") or nome.endswith("vicini reciproci, solo pesce"):
            base = nome.split(" | ")[0] + " | media dei token"
            lines.append(differenza(f"{nome} - media", tutte[nome], tutte[base]))
    if "SIFT + RANSAC" in tutte and "ALIKED + LightGlue + RANSAC" in tutte:
        lines.append(differenza("ALIKED + LightGlue - SIFT", tutte["ALIKED + LightGlue + RANSAC"], tutte["SIFT + RANSAC"]))
    lines += ["", "Per distanza dalla cattura precedente:"]
    for etichetta, lo, hi in FASCE:
        lines.append(f"{etichetta}:")
        for nome, righe in tutte.items():
            sel = [r for r in righe if lo <= r["giorni"] <= hi]
            lines.append(f"  {nome}: {len(sel)} query | rank-1 {100 * np.mean([r['r1'] for r in sel]):.2f} | "
                         f"rank-5 {100 * np.mean([r['r5'] for r in sel]):.2f} | mAP {100 * np.mean([r['ap'] for r in sel]):.2f}")
    lines += ["", "Punteggio della prima foto in classifica e del miglior positivo (mediana sulle query):"]
    for nome, righe in tutte.items():
        lines.append(f"  {nome}: primo {np.median([r['punteggio_1'] for r in righe]):.3g} | "
                     f"positivo {np.median([r['punteggio_giusto'] for r in righe]):.3g}")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(OUT_DIR, "risultati.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    with open(os.path.join(OUT_DIR, "risultati.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metodo", "query", "pesce", "lato", "giorni", "rank1", "rank5", "ap"])
        for nome, righe in tutte.items():
            for r in righe:
                q = records[r["q"]]
                w.writerow([nome, q["stem"], r["fish"], q["side"], r["giorni"], r["r1"], r["r5"], f"{r['ap']:.5f}"])
    for nome, righe in tutte.items():
        if not nome.startswith("encoder casuale") and not nome.endswith("media dei token"):
            figura_esempi(ds, records, righe, os.path.join(OUT_DIR, f"esempi_{nome_file(nome)}.png"), nome)
    return tutte


def main():
    ap = argparse.ArgumentParser(description="Re-identificazione con confronto locale")
    ap.add_argument("metodo", choices=["token", "sift", "aliked", "riepilogo"])
    ap.add_argument("--runs", type=int, nargs="+", default=[4, 6, 7, 8])
    ap.add_argument("--checkpoint", default="epoch_030")
    ap.add_argument("--prova", type=int, default=0, help="solo le prime N query (misura dei tempi), senza salvare")
    args = ap.parse_args()
    sveglio()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = carica_foto()
    records = ds.records
    galleria, query = protocollo(records)
    if args.metodo == "riepilogo":
        riepilogo(ds, records, galleria, query)
        return
    q_usate = query[:args.prova] if args.prova else query
    t0 = time.time()
    if args.metodo == "token":
        punteggi = esegui_token(args, ds, records, galleria, q_usate, device)
    elif args.metodo == "sift":
        punteggi = esegui_sift(records, galleria, q_usate, device)
    else:
        punteggi = esegui_aliked(records, galleria, q_usate, device)
    print(f"{args.metodo}: {len(q_usate)} query in {time.time() - t0:.0f} s", flush=True)
    if args.prova:
        for nome, P in punteggi.items():
            righe = valuta_punteggi(P, records, galleria, q_usate)
            print(f"[prova] {nome}: rank-1 {100 * np.mean([r['r1'] for r in righe]):.1f} | "
                  f"mAP {100 * np.mean([r['ap'] for r in righe]):.1f}")
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    salva(os.path.join(OUT_DIR, f"punteggi_{args.metodo}.npz"), punteggi)
    print(f"Punteggi in {OUT_DIR}/punteggi_{args.metodo}.npz", flush=True)


if __name__ == "__main__":
    main()
