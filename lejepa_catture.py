"""
Invarianza tra catture con la loss di LeJEPA: le due viste globali sono due catture dello stesso pesce e lato in date
diverse (pesci di train con almeno due date), senza condizionamento sul tempo.
  --da dinov2_puro | dinov2 | zero   backbone di partenza: DINOv2, oppure il modello multi-vista da DINOv2 o da zero
  --viste lejepa                     1 globale + 3 locali per cattura (default)
  --viste arcface                    un ritaglio 224 (scala 0,6-1) per cattura, come in arcface.py
  --senza_proiettore                 loss e SIGReg sul token di classe
  --campioni foto                    un elemento per foto a epoca, come arcface.py (default: uno per (pesce, lato))
  --foto_per_pesce N                 N foto dello stesso pesce per elemento, centro sulla loro media
  --repulsione W                     + W * loss di ArcFace con i centri dei pesci del batch come vettori di classe
  --repulsione_su cls                quella loss sul token di classe (come arcface.py) invece che sul proiettore
  --s_rep, --m_rep                   scala e margine di quella loss (default 64 e 0,5, come arcface.py)
  --mescola                          controllo di fuga: identita' permutate tra le foto
  --senza_etichette                  ogni foto di train con se stessa: nessuna identita' nel training

    python lejepa_catture.py --da dinov2_puro --epochs 60 --checkpointing     output: runs/lejepa_catture_<da>[_<etichetta>]/
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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from dataset import build_records
from split_fisso import carica_split
from reid_locale import sveglio
from ijepa_model import SIGReg
from lejepa_multivista import FotoCache, Viste, Modello, loss_lejepa, lr_al_passo, valuta_backbone, N_GLOBALI

N_LOCALI_PER_CATTURA = 3


class CoppieCatture(Dataset):
    """Per ogni (pesce, lato) con almeno due date: n foto, le prime due da due date diverse scelte a caso, le altre
    tra le restanti foto del pesce. per_foto: un elemento per ogni foto, che fa da prima foto."""

    def __init__(self, records, seed=0, n=2, per_foto=False, mescola=False):
        self.foto = FotoCache(records)
        per_chiave = defaultdict(lambda: defaultdict(list))
        for i, r in enumerate(records):
            per_chiave[(r["fish_id"], r["side"])][r["date"]].append(i)
        if mescola:
            # controllo di fuga: le foto dei (pesce, lato) con almeno due date ricevono identita' permutate a caso
            idx = [i for d in per_chiave.values() if len(d) >= 2 for foto in d.values() for i in foto]
            etichette = [(records[i]["fish_id"], records[i]["side"]) for i in idx]
            random.Random(123).shuffle(etichette)
            per_chiave = defaultdict(lambda: defaultdict(list))
            for i, c in zip(idx, etichette):
                per_chiave[c][records[i]["date"]].append(i)
        self.chiavi = [list(d.values()) for d in per_chiave.values() if len(d) >= 2]
        self.elementi = ([(c, a, i) for c, date in enumerate(self.chiavi) for a, foto in enumerate(date) for i in foto]
                         if per_foto else [(c, None, None) for c in range(len(self.chiavi))])
        self.n = n
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.elementi)

    def __getitem__(self, k):
        c, a, i = self.elementi[k]
        date = self.chiavi[c]
        if a is None:
            a, b = self.rng.sample(range(len(date)), 2)
            i = self.rng.choice(date[a])
        else:
            b = self.rng.choice([x for x in range(len(date)) if x != a])
        scelte = [i, self.rng.choice(date[b])]
        if self.n > 2:
            resto = [j for d in date for j in d if j not in scelte]
            extra = self.rng.sample(resto, min(self.n - 2, len(resto)))
            scelte += extra + [self.rng.choice(scelte) for _ in range(self.n - 2 - len(extra))]
        return tuple(self.foto[j][0] for j in scelte)


class StessaFoto(Dataset):
    """Senza etichette: ogni foto di train compare n volte, le viste differiscono solo per ritaglio e aumentazioni."""

    def __init__(self, records, n=2):
        self.foto = FotoCache(records)
        self.n = n

    def __len__(self):
        return len(self.foto)

    def __getitem__(self, k):
        x = self.foto[k][0]
        return tuple(x for _ in range(self.n))


@torch.no_grad()
def viste_coppia(viste, ua, ub):
    """Catture A e B -> 2 viste globali [2B, ...] (A poi B) e 6 locali [6B, ...] (3 da A, 3 da B)."""
    xa, xb = ua.float() / 255, ub.float() / 255
    g = torch.cat([viste.globale(xa), viste.globale(xb)])
    l = torch.cat([viste.locale(xa) for _ in range(N_LOCALI_PER_CATTURA)] + [viste.locale(xb) for _ in range(N_LOCALI_PER_CATTURA)])
    return (g.clamp(0, 1) - viste.mean) / viste.std, (l.clamp(0, 1) - viste.mean) / viste.std


@torch.no_grad()
def viste_coppia_arcface(ritaglio, fotometriche, viste, *foto):
    """Viste come in arcface.py: un ritaglio 224 (scala 0,6-1) per foto, nessuna vista locale."""
    g = torch.cat([fotometriche(ritaglio(u.float() / 255)) for u in foto]).clamp(0, 1)
    return (g - viste.mean) / viste.std, None


def loss_centro(z, sigreg, passo, lam):
    """loss_lejepa con il centro sulla media di tutte le viste."""
    sim = (z.mean(0) - z).square().mean()
    sig = torch.stack([sigreg(z[v], seed=passo * len(z) + v) for v in range(len(z))]).mean()
    return (1 - lam) * sim + lam * sig, sim.detach(), sig.detach()


def repulsione(z, s=64.0, m=0.5):
    """Loss di ArcFace con i centri dei pesci del batch al posto dei vettori di classe: ogni vista [V, B, K] deve essere
    piu' vicina al centro delle altre viste del proprio pesce che ai centri degli altri pesci, con margine angolare m."""
    zn = F.normalize(z, dim=-1)
    centri = F.normalize(z.mean(0), dim=-1)
    propri = F.normalize((z.sum(0, keepdim=True) - z) / (len(z) - 1), dim=-1)
    cos = zn @ centri.T
    giusto = torch.eye(z.shape[1], dtype=torch.bool, device=z.device)
    cos = torch.where(giusto, (zn * propri).sum(-1, keepdim=True).expand_as(cos), cos)
    ang = torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6))
    giusto = torch.eye(z.shape[1], dtype=torch.bool, device=z.device)
    logit = torch.where(giusto, torch.cos(ang + m), cos) * s
    y = torch.arange(z.shape[1], device=z.device)
    return torch.stack([F.cross_entropy(logit[v], y) for v in range(len(z))]).mean()


def main():
    ap = argparse.ArgumentParser(description="Invarianza tra catture con la loss di LeJEPA")
    ap.add_argument("--da", choices=["dinov2", "zero", "dinov2_puro"], required=True,
                    help="dinov2_puro: pesi di DINOv2; dinov2 / zero: modello multi-vista corrispondente")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lr_backbone", type=float, default=5e-5)
    ap.add_argument("--lam", type=float, default=0.05)
    ap.add_argument("--ogni", type=int, default=5)
    ap.add_argument("--checkpointing", action="store_true")
    ap.add_argument("--riprendi", action="store_true")
    ap.add_argument("--passi_prova", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--viste", choices=["lejepa", "arcface"], default="lejepa",
                    help="lejepa: 1 globale + 3 locali per cattura; arcface: un ritaglio per cattura")
    ap.add_argument("--senza_proiettore", action="store_true", help="loss e SIGReg sul token di classe del backbone")
    ap.add_argument("--campioni", choices=["coppie", "foto"], default="coppie",
                    help="coppie: un elemento per (pesce, lato) a epoca; foto: un elemento per foto, come arcface.py")
    ap.add_argument("--foto_per_pesce", type=int, default=2,
                    help="foto dello stesso pesce per elemento, centro sulla loro media (oltre 2 solo con --viste arcface)")
    ap.add_argument("--repulsione", type=float, default=0.0,
                    help="peso della loss di ArcFace sui centri dei pesci del batch (0 = assente)")
    ap.add_argument("--repulsione_su", choices=["proiezione", "cls"], default="proiezione",
                    help="proiezione: sulle uscite del proiettore; cls: sul token di classe del backbone, come arcface.py")
    ap.add_argument("--s_rep", type=float, default=64.0, help="scala della loss sui centri del batch")
    ap.add_argument("--m_rep", type=float, default=0.5, help="margine angolare della loss sui centri del batch")
    ap.add_argument("--mescola", action="store_true", help="controllo di fuga: identita' permutate tra le foto")
    ap.add_argument("--senza_etichette", action="store_true",
                    help="nessuna identita': ogni foto di train con se stessa, viste solo da ritagli e aumentazioni")
    ap.add_argument("--seme", type=int, default=0, help="seme del training (pesi del proiettore, coppie, viste)")
    ap.add_argument("--etichetta", default="", help="suffisso della cartella di output")
    args = ap.parse_args()
    sveglio()
    torch.manual_seed(args.seme)
    device = torch.device(args.device)
    out_dir = os.path.join("runs", f"lejepa_catture_{args.da}" + (f"_{args.etichetta}" if args.etichetta else ""))
    os.makedirs(out_dir, exist_ok=True)

    insiemi = carica_split(build_records("head"))
    assert args.foto_per_pesce == 2 or args.viste == "arcface", "--foto_per_pesce > 2 richiede --viste arcface"
    assert not args.repulsione or args.viste == "arcface", "--repulsione richiede --viste arcface"
    if args.senza_etichette:
        dati = StessaFoto(insiemi["train"], n=args.foto_per_pesce)
    else:
        dati = CoppieCatture(insiemi["train"], seed=args.seme, n=args.foto_per_pesce, per_foto=args.campioni == "foto",
                             mescola=args.mescola)
    loader = DataLoader(dati, batch_size=args.batch, shuffle=True, drop_last=True, num_workers=0, pin_memory=device.type == "cuda")
    viste = Viste().to(device)
    modello = Modello("dinov2" if args.da == "dinov2_puro" else "zero", proiettore=not args.senza_proiettore).to(device)
    if args.viste == "arcface":
        import kornia.augmentation as K
        from lejepa_multivista import LATO_GLOBALE
        ritaglio = K.RandomResizedCrop((LATO_GLOBALE, LATO_GLOBALE), scale=(0.6, 1.0)).to(device)
        fotometriche = viste.globale[1:]
        fai_viste = lambda *foto: viste_coppia_arcface(ritaglio, fotometriche, viste, *foto)
    else:
        fai_viste = lambda ua, ub: viste_coppia(viste, ua, ub)
    partenza = os.path.join("runs", f"lejepa_multivista_{args.da}", "migliore.pth")
    if args.da == "dinov2_puro":
        print("backbone: pesi di DINOv2, senza il passo 1", flush=True)
    elif os.path.exists(partenza):
        ck = torch.load(partenza, map_location="cpu", weights_only=False)
        modello.backbone.load_state_dict(ck["backbone"])
        print(f"backbone dal passo 1: {partenza}, epoca {ck['epoca']}, validation {ck['val']['mAP'][0]:.2f} mAP", flush=True)
    elif not args.passi_prova:
        raise FileNotFoundError(f"manca il modello del passo 1: {partenza}")
    if args.checkpointing:
        modello.backbone.set_grad_checkpointing(True)
    sigreg = SIGReg().to(device)
    gruppi = [{"params": modello.backbone.parameters(), "lr": args.lr_backbone, "base": args.lr_backbone}]
    gruppi.append({"params": list(modello.proj.parameters()), "lr": args.lr, "base": args.lr})   # vuoto senza proiettore
    opt = torch.optim.AdamW(gruppi, weight_decay=0.05)
    totale = args.epochs * len(loader)
    warmup = max(1, round(0.05 * totale))
    epoca0, passo, migliore = 1, 0, -1.0
    log = os.path.join(out_dir, "train_log.csv")
    print(("senza etichette" if args.senza_etichette else f"{len(dati.chiavi)} (pesce, lato) con almeno due date")
          + f", {len(dati)} elementi e {len(loader)} passi per epoca", flush=True)
    if args.riprendi and os.path.exists(os.path.join(out_dir, "ultimo.pth")):
        ck = torch.load(os.path.join(out_dir, "ultimo.pth"), map_location="cpu", weights_only=False)
        modello.load_state_dict(ck["modello"])
        opt.load_state_dict(ck["opt"])
        epoca0, passo, migliore = ck["epoca"] + 1, ck["passo"], ck["migliore"]
        print(f"riprendo dall'epoca {epoca0}", flush=True)
    elif not args.passi_prova:
        with open(log, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["epoca", "loss", "invarianza", "sigreg", "lr", "val_rank1", "val_rank5", "val_mAP", "minuti"])
        _, m, _ = valuta_backbone(modello.backbone, insiemi["validation"], device)
        print(f"[prima del training] validation: {m}", flush=True)
        with open(log, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([0, "", "", "", "", m["rank-1"][0], m["rank-5"][0], m["mAP"][0], 0])

    t0 = time.time()
    modello.train()
    for epoca in range(epoca0, args.epochs + 1):
        somme, n = np.zeros(3), 0
        for foto in loader:
            for gr in opt.param_groups:
                gr["lr"] = lr_al_passo(passo, totale, warmup, gr["base"])
            g, l = fai_viste(*(u.to(device, non_blocking=True) for u in foto))
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                if args.repulsione_su == "cls":
                    e = modello.backbone(g)
                    z = modello.proj(e).float().view(args.foto_per_pesce, len(foto[0]), -1)
                elif args.foto_per_pesce == 2:
                    z = modello(g, l)
                else:
                    z = modello.proj(modello.backbone(g)).float().view(args.foto_per_pesce, len(foto[0]), -1)
            if args.foto_per_pesce == 2:
                loss, sim, sig = loss_lejepa(z, sigreg, passo, args.lam)
            else:
                loss, sim, sig = loss_centro(z, sigreg, passo, args.lam)
            if args.repulsione:
                bersaglio = e.float().view(args.foto_per_pesce, len(foto[0]), -1) if args.repulsione_su == "cls" else z
                loss = loss + args.repulsione * repulsione(bersaglio, s=args.s_rep, m=args.m_rep)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            somme += [loss.item(), sim.item(), sig.item()]
            n, passo = n + 1, passo + 1
            if args.passi_prova and passo >= args.passi_prova:
                print(f"[prova] {passo} passi | loss {loss.item():.4f} (invarianza {sim.item():.4f}, sigreg {sig.item():.3f}) | "
                      f"{(time.time() - t0) / passo:.2f} s/passo | z {tuple(z.shape)}", flush=True)
                return
        medie = somme / max(n, 1)
        minuti = (time.time() - t0) / 60
        riga = [epoca, f"{medie[0]:.5f}", f"{medie[1]:.5f}", f"{medie[2]:.4f}", f"{opt.param_groups[1]['lr']:.2e}", "", "", "", f"{minuti:.1f}"]
        testo = f"[epoca {epoca}/{args.epochs}] loss {medie[0]:.4f} | invarianza {medie[1]:.4f} | sigreg {medie[2]:.3f} | {minuti:.0f} min"
        if epoca % args.ogni == 0 or epoca == args.epochs:
            _, m, _ = valuta_backbone(modello.backbone, insiemi["validation"], device)
            riga[5:8] = [m["rank-1"][0], m["rank-5"][0], m["mAP"][0]]
            testo += f" | validation rank-1 {m['rank-1'][0]:.2f} mAP {m['mAP'][0]:.2f}"
            if m["mAP"][0] > migliore:
                migliore = m["mAP"][0]
                torch.save({"backbone": modello.backbone.state_dict(), "epoca": epoca, "val": m, "args": vars(args)},
                           os.path.join(out_dir, "migliore.pth"))
            torch.save({"modello": modello.state_dict(), "opt": opt.state_dict(), "epoca": epoca, "passo": passo,
                        "migliore": migliore}, os.path.join(out_dir, "ultimo.pth"))
        print(testo, flush=True)
        with open(log, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(riga)

    ck = torch.load(os.path.join(out_dir, "migliore.pth"), map_location="cpu", weights_only=False)
    modello.backbone.load_state_dict(ck["backbone"])
    _, m_test, pf = valuta_backbone(modello.backbone, insiemi["test"], device)
    ris = {"da": args.da, "epoca scelta (validation)": ck["epoca"], "validation": ck["val"], "test": m_test, "test per fascia": pf}
    with open(os.path.join(out_dir, "risultati.json"), "w", encoding="utf-8") as f:
        json.dump(ris, f, indent=1, ensure_ascii=False, default=float)
    print(f"FINE: epoca scelta {ck['epoca']} | test {m_test}", flush=True)


if __name__ == "__main__":
    main()
