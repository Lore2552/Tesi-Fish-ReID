"""
Foto del dataset Melops: record con i metadati (path, stem, fish_id, date, year, dayseq, side), resize deterministico,
cache memmap delle foto ridimensionate e Dataset PyTorch.
"""

import os
import csv
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_IMG_SIZE = {"head": (224, 224)}

PART_DIRS = {"head": os.path.join("Melops_head", "Melops_head")}


def load_metadata(metadata_path="Melops_metadata.txt"):
    meta = {}
    with open(metadata_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            meta[row["filename_year"]] = {
                "fish_id": row["ID"],
                "date": row["date"],
                "year": int(row["year"]),
                "dayseq": int(row["dayseq"]),
                "side": row["side"],
                "blurry": int(row["blurry"]) if row["blurry"] not in ("", "NA") else 0,
            }
    return meta


def build_records(part="head", base_dataset_dir="dataset", metadata_path="Melops_metadata.txt", exclude_blurry=True):
    """Record delle foto della parte richiesta presenti nei metadati (le foto mosse contrassegnate come blurry sono escluse)."""
    data_dir = os.path.join(base_dataset_dir, PART_DIRS[part])
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Cartella dataset non trovata: {data_dir}")
    meta = load_metadata(metadata_path)
    records = []
    for filename in sorted(os.listdir(data_dir)):
        stem, ext = os.path.splitext(filename)
        m = meta.get(stem)
        if ext.lower() not in (".jpg", ".jpeg", ".png") or m is None or (exclude_blurry and m["blurry"]):
            continue
        records.append({"path": os.path.join(data_dir, filename), "stem": stem, **m})
    return records


def split_fish(records, test_frac=0.1, seed=42):
    """Divisione casuale dei pesci della prima fase (test_frac dei pesci fuori dal training)."""
    fish = sorted({r["fish_id"] for r in records})
    random.Random(seed).shuffle(fish)
    test_ids = set(fish[:int(round(len(fish) * test_frac))])
    return set(fish) - test_ids, test_ids


def subset(records, ids):
    return [r for r in records if r["fish_id"] in ids]


def load_image_uint8(path, img_size):
    """Resize deterministico (bicubico) a img_size=(H, W) -> array uint8 (3, H, W)."""
    h, w = img_size
    img = Image.open(path)
    img.draft("RGB", (2 * w, 2 * h))          # decodifica JPEG gia' ridotta, molto piu' veloce
    img = img.convert("RGB").resize((w, h), Image.BICUBIC)
    return np.asarray(img, dtype=np.uint8).transpose(2, 0, 1).copy()


def normalize_uint8(x):
    """uint8 (..., 3, H, W) -> float normalizzato ImageNet."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(3, 1, 1)
    return (x.float() / 255.0 - mean) / std


def build_image_cache(part, img_size, cache_dir="cache", base_dataset_dir="dataset",
                      metadata_path="Melops_metadata.txt"):
    """Cache memmap uint8 di tutte le foto ridimensionate (creata una volta); l'indice dei nomi e' in <percorso>.txt."""
    records = build_records(part, base_dataset_dir, metadata_path, exclude_blurry=False)
    h, w = img_size
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{part}_{h}x{w}.u8")
    stems = [r["stem"] for r in records]
    if os.path.exists(path) and os.path.exists(path + ".txt"):
        with open(path + ".txt", "r", encoding="utf-8") as f:
            if f.read().split("\n") == stems:
                return path
    print(f"[dataset] Creazione cache {path} ({len(records)} foto, {len(records) * 3 * h * w / 2**30:.2f} GB)...")
    mm = np.memmap(path + ".tmp", dtype=np.uint8, mode="w+", shape=(len(records), 3, h, w))
    for i, r in enumerate(records):
        mm[i] = load_image_uint8(r["path"], img_size)
    mm.flush()
    del mm
    os.replace(path + ".tmp", path)
    with open(path + ".txt", "w", encoding="utf-8") as f:
        f.write("\n".join(stems))
    return path


class FishImages(Dataset):
    """(foto uint8 [3, H, W], indice). Il memmap si apre in ogni processo al primo accesso."""

    def __init__(self, records, img_size=(224, 224), cache_path=None):
        self.records = records
        self.img_size = tuple(img_size)
        self.cache_path = cache_path
        self._mm = None
        self.rows = None
        if cache_path is not None:
            with open(cache_path + ".txt", "r", encoding="utf-8") as f:
                stems = f.read().split("\n")
            row_of = {s: i for i, s in enumerate(stems)}
            self.n_cache = len(stems)
            self.rows = [row_of.get(r["stem"], -1) for r in records]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_mm"] = None
        return state

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.rows[idx] if self.rows is not None else -1
        if row >= 0:
            if self._mm is None:
                self._mm = np.memmap(self.cache_path, dtype=np.uint8, mode="r", shape=(self.n_cache, 3, *self.img_size))
            x = np.array(self._mm[row])
        else:
            x = load_image_uint8(self.records[idx]["path"], self.img_size)
        return torch.from_numpy(x), idx
