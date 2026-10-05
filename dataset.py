import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def load_metadata(metadata_path):
    """
    Legge il file dei metadati e restituisce un dizionario {filename_base: fish_id}.
    """
    filename_to_id = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            next(f)  # Salta intestazione
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 5:
                    fname = parts[0].strip('"')
                    true_id = parts[4].strip('"')
                    filename_to_id[fname] = true_id
    return filename_to_id


def build_images_by_id(data_dir, filename_to_id):
    """
    Raccoglie tutte le immagini in data_dir e le raggruppa per ID pesce.
    Restituisce:
        images_by_id: dict {fish_id: [path, ...]}
        image_to_id:  dict {path: fish_id}
    """
    images_by_id = {}
    image_to_id = {}

    if not os.path.exists(data_dir):
        print(f"ATTENZIONE: Cartella dataset non trovata: {data_dir}")
        return images_by_id, image_to_id

    for filename in sorted(os.listdir(data_dir)):
        if not (filename.endswith(".jpg") or filename.endswith(".png")):
            continue
        base_name = os.path.splitext(filename)[0]
        fish_id = filename_to_id.get(base_name, base_name)
        path = os.path.join(data_dir, filename)

        if fish_id not in images_by_id:
            images_by_id[fish_id] = []
        images_by_id[fish_id].append(path)
        image_to_id[path] = fish_id

    return images_by_id, image_to_id


def split_ids_by_identity(images_by_id, test_ratio=0.2, min_images_per_id=2, seed=42):
    """
    Divide le identita' (non le immagini!) in train e test set.

    Principio fondamentale del Re-ID: le identita' di test non devono MAI
    essere state viste durante il training. Questo testa la capacita' del
    modello di generalizzare a nuovi individui.

    Args:
        images_by_id: dict {fish_id: [path, ...]}
        test_ratio:   frazione di identita' da riservare al test (default 20%)
        min_images_per_id: soglia minima di immagini per includere un'identita'
        seed:         seed per la riproducibilita' dello split

    Returns:
        train_ids: set di fish_id per il training
        test_ids:  set di fish_id per il test
    """
    # Escludiamo identita' con troppo poche immagini (non utili per coppie positive)
    valid_ids = sorted([fid for fid, imgs in images_by_id.items()
                        if len(imgs) >= min_images_per_id])

    rng = random.Random(seed)
    rng.shuffle(valid_ids)

    n_test = max(1, int(len(valid_ids) * test_ratio))
    test_ids = set(valid_ids[:n_test])
    train_ids = set(valid_ids[n_test:])

    return train_ids, test_ids


class RealFishDataset(Dataset):
    """
    Dataset per il Contrastive Learning sul dataset Melops.

    Utilizza SOLO le identita' nel sottoinsieme `id_subset` (train o test).
    Cio' garantisce che le identita' di test non vengano mai viste in training.

    Restituisce coppie (img1, img2, label):
        label = 1.0  -> stesso pesce (coppia positiva)
        label = 0.0  -> pesce diverso (coppia negativa)
    """
    def __init__(self, part="head",
                 base_dataset_dir="dataset",
                 metadata_path="Melops_metadata.txt",
                 img_size=224,
                 id_subset=None,
                 transform=None):

        if part not in ["head", "body", "full"]:
            raise ValueError("Il parametro 'part' deve essere 'head', 'body', o 'full'.")

        folder_name = f"Melops_{part}"
        self.data_dir = os.path.join(base_dataset_dir, folder_name, folder_name)
        self.part = part
        self.img_size = img_size

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

        # Carica metadati e costruisce indice completo
        filename_to_id = load_metadata(metadata_path)
        all_images_by_id, self.image_to_id = build_images_by_id(self.data_dir, filename_to_id)

        # Filtra per il sottoinsieme di identita' (train o test)
        if id_subset is not None:
            self.images_by_id = {fid: imgs for fid, imgs in all_images_by_id.items()
                                 if fid in id_subset}
        else:
            self.images_by_id = all_images_by_id

        self.fish_ids = list(self.images_by_id.keys())
        self.all_images = [p for imgs in self.images_by_id.values() for p in imgs]

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        anchor_path = self.all_images[idx]
        anchor_id = self.image_to_id[anchor_path]

        same_fish = random.random() > 0.5

        if same_fish and len(self.images_by_id.get(anchor_id, [])) > 1:
            positive_path = random.choice(
                [p for p in self.images_by_id[anchor_id] if p != anchor_path]
            )
            compare_path = positive_path
            label = 1.0
        else:
            negative_id = random.choice([i for i in self.fish_ids if i != anchor_id])
            compare_path = random.choice(self.images_by_id[negative_id])
            label = 0.0

        img1 = self.transform(Image.open(anchor_path).convert('RGB'))
        img2 = self.transform(Image.open(compare_path).convert('RGB'))

        return img1, img2, torch.tensor(label, dtype=torch.float32)


class GalleryDataset(Dataset):
    """
    Dataset per la fase di valutazione (Gallery + Query).

    Restituisce singole immagini con il loro fish_id, per permettere
    il calcolo delle metriche Re-ID standard (Rank-1, mAP, CMC Curve).
    Usato SOLO con identita' di test (mai viste in training).
    """
    def __init__(self, part="head",
                 base_dataset_dir="dataset",
                 metadata_path="Melops_metadata.txt",
                 img_size=224,
                 id_subset=None,
                 transform=None):

        folder_name = f"Melops_{part}"
        data_dir = os.path.join(base_dataset_dir, folder_name, folder_name)

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

        filename_to_id = load_metadata(metadata_path)
        all_images_by_id, _ = build_images_by_id(data_dir, filename_to_id)

        if id_subset is not None:
            images_by_id = {fid: imgs for fid, imgs in all_images_by_id.items()
                            if fid in id_subset}
        else:
            images_by_id = all_images_by_id

        # Lista piatta di (path, fish_id) per iterazione semplice
        self.samples = [
            (path, fid)
            for fid, paths in images_by_id.items()
            for path in paths
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, fish_id = self.samples[idx]
        img = self.transform(Image.open(path).convert('RGB'))
        return img, fish_id

