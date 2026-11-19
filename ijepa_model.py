"""
I-JEPA con SIGReg di LeJEPA:
encoder ViT, predittore, mascheramento multi-blocco, SIGReg.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block
from torch.utils.checkpoint import checkpoint


# --- embedding posizionali 2D sin-cos ---------------------------------------
def _sincos_1d(embed_dim, pos):
    omega = np.arange(embed_dim // 2, dtype=np.float64) / (embed_dim / 2.0)
    omega = 1.0 / 10000 ** omega
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def get_2d_sincos_pos_embed(embed_dim, grid_h, grid_w):
    grid = np.meshgrid(np.arange(grid_w, dtype=np.float64), np.arange(grid_h, dtype=np.float64))  # w per primo
    grid = np.stack(grid, axis=0).reshape(2, 1, grid_h, grid_w)
    emb_h = _sincos_1d(embed_dim // 2, grid[0])
    emb_w = _sincos_1d(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)  # (grid_h * grid_w, D)


# --- maschere ---------------------------------------------------------------
def apply_masks(x, masks):
    """x [B, N, D], masks: lista di indici [B, K] -> [B * len(masks), K, D]."""
    out = []
    for m in masks:
        out.append(torch.gather(x, dim=1, index=m.unsqueeze(-1).expand(-1, -1, x.size(-1))))
    return torch.cat(out, dim=0)


def repeat_interleave_batch(x, B, repeat):
    N = len(x) // B
    return torch.cat([torch.cat([x[i * B:(i + 1) * B] for _ in range(repeat)], dim=0) for i in range(N)], dim=0)


# --- encoder e predittore ViT -------------------------------------------------
class PatchEmbed(nn.Module):
    def __init__(self, patch_size=16, in_chans=3, embed_dim=384):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


def _init_weights(m):
    """Inizializzazione dei pesi di I-JEPA."""
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class VisionTransformer(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0):
        super().__init__()
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.patch_embed = PatchEmbed(patch_size, 3, embed_dim)
        pos = get_2d_sincos_pos_embed(embed_dim, *self.grid_size)
        self.pos_embed = nn.Parameter(torch.from_numpy(pos).float().unsqueeze(0), requires_grad=False)
        self.blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True)
                                     for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.apply(_init_weights)

    def forward(self, x, masks=None):
        """x [B, 3, H, W]; masks: None (foto intera) o indici delle patch da tenere."""
        x = self.patch_embed(x) + self.pos_embed
        if masks is not None:
            x = apply_masks(x, masks)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class VisionTransformerPredictor(nn.Module):
    """Token di contesto + mask token alle posizioni dei bersagli -> token previsti."""

    def __init__(self, grid_size, embed_dim=384, predictor_embed_dim=384, depth=6, num_heads=6, mlp_ratio=4.0):
        super().__init__()
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        pos = get_2d_sincos_pos_embed(predictor_embed_dim, *grid_size)
        self.predictor_pos_embed = nn.Parameter(torch.from_numpy(pos).float().unsqueeze(0), requires_grad=False)
        self.predictor_blocks = nn.ModuleList([Block(dim=predictor_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                                     qkv_bias=True) for _ in range(depth)])
        self.predictor_norm = nn.LayerNorm(predictor_embed_dim, eps=1e-6)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim)
        self.apply(_init_weights)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, x, masks_x, masks):
        """Token di contesto [B, Kc, D] -> token previsti [B * len(masks), Kt, D]."""
        B = len(x) // len(masks_x)
        x = self.predictor_embed(x)
        x = x + apply_masks(self.predictor_pos_embed.repeat(B, 1, 1), masks_x)
        n_ctx = x.shape[1]
        pos = apply_masks(self.predictor_pos_embed.repeat(B, 1, 1), masks)
        pos = repeat_interleave_batch(pos, B, repeat=len(masks_x))
        pred_tokens = self.mask_token.repeat(pos.size(0), pos.size(1), 1) + pos
        x = torch.cat([x.repeat(len(masks), 1, 1), pred_tokens], dim=1)
        for blk in self.predictor_blocks:
            x = checkpoint(blk, x, use_reentrant=False) if torch.is_grad_enabled() else blk(x)
        x = self.predictor_norm(x)[:, n_ctx:]
        return self.predictor_proj(x)


# --- mascheramento multi-blocco -----------------------------------------------
class MultiBlockMaskGenerator:
    """
    Maschere di I-JEPA: 4 blocchi bersaglio (scala 0,15-0,2, proporzioni 0,75-1,5) e un blocco di
    contesto (scala 0,85-1) senza le patch dei bersagli; forma estratta per batch, posizione per foto.
    Scala e proporzioni sono estratte in modo indipendente e su tutte le posizioni.
    """

    def __init__(self, grid_size, enc_mask_scale=(0.85, 1.0), pred_mask_scale=(0.15, 0.2),
                 aspect_ratio=(0.75, 1.5), npred=4, min_keep=10):
        self.height, self.width = grid_size
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.npred = npred
        self.min_keep = min_keep

    def _sample_block_size(self, g, scale, aspect_ratio_scale):
        r_scale, r_ar = torch.rand(2, generator=g).tolist()
        keep = int(self.height * self.width * (scale[0] + r_scale * (scale[1] - scale[0])))
        ar = aspect_ratio_scale[0] + r_ar * (aspect_ratio_scale[1] - aspect_ratio_scale[0])
        h = min(int(round(math.sqrt(keep * ar))), self.height - 1)
        w = min(int(round(math.sqrt(keep / ar))), self.width - 1)
        return max(h, 1), max(w, 1)

    def _sample_block_mask(self, g, b_size, acceptable_regions=None):
        h, w = b_size
        tries, timeout = 0, 20
        while True:
            top = torch.randint(0, self.height - h + 1, (1,), generator=g).item()
            left = torch.randint(0, self.width - w + 1, (1,), generator=g).item()
            mask = torch.zeros((self.height, self.width), dtype=torch.int32)
            mask[top:top + h, left:left + w] = 1
            if acceptable_regions is not None:
                for k in range(max(len(acceptable_regions) - tries, 0)):
                    mask *= acceptable_regions[k]
            idx = torch.nonzero(mask.flatten()).squeeze(-1)
            if len(idx) > self.min_keep:
                break
            timeout -= 1
            if timeout == 0:
                tries, timeout = tries + 1, 20
        complement = torch.ones((self.height, self.width), dtype=torch.int32)
        complement[top:top + h, left:left + w] = 0
        return idx, complement

    def __call__(self, batch_size, seed):
        """Maschere deterministiche dato il seme: ([1 tensore [B, Kc]], [4 tensori [B, Kt]])."""
        g = torch.Generator()
        g.manual_seed(int(seed))
        p_size = self._sample_block_size(g, self.pred_mask_scale, self.aspect_ratio)
        e_size = self._sample_block_size(g, self.enc_mask_scale, (1.0, 1.0))
        if p_size[0] * p_size[1] <= self.min_keep:      # _sample_block_mask non terminerebbe
            raise ValueError(f"Blocco bersaglio {p_size} troppo piccolo per min_keep={self.min_keep}")
        masks_pred, masks_enc = [], []
        for _ in range(batch_size):
            mp, comps = [], []
            for _ in range(self.npred):
                m, c = self._sample_block_mask(g, p_size)
                mp.append(m)
                comps.append(c)
            masks_pred.append(mp)
            masks_enc.append(self._sample_block_mask(g, e_size, acceptable_regions=comps)[0])
        k_pred = min(len(m) for mp in masks_pred for m in mp)
        k_enc = min(len(m) for m in masks_enc)

        def _subsample(m, k):
            return m if len(m) == k else m[torch.randperm(len(m), generator=g)[:k].sort().values]

        masks_pred = [torch.stack([_subsample(masks_pred[b][j], k_pred) for b in range(batch_size)])
                      for j in range(self.npred)]
        masks_enc = [torch.stack([_subsample(m, k_enc) for m in masks_enc])]
        return masks_enc, masks_pred


class MaskCollator:
    """collate_fn: impila le foto e genera le maschere del batch."""

    def __init__(self, mask_generator):
        self.mask_generator = mask_generator

    def __call__(self, batch):
        imgs = torch.stack([b[0] for b in batch])
        idx = torch.tensor([b[1] for b in batch])
        masks_enc, masks_pred = self.mask_generator(len(batch), int(torch.randint(0, 2 ** 62, (1,)).item()))
        return imgs, idx, masks_enc, masks_pred


# --- SIGReg -----------------------------------------------------------------
class SIGReg(nn.Module):
    """
    SIGReg di LeJEPA: test di Epps-Pulley (17 punti su [0, 3]) sulle proiezioni di z [N, K] lungo direzioni casuali,
    media sulle direzioni. Vale circa 1 per una gaussiana isotropa.
    """

    def __init__(self, num_slices=1024, knots=17, t_max=3.0):
        super().__init__()
        self.num_slices = num_slices
        t = torch.linspace(0, t_max, knots, dtype=torch.float32)
        dt = t_max / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        phi = torch.exp(-0.5 * t ** 2)
        self.register_buffer("t", t, persistent=False)
        self.register_buffer("phi", phi, persistent=False)
        self.register_buffer("weights", weights * phi, persistent=False)

    def _ecf_sums(self, z, A):
        xt = (z @ A).unsqueeze(-1) * self.t                              # [n, M, T]
        return torch.cos(xt).sum(0), torch.sin(xt).sum(0)                 # [M, T]

    def forward(self, z, seed, sample_chunk=2048):
        with torch.autocast(device_type=z.device.type, enabled=False):
            z = z.float()
            g = torch.Generator(device=z.device)
            g.manual_seed(int(seed))
            A = torch.randn(z.shape[-1], self.num_slices, generator=g, device=z.device)
            A = A / A.norm(p=2, dim=0, keepdim=True)
            # funzione caratteristica empirica sommata a blocchi di campioni, ricalcolati nel backward
            N = z.shape[0]
            cos_sum = sin_sum = 0
            for i in range(0, N, sample_chunk):
                c, s = checkpoint(self._ecf_sums, z[i:i + sample_chunk], A, use_reentrant=False)
                cos_sum, sin_sum = cos_sum + c, sin_sum + s
            err = (cos_sum / N - self.phi) ** 2 + (sin_sum / N) ** 2
            return ((err @ self.weights) * N).mean()


# --- modello completo -------------------------------------------------------
VIT_SMALL = (384, 12, 6, 384, 6, 6)     # encoder: dim, blocchi, teste; predittore: dim, blocchi, teste


class IJEPALeJEPA(nn.Module):
    """
    h = encoder(foto intera), z = encoder(contesto), h_hat = predittore(z, posizioni dei bersagli).
    L = (1 - lambda) * ||h_hat - LN(h)[bersagli]||^2 + lambda * SIGReg(LN(h) di tutti i token del batch).
    """

    def __init__(self, img_size=(224, 224), patch_size=16, lam=0.05, num_slices=1024, pred_mask_scale=(0.15, 0.2)):
        super().__init__()
        d, depth, heads, pd, pdepth, pheads = VIT_SMALL
        self.encoder = VisionTransformer(img_size, patch_size, d, depth, heads)
        self.predictor = VisionTransformerPredictor(self.encoder.grid_size, d, pd, pdepth, pheads)
        self.sigreg = SIGReg(num_slices)
        self.mask_generator = MultiBlockMaskGenerator(self.encoder.grid_size, pred_mask_scale=tuple(pred_mask_scale))
        self.lam = lam

    def predict(self, imgs, masks_enc, masks_pred):
        """Token previsti e veri dei bersagli, entrambi [4B, Kt, D]."""
        h = self.encoder(imgs)
        z = self.encoder(imgs, masks_enc)
        h_hat = self.predictor(z, masks_enc, masks_pred)
        targets = repeat_interleave_batch(apply_masks(F.layer_norm(h, (h.size(-1),)), masks_pred),
                                          len(imgs), repeat=len(masks_enc))
        return h_hat, targets, h

    def forward(self, imgs, masks_enc, masks_pred, step):
        h_hat, targets, h = self.predict(imgs, masks_enc, masks_pred)
        pred_loss = (h_hat.float() - targets.float()).square().mean()
        sig = self.sigreg(F.layer_norm(h, (h.size(-1),)).reshape(-1, h.size(-1)), seed=step)
        loss = (1 - self.lam) * pred_loss + self.lam * sig
        return {"loss": loss, "pred_loss": pred_loss.detach(), "sigreg": sig.detach(), "tokens": h.detach()}


def build_model(cfg):
    return IJEPALeJEPA(img_size=tuple(cfg["img_size"]), patch_size=cfg["patch_size"], lam=cfg["lam"],
                       num_slices=cfg["num_slices"], pred_mask_scale=cfg.get("pred_mask_scale", (0.15, 0.2)))


def load_checkpoint(path, device):
    """(modello in eval, cfg) da un checkpoint della prima fase."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["cfg"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt["cfg"]
