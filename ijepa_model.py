"""
ijepa_model.py — Image-based Joint-Embedding Predictive Architecture
"""

import torch
import torch.nn as nn
import math
import copy

import timm
from timm.models.vision_transformer import Block

class MaskingStrategy:
    def __init__(self, input_size=14, num_targets=4, target_scale=(0.15, 0.2), target_aspect_ratio=(0.75, 1.5), context_scale=(0.85, 1.0)):
        self.input_size = input_size
        self.num_targets = num_targets
        self.target_scale = target_scale
        self.target_aspect_ratio = target_aspect_ratio
        self.context_scale = context_scale

    def _sample_block(self, scale_range, aspect_ratio_range):
        h = self.input_size
        w = self.input_size
        total = h * w
        scale = torch.empty(1).uniform_(*scale_range).item()
        ar = torch.empty(1).uniform_(*aspect_ratio_range).item()
        area = int(total * scale)
        bh = max(1, int(math.sqrt(area * ar)))
        bw = max(1, int(math.sqrt(area / ar)))
        bh = min(bh, h)
        bw = min(bw, w)
        top = torch.randint(0, h - bh + 1, (1,)).item()
        left = torch.randint(0, w - bw + 1, (1,)).item()
        indices = []
        for i in range(top, top + bh):
            for j in range(left, left + bw):
                indices.append(i * w + j)
        return set(indices)

    def __call__(self):
        total = self.input_size * self.input_size
        target_blocks = []
        all_target_indices = set()
        for _ in range(self.num_targets):
            block = self._sample_block(self.target_scale, self.target_aspect_ratio)
            target_blocks.append(block)
            all_target_indices.update(block)
        context_block = self._sample_block(self.context_scale, (1.0, 1.0))
        context_indices = sorted(context_block - all_target_indices)
        if len(context_indices) < total * 0.5:
            context_indices = sorted(set(range(total)) - all_target_indices)
        return context_indices, target_blocks

class Predictor(nn.Module):
    def __init__(self, embed_dim=768, predictor_dim=384, depth=6, num_heads=12, num_patches=196):
        super().__init__()
        self.predictor_embed = nn.Linear(embed_dim, predictor_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, predictor_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([Block(dim=predictor_dim, num_heads=num_heads, mlp_ratio=4.0, qkv_bias=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(predictor_dim)
        self.predictor_proj = nn.Linear(predictor_dim, embed_dim)

    def forward(self, context_tokens, context_indices, target_indices):
        B = context_tokens.shape[0]
        ctx = self.predictor_embed(context_tokens)
        ctx_pos = self.pos_embed[:, context_indices, :]
        ctx = ctx + ctx_pos
        n_targets = len(target_indices)
        mask_tokens = self.mask_token.expand(B, n_targets, -1)
        target_pos = self.pos_embed[:, target_indices, :]
        mask_tokens = mask_tokens + target_pos
        tokens = torch.cat([ctx, mask_tokens], dim=1)
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        predictions = tokens[:, -n_targets:, :]
        predictions = self.predictor_proj(predictions)
        return predictions

class IJEPA(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', predictor_depth=6, predictor_dim=384, ema_momentum=0.996):
        super().__init__()
        self.context_encoder = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool='')
        embed_dim = self.context_encoder.embed_dim
        num_patches = self.context_encoder.patch_embed.num_patches
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.predictor = Predictor(embed_dim=embed_dim, predictor_dim=predictor_dim, depth=predictor_depth, num_heads=12, num_patches=num_patches)
        grid_size = int(num_patches ** 0.5)
        self.masking = MaskingStrategy(input_size=grid_size)
        self.ema_momentum = ema_momentum

    @torch.no_grad()
    def update_target_encoder(self):
        m = self.ema_momentum
        for param_q, param_k in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(m).add_((1.0 - m) * param_q.data)

    def encode_context(self, x, patch_indices):
        x = self.context_encoder.patch_embed(x)
        idx = torch.tensor(patch_indices, device=x.device).long()
        x = x[:, idx, :]
        pos_embed = self.context_encoder.pos_embed[:, 1:, :]
        x = x + pos_embed[:, idx, :]
        cls_token = self.context_encoder.cls_token.expand(x.shape[0], -1, -1)
        cls_pos = self.context_encoder.pos_embed[:, :1, :]
        cls_token = cls_token + cls_pos
        x = torch.cat([cls_token, x], dim=1)
        x = self.context_encoder.pos_drop(x)
        for blk in self.context_encoder.blocks:
            x = blk(x)
        x = self.context_encoder.norm(x)    
        x = x[:, 1:, :]
        return x

    @torch.no_grad()
    def encode_target(self, x):
        x = self.target_encoder.patch_embed(x)
        pos_embed = self.target_encoder.pos_embed[:, 1:, :]
        x = x + pos_embed
        cls_token = self.target_encoder.cls_token.expand(x.shape[0], -1, -1)
        cls_pos = self.target_encoder.pos_embed[:, :1, :]
        cls_token = cls_token + cls_pos
        x = torch.cat([cls_token, x], dim=1)
        x = self.target_encoder.pos_drop(x)
        for blk in self.target_encoder.blocks:
            x = blk(x)
        x = self.target_encoder.norm(x)
        x = x[:, 1:, :]
        return x

    def forward(self, x):
        context_indices, target_blocks = self.masking()
        context_tokens = self.encode_context(x, context_indices)
        with torch.no_grad():
            target_tokens = self.encode_target(x)
            
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        for target_block in target_blocks:
            target_indices = sorted(target_block)
            pred = self.predictor(context_tokens, context_indices, target_indices)
            
            idx = torch.tensor(target_indices, device=x.device).long()
            target = target_tokens[:, idx, :]
            
            total_loss += torch.mean((pred - target) ** 2)
            all_predictions.append(pred)
            all_targets.append(target)
            
        total_loss = total_loss / len(target_blocks)
        
        # RESTITUIAMO 5 ELEMENTI COME CI SI ASPETTA IN TRAIN_IJEPA
        return total_loss, all_predictions, all_targets, context_indices, target_blocks

def build_ijepa(pretrained=True, predictor_depth=6, predictor_dim=384, ema_momentum=0.996):
    model = IJEPA(
        model_name='vit_base_patch16_224',
        predictor_depth=predictor_depth,
        predictor_dim=predictor_dim,
        ema_momentum=ema_momentum
    )
    return model
