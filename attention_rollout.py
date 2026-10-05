import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

def rollout(attentions, discard_ratio=0.9, head_fusion="mean"):
    """
    Implementazione dell'Attention Rollout per Vision Transformers.
    :param attentions: lista di tensori dell'attenzione estratti dai vari layer del ViT.
    :param discard_ratio: percentuale di pixel con attenzione minore da scartare.
    :param head_fusion: metodo per fondere le diverse attention head ("mean", "max", "min").
    """
    # Matrice identità da cui partiamo
    result = torch.eye(attentions[0].size(-1)).to(attentions[0].device)
    
    with torch.no_grad():
        for attention in attentions:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise ValueError("Metodo di fusione non supportato")

            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
            indices = indices[indices != 0] # non droppare mai l'attenzione del CLS token stesso
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1)).to(attention.device)
            a = (attention_heads_fused + 1.0 * I) / 2
            a = a / a.sum(dim=-1, keepdim=True)
            
            result = torch.matmul(a, result)
    
    mask = result[0, 0, 1:]
    
    width = int(mask.size(-1) ** 0.5)
    mask = mask.reshape(width, width).cpu().numpy()
    
    # Normalizzazione tra 0 e 1
    mask = mask / np.max(mask)
    return mask

def generate_visualization(img_np, mask):
    """
    Genera e sovrappone la heatmap all'immagine originale.
    :param img_np: immagine originale numpy RGB shape (H, W, 3) con valori in [0, 1]
    :param mask: maschera 2D restituita dal rollout
    """
    mask_resized = cv2.resize(mask, (img_np.shape[1], img_np.shape[0]))
    
    heatmap = cv2.applyColorMap(np.uint8(255 * mask_resized), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1] # converte da BGR a RGB
    
    cam = heatmap + img_np
    cam = cam / np.max(cam)
    
    return cam

if __name__ == '__main__':
    print("Script utility per Attention Rollout caricato. Usa 'import attention_rollout' per utilizzarlo.")
