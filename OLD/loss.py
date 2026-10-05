import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss classica.
    Calcola la distanza Euclidea tra i due embedding.
    Se label=1 (stesso pesce), minimizza la distanza.
    Se label=0 (pesci diversi), massimizza la distanza spingendola oltre un certo margine.
    """
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, emb1, emb2, label):
        # Assicuriamoci che la label sia della forma corretta (B, 1)
        label = label.view(-1, 1)
        
        # Calcolo della distanza Euclidea
        euclidean_distance = F.pairwise_distance(emb1, emb2, keepdim=True)
        
        # y * d^2 + (1 - y) * max(0, margin - d)^2
        loss = torch.mean(
            label * torch.pow(euclidean_distance, 2) +
            (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        
        return loss, euclidean_distance
