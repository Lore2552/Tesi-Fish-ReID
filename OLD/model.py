import torch
import torch.nn as nn
import timm

class SiameseViT(nn.Module):
    """
    Siamese Network basata su un Vision Transformer (ViT).
    Prende in input due immagini, estrae le feature con lo stesso backbone, 
    e le proietta in uno spazio a dimensionalità ridotta (embedding_dim).
    """
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True, embedding_dim=256):
        super(SiameseViT, self).__init__()
        
        # Carica un ViT pre-addestrato dalla libreria timm.
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        # Ottieni la dimensione delle feature in uscita dal backbone (es. 768 per vit_base)
        in_features = self.backbone.num_features
        
        # Una Projection Head per mappare le feature del ViT nello spazio di embedding
        self.projection_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )

    def forward_one(self, x):
        # Estrazione delle feature
        features = self.backbone(x)
        # Proiezione
        embedding = self.projection_head(features)
        
        # Normalizziamo l'embedding con L2 (spesso utile nel contrastive learning)
        embedding = nn.functional.normalize(embedding, p=2, dim=1)
        
        return embedding

    def forward(self, x1, x2=None):
        """
        Se viene fornita solo x1, estrae l'embedding di una singola immagine.
        Se vengono fornite x1 e x2, estrae gli embedding di entrambe (fase di training).
        """
        if x2 is None:
            return self.forward_one(x1)
            
        emb1 = self.forward_one(x1)
        emb2 = self.forward_one(x2)
        
        return emb1, emb2
