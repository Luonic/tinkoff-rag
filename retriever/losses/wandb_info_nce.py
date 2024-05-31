import torch
import torch.nn.functional as F
import torch.nn as nn

class InfoNCE(nn.Module):
    def __init__(self, temperature=0.1, reduction='mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        
    def forward(self, query_logits, passage_logits) -> torch.Tensor:
    
        query_embeddings = F.normalize(query_logits, p=2, dim=1)
        passage_embeddings = F.normalize(passage_logits, p=2, dim=1)
        # Negative keys are implicitly off-diagonal positive keys.
        # Cosine between all combinations
        logits = query_embeddings @ passage_embeddings.transpose(-2, -1)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query_logits), device=query_logits.device)
        # print(labels)

        return F.cross_entropy(logits / self.temperature, labels, reduction=self.reduction)
