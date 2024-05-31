import torch
import torch.nn.functional as F
import torch.nn as nn

class MeanPositiveSimilarity(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, query_logits, passage_logits) -> torch.Tensor:
    
        query_embeddings = F.normalize(query_logits, p=2, dim=1)
        passage_embeddings = F.normalize(passage_logits, p=2, dim=1)
        # Negative keys are implicitly off-diagonal positive keys.
        # Cosine between all combinations
        cosine_distances = query_embeddings @ passage_embeddings.transpose(-2, -1)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query_logits), device=query_logits.device)
        labels = F.one_hot(labels, num_classes=query_logits.size(0))
        # print(labels)
        positives = cosine_distances[labels.bool()]

        return positives.mean()
    

class MeanPositiveAndNegativeSimilarity(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, query_logits, passage_logits) -> torch.Tensor:
    
        query_embeddings = F.normalize(query_logits, p=2, dim=1)
        passage_embeddings = F.normalize(passage_logits, p=2, dim=1)
        # Negative keys are implicitly off-diagonal positive keys.
        # Cosine between all combinations
        cosine_distances = query_embeddings @ passage_embeddings.transpose(-2, -1)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query_logits), device=query_logits.device)
        labels = F.one_hot(labels, num_classes=query_logits.size(0))
        positives = cosine_distances[labels.bool()]
        negatives = cosine_distances[~labels.bool()]

        return (positives.mean() + (1 - negatives.mean())) / 2