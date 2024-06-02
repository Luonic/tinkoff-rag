import torch
from ignite.metrics import Metric
from ignite.exceptions import NotComputableError

# These decorators helps with distributed settings
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced


class Ranking(Metric):
    def __init__(self, k, output_transform=lambda x: x, device="cpu"):
        self.k = k
        self.queries_embeddings = None
        self.passage_embeddings = None
        super(Ranking, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self.queries_embeddings = []
        self.passage_embeddings = []
        super(Ranking, self).reset()

    @reinit__is_reduced
    def update(self, output):
        query_emb, passage_emb = output[0].detach().cpu(), output[1].detach().cpu()
        self.queries_embeddings.append(query_emb)
        self.passage_embeddings.append(passage_emb)

    # @sync_all_reduce("_num_examples", "_num_correct:SUM")
    def compute(self):
        if len(self.queries_embeddings) == 0 or len(self.passage_embeddings) == 0:
            raise NotComputableError('Ranking must have at least one example before it can be computed.')

        self.queries_embeddings = torch.concat(self.queries_embeddings, dim=0)
        self.queries_embeddings = torch.nn.functional.normalize(self.queries_embeddings, p=2, dim=1)
        
        self.passage_embeddings = torch.concat(self.passage_embeddings, dim=0)
        self.passage_embeddings = torch.nn.functional.normalize(self.passage_embeddings, p=2, dim=1)
        
        cosine_scores = self.queries_embeddings @ self.passage_embeddings.transpose(-2, -1)
        top_k_indices = torch.topk(cosine_scores, k=self.k, dim=1, largest=True, sorted=False).indices

        hits = 0
        for i, indices in enumerate(top_k_indices):
            if i in indices:
                hits += 1

        return hits / len(indices)