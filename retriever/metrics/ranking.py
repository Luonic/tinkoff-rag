import torch
from ignite.metrics import Metric
from ignite.exceptions import NotComputableError

# These decorators helps with distributed settings
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced


class Ranking(Metric):
    def __init__(self, output_transform=lambda x: x, device="cpu"):
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

    @sync_all_reduce("_num_examples", "_num_correct:SUM")
    def compute(self):
        if len(self.queries_embeddings == 0) or len(self.passage_embeddings) == 0:
            raise NotComputableError('Ranking must have at least one example before it can be computed.')

        self.queries_embeddings = torch.concat(self.queries_embeddings, dim=0)
        self.queries_embeddings = torch.nn.functional.normalize(self.queries_embeddings, p=2, dim=1)
        
        self.passage_embeddings = torch.concat(self.passage_embeddings, dim=0)
        self.passage_embeddings = torch.nn.functional.normalize(self.passage_embeddings_embeddings, p=2, dim=1)
        
        cosine_scores = self.queries_embeddings @ self.passage_embeddings.transpose(-2, -1)
        indices = torch.argmax(cosine_scores, dim=1)

        labels = torch.arange(len(indices), device="cpu")
        score = torch.eq(indices, labels).to(torch.float32).sum() / len(indices)
        
        return score