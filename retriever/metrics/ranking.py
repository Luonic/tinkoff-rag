import torch
from ignite.metrics import Metric
from ignite.exceptions import NotComputableError

# These decorators helps with distributed settings
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced


class Ranking(Metric):
    def __init__(self, output_transform=lambda x: x, device="cpu"):
        self._num_correct = None
        self._num_examples = None
        super(Ranking, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._num_correct = torch.tensor(0, device=self._device)
        self._num_examples = 0
        super(Ranking, self).reset()

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output[0].detach(), output[1].detach()

        indices = torch.argmax(y_pred, dim=1)

        mask = (y != self.ignored_class)
        mask &= (indices != self.ignored_class)
        y = y[mask]
        indices = indices[mask]
        correct = torch.eq(indices, y).view(-1)

        self._num_correct += torch.sum(correct).to(self._device)
        self._num_examples += correct.shape[0]

    @sync_all_reduce("_num_examples", "_num_correct:SUM")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('Ranking must have at least one example before it can be computed.')
        return self._num_correct.item() / self._num_examples