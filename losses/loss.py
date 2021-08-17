from typing import Dict

import torch
import ignite.metrics as metrics
from ignite.engine.engine import Engine

from captioning.models.utils import generate_length_mask


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    def forward(self, output: Dict):
        # logit: [bs, max_len, c]
        # target: [bs, max_len]
        # lens: [bs]
        logits = output["logits"]
        targets = output["targets"]
        lens = output["lens"]
        c = logits.size(-1)
        loss = self.loss_fn(logits.reshape(-1, c), targets.reshape(-1))
        loss = loss.reshape(*targets.shape)
        mask = generate_length_mask(lens).to(logits.device)
        loss = (loss * mask).sum() / mask.sum()
        return loss


class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, smoothing=0.0, dim=-1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        # self.cls = classes
        self.dim = dim

    def forward(self, output: Dict):
        # logit: [bs, max_len, c]
        # target: [bs, max_len]
        # lens: [bs]
        logits = output["logits"]
        targets = output["targets"]
        lens = output["lens"]
        preds = logits.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(preds)
            true_dist.fill_(self.smoothing / (logits.size(-1) - 1))
            true_dist.scatter_(self.dim, targets.data.unsqueeze(self.dim), self.confidence)
        loss = torch.sum(-true_dist * preds, dim=self.dim)
        mask = generate_length_mask(lens).to(logits.device)
        loss = (loss * mask).sum() / mask.sum()
        return loss


class Loss(metrics.Loss):

    def update(self, output: Dict) -> None:
        # logit: [bs, max_len, c]
        # target: [bs, max_len]
        # lens: [bs]
        lens = output["lens"]
        average_loss = self._loss_fn(output).detach()

        if len(average_loss.shape) != 0:
            raise ValueError("loss_fn did not return the average loss.")

        n = torch.sum(lens)
        self._sum += average_loss.to(self._device) * n
        self._num_examples += n

    @torch.no_grad()
    def iteration_completed(self, engine: Engine) -> None:
        output = self._output_transform(engine.state.output)
        self.update(output)

