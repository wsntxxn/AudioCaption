from typing import Callable, Union, Dict, Mapping

import torch
import ignite.metrics as metrics
from ignite.engine.engine import Engine

from captioning.models.utils import generate_length_mask


# Reimplement ignite metrics to support dict input
class Accuracy(metrics.Accuracy):

    def update(self, output: Dict) -> None:
        # logits: [bs, max_len, c]
        # targets: [bs, max_len]
        # lens: [bs]
        logits = output["logits"]
        targets = output["targets"]
        lens = output["lens"]
        indices = torch.argmax(logits, dim=-1)
        correct = torch.eq(indices, targets)
        mask = generate_length_mask(lens).to(logits.device)
        correct = correct * mask
        self._num_correct += torch.sum(correct).to(self._device)
        self._num_examples += torch.sum(lens)

    @torch.no_grad()
    def iteration_completed(self, engine: Engine) -> None:
        output = self._output_transform(engine.state.output)
        self.update(output)
