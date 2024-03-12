from typing import Dict, List

import numpy as np
import torch
import ignite.metrics as metrics
from ignite.engine.engine import Engine
import wandb

from captioning.models.utils import generate_length_mask, mean_with_lens


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self, reduction="mean", logit_name="logit", target_name="tgt"):
        super().__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        self.reduction = reduction
        self.logit_name = logit_name
        self.target_name = target_name

    def forward(self, output: Dict):
        # logit: [bs, max_len, c]
        # tgt: [bs, max_len]
        # tgt_len: [bs]
        logit = output[self.logit_name]
        tgt = output[self.target_name]
        tgt_len = output[f"{self.target_name}_len"]
        c = logit.size(-1)
        loss = self.loss_fn(logit.reshape(-1, c), tgt.reshape(-1))
        loss = loss.reshape(*tgt.shape)
        mask = generate_length_mask(tgt_len).to(logit.device)
        loss *= mask
        if self.reduction == "none":
            return loss
        elif self.reduction == "mean":
            loss = loss.sum() / mask.sum()
            return loss
        elif self.reduction == "sum":
            loss = loss.sum()
            return loss


class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, smoothing=0.0, dim=-1, reduction="mean", logit_name="logit",
                 target_name="tgt"):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.dim = dim
        self.reduction = reduction
        self.logit_name = logit_name
        self.target_name = target_name

    def forward(self, output: Dict):
        # logit: [bs, max_len, c]
        # tgt: [bs, max_len]
        # tgt_len: [bs]
        logit = output[self.logit_name]
        tgt = output[self.target_name]
        tgt_len = output[f"{self.target_name}_len"]
        preds = logit.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(preds)
            true_dist.fill_(self.smoothing / (logit.size(-1) - 1))
            true_dist.scatter_(self.dim, tgt.data.unsqueeze(self.dim), self.confidence)
        loss = torch.sum(-true_dist * preds, dim=self.dim)
        mask = generate_length_mask(tgt_len).to(logit.device)
        loss *= mask
        if self.reduction == "none":
            return loss
        elif self.reduction == "mean":
            loss = loss.sum() / mask.sum()
            return loss
        elif self.reduction == "sum":
            loss = loss.sum()
            return loss


class MultipleLossSum(torch.nn.Module):

    def __init__(self,
                 names: List[str],
                 weights: List[float],
                 **loss_fns):
        super().__init__()
        self.names = names
        self.weights = weights
        for name, loss_fn in loss_fns.items():
            self.add_module(name, loss_fn)

    def forward(self, output: Dict):
        verbose = output.get("verbose", True)
        tot_loss = 0
        for name, weight in zip(self.names, self.weights):
            if name in output:
                loss = output[name]
            else:
                loss = getattr(self, name)(output)
            tot_loss += weight * loss
            if self.training and verbose and wandb.run is not None:
                wandb.log({f"train/{name}": loss.item()},
                          step=output["step"])
        return tot_loss


class AugmentLossWrapper(torch.nn.Module):
    def __init__(self, loss_fn, eps=1e-12):
        super().__init__()
        self.loss_fn = loss_fn
        self.loss_fn.reduction = "none"
        self.eps = eps

    def forward(self, output: Dict):
        loss = self.loss_fn(output)
        cap_ids = output["cap_ids"]
        use_aug_prob = output["use_aug_prob"]
        aug_mask = np.array(["aug" not in cap_id for cap_id in cap_ids])
        use_aug = np.random.choice(
            [0, 1],
            size=(~aug_mask).sum(),
            p=[1 - use_aug_prob, use_aug_prob]
        )
        aug_mask[~aug_mask] = use_aug
        aug_mask = torch.as_tensor(aug_mask).to(loss.device)
        loss *= aug_mask.reshape(-1, 1)
        mask = generate_length_mask(output["tgt_len"]).to(loss.device)
        mask *= aug_mask.reshape(-1, 1)
        return loss.sum() / (mask.sum() + self.eps)


def reparameterize_argmax(logit, dim=-1):
    # y = torch.softmax(logit, dim)
    y = logit
    shape = y.size()
    _, ind = y.max(dim=dim)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logit, temperature):
    y = logit + sample_gumbel(logit.size()).to(logit.device)
    return torch.softmax(y / temperature, dim=-1)


def gumbel_softmax(logit, temperature=1.0):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logit, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y


class ConditionLossWrapper(torch.nn.Module):
    def __init__(self, loss_fn, dscrm, alpha=1, sample_method="argmax"):
        super().__init__()
        self.loss_fn = loss_fn
        self.dscrm = dscrm
        self.alpha = alpha
        self.sample_method = sample_method
        self.condition_fn = torch.nn.BCELoss()

    def forward(self, output: Dict):
        word_loss = self.loss_fn(output)
        logit = output["logit"]
        conditions = output["conditions"].to(logit.device)
        # preds = reparameterize_argmax(logit).to(logit.device)
        # preds = gumbel_softmax(logit).to(logit.device)
        if self.sample_method == "argmax":
            preds = reparameterize_argmax(logit)
        elif self.sample_method == "gumbel":
            preds = gumbel_softmax(logit)
        elif self.sample_method == "weighted":
            preds = torch.softmax(logit, -1)
        else:
            raise Exception(f"sample method {self.sample_method} not supported")
        preds = preds.to(logit.device)
        tgt_len = output["tgt_len"] - 1 # remove <eos>
        probs = self.dscrm({"caps": preds, "tgt_len": tgt_len})
        condition_loss = self.condition_fn(probs, conditions)
        loss = word_loss + self.alpha * condition_loss
        return loss, word_loss, condition_loss
        

class SpecificityLossWrapper(torch.nn.Module):
    def __init__(self, loss_fn, word_specificity, sentence_reduce="sum", alpha=1):
        super().__init__()
        self.loss_fn = loss_fn
        self.word_specificity = word_specificity # [vocab_size]
        self.alpha = alpha
        self.sentence_reduce = sentence_reduce
        self.condtion_fn = torch.nn.MSELoss()

    def forward(self, output: Dict):
        word_loss = self.loss_fn(output)
        logit = output["logit"]
        conditions = output["conditions"].to(logit.device)
        probs = torch.softmax(logit, dim=-1)
        cond_pred = torch.matmul(probs, self.word_specificity) # [N, T]
        tgt_len = output["tgt_len"] - 1 # remove <eos>
        if self.sentence_reduce == "sum":
            mask = generate_length_mask(tgt_len, max_length=cond_pred.size(1)).to(logit.device)
            cond_pred *= mask
            cond_pred = cond_pred.sum(1)
        else:
            cond_pred = mean_with_lens(cond_pred, tgt_len)
        condition_loss = self.condtion_fn(cond_pred, conditions)
        loss = word_loss + self.alpha * condition_loss
        return loss, word_loss, condition_loss


class Loss(metrics.Loss):

    def update(self, output: Dict) -> None:
        # logit: [bs, max_len, c]
        # target: [bs, max_len]
        # tgt_len: [bs]
        tgt_len = output["tgt_len"]
        average_loss = self._loss_fn(output).detach()

        if len(average_loss.shape) != 0:
            raise ValueError("loss_fn did not return the average loss.")

        n = torch.sum(tgt_len)
        self._sum += average_loss.to(self._device) * n
        self._num_examples += n

    @torch.no_grad()
    def iteration_completed(self, engine: Engine) -> None:
        output = self._output_transform(engine.state.output)
        self.update(output)

