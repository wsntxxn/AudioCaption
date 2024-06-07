import torch
import torch.nn as nn

from captioning.utils.model_util import generate_length_mask, mean_with_lens


class TokenLevelKdLoss(nn.Module):

    def __init__(self, temp=1.0, loss_type="kl"):
        super().__init__()
        self.temp = temp
        if loss_type == "kl":
            self.loss_fn = nn.CrossEntropyLoss(reduction="none")
        elif loss_type == "l2":
            self.loss_fn = nn.MSELoss(reduction="none")
        elif loss_type == "l1":
            self.loss_fn = nn.L1Loss(reduction="none")

    def forward(self, output: dict):

        logit_s = output["logit"] / self.temp
        logit_t = output["tchr_logit"] / self.temp
        tgt = output["tgt"]
        tgt_len = output["tgt_len"]

        c = logit_s.size(-1)
        prob_t = torch.softmax(logit_t, dim=-1)
        loss = self.loss_fn(logit_s.view(-1, c), prob_t.view(-1, c))
        loss = loss.reshape(*tgt.shape)
        mask = generate_length_mask(tgt_len).to(logit_s.device)
        loss *= mask
        loss = loss.sum() / mask.sum()
        return loss


class SupKdLoss(nn.Module):

    def __init__(self, sup_loss, kd_loss, sup_weight=0.5):
        super().__init__()
        self.sup_loss = sup_loss
        self.kd_loss = kd_loss
        self.sup_weight = sup_weight

    def forward(self, output: dict):
        sup_loss = self.sup_loss(output)
        kd_loss = self.kd_loss(output)
        loss = sup_loss * self.sup_weight + kd_loss * (1 - self.sup_weight)
        return loss
