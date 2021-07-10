# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn

from utils import score_util

class ScstWrapper(nn.Module):

    def __init__(self, model):
        super(ScstWrapper, self).__init__()
        self.model = model

    def forward(self, *input, **kwargs):
        """Decode audio feature vectors and generates captions.
        """
        if len(input) != 5 and len(input) != 2:
            raise Exception(
                "number of input should be either 5 (feats, feat_lens, keys, key2refs, vocabulary) \
                or 2 (feats, feat_lens)!")

        if len(input) == 2:
            output = self.model(*input, **kwargs)
        else:
            output = self.scst(*input, **kwargs)
        return output

    def scst(self, feats, feat_lens, keys, key2refs, vocabulary, **kwargs):
        output = {}

        sample_kwargs = {
            "temperature": kwargs.get("temperature", 1.0),
            "max_length": kwargs["max_length"]
        }

        # prepare baseline
        self.model.eval()
        with torch.no_grad():
            sampled_greedy = self.model(feats, feat_lens, method="greedy", **sample_kwargs)
        output["greedy_seqs"] = sampled_greedy["seqs"]

        self.model.train()
        sampled = self.model(feats, feat_lens, method="sample", **sample_kwargs)
        output["sampled_seqs"] = sampled["seqs"]

        reward_score = self.get_self_critical_reward(sampled_greedy["seqs"],
                                                     sampled["seqs"],
                                                     keys,
                                                     key2refs,
                                                     vocabulary,
                                                     kwargs["scorer"])
        # reward: [N, ]
        output["reward"] = torch.as_tensor(reward_score["reward"])
        output["score"] = torch.as_tensor(reward_score["score"])

        reward = np.repeat(reward_score["reward"][:, np.newaxis], sampled["seqs"].size(-1), 1)
        reward = torch.as_tensor(reward).float()
        mask = (sampled["seqs"] != self.model.end_idx).float()
        mask = torch.cat([torch.ones(mask.size(0), 1), mask[:, :-1]], 1)
        mask = torch.as_tensor(mask).float()
        loss = - sampled["sampled_logprobs"] * reward * mask
        loss = loss.to(feats.device)
        # loss: [N, max_length]
        loss = torch.sum(loss, dim=1).mean()
        # loss = torch.sum(loss) / torch.sum(mask)
        output["loss"] = loss

        return output

    def get_self_critical_reward(self, greedy_seqs, sampled_seqs,
                                 keys, key2refs, vocabulary, scorer):
        # greedy_seqs, sampled_seqs: [N, max_length]
        greedy_seqs = greedy_seqs.cpu().numpy()
        sampled_seqs = sampled_seqs.cpu().numpy()

        sampled_score = score_util.compute_batch_score(sampled_seqs,
                                                       key2refs,
                                                       keys,
                                                       self.model.start_idx,
                                                       self.model.end_idx,
                                                       vocabulary,
                                                       scorer)
        greedy_score = score_util.compute_batch_score(greedy_seqs, 
                                                      key2refs,
                                                      keys,
                                                      self.model.start_idx,
                                                      self.model.end_idx,
                                                      vocabulary,
                                                      scorer)
        reward = sampled_score - greedy_score
        return {"reward": reward, "score": sampled_score}
