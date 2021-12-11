import numpy as np
import torch
import torch.nn as nn

def generate_length_mask(lens, max_length=None):
    lens = torch.as_tensor(lens)
    N = lens.size(0)
    if max_length is None:
        max_length = max(lens)
    idxs = torch.arange(max_length).repeat(N).view(N, max_length)
    mask = (idxs < lens.view(-1, 1))
    return mask


def mean_with_lens(features, lens):
    """
    features: [N, T, ...] (assume the second dimension represents length)
    lens: [N,]
    """
    lens = torch.as_tensor(lens)
    mask = generate_length_mask(lens).to(features.device) # [N, T]

    feature_mean = features * mask.unsqueeze(-1)
    feature_mean = feature_mean.sum(1) / lens.unsqueeze(1).to(features.device)
    return feature_mean


def max_with_lens(features, lens):
    """
    features: [N, T, ...] (assume the second dimension represents length)
    lens: [N,]
    """
    lens = torch.as_tensor(lens)
    mask = generate_length_mask(lens).to(features.device) # [N, T]

    feature_max = features.clone()
    feature_max[~mask] = float("-inf")
    feature_max, _ = feature_max.max(1)
    return feature_max

def repeat_tensor(x, n):
    return x.unsqueeze(0).repeat(n, *([1] * len(x.shape)))


def init(m, method="kaiming"):
    if isinstance(m, (nn.Conv2d, nn.Conv1d)):
        if method == "kaiming":
            nn.init.kaiming_uniform_(m.weight)
        else:
            nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Linear):
        if method == "kaiming":
            nn.init.kaiming_uniform_(m.weight)
        else:
            nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def compute_batch_score(decode_res,
                        key2refs,
                        keys,
                        start_idx,
                        end_idx,
                        vocabulary,
                        scorer):
    """
    Args:
        decode_res: decoding results of model, [N, max_length]
        key2refs: references of all samples, dict(<key> -> [ref_1, ref_2, ..., ref_n]
        keys: keys of this batch, used to match decode results and refs
    Return:
        scores of this batch, [N,]
    """

    if scorer is None:
        from pycocoevalcap.cider.cider import Cider
        scorer = Cider()

    hypothesis = {}
    references = {}

    for i in range(len(keys)):

        if keys[i] in hypothesis.keys():
            continue

        # prepare candidate sentence
        candidate = []
        for w_t in decode_res[i]:
            if w_t == start_idx:
                continue
            elif w_t == end_idx:
                break
            candidate.append(vocabulary.idx2word[w_t])

        hypothesis[keys[i]] = [" ".join(candidate), ]

        # prepare reference sentences
        references[keys[i]] = key2refs[keys[i]]

    score, scores = scorer.compute_score(references, hypothesis)
    key2score = {key: scores[i] for i, key in enumerate(references.keys())}
    results = np.zeros(decode_res.shape[0])
    for i in range(decode_res.shape[0]):
        results[i] = key2score[keys[i]]
    return results 
