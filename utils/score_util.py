import os
import sys
import numpy as np

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

# def compute_batch_score(decode_res,
                        # refs,
                        # keys,
                        # start_idx,
                        # end_idx,
                        # vocabulary,
                        # scorer):
    # """
    # Args:
        # decode_res: decoding results of model, [N, max_length]
        # refs: references of all samples, dict(<key> -> [ref_1, ref_2, ..., ref_n]
        # keys: keys of this batch, used to match decode results and refs
    # Return:
        # scores of this batch, [N,]
    # """

    # if scorer is None:
        # from pycocoevalcap.cider.cider import Cider
        # scorer = Cider()

    # key2pred = {}
    # key2refs = {}

    # for i in range(len(keys)):

        # key = keys[i]

        # if key in key2pred.keys():
            # continue

        # # prepare candidate sentence
        # candidate = []
        # for w_t in decode_res[i]:
            # if w_t == start_idx:
                # continue
            # elif w_t == end_idx:
                # break
            # candidate.append(vocabulary.idx2word[w_t])
        # key2pred[key] = [" ".join(candidate), ]

        # # prepare reference sentences
        # reference = []
        # for w_t in refs[i]:
            # if w_t == start_idx:
                # continue
            # elif w_t == end_idx:
                # break
            # reference.append(vocabulary.idx2word[w_t])
        # reference = " ".join(reference)
        # key2refs[key] = [reference, ]

    # score, scores = scorer.compute_score(key2refs, key2pred)

    # key2score = {key: scores[i] for i, key in enumerate(key2refs.keys())}
    # results = np.zeros(decode_res.shape[0])
    # for i in range(decode_res.shape[0]):
        # results[i] = key2score[keys[i]]
    # return results
