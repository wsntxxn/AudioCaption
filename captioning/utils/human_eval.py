import os
import sys
import copy
import pickle

import numpy as np
import pandas as pd
import fire

sys.path.append(os.getcwd())


def coco_score(refs, scorer):
    if scorer.method() == "Bleu":
        scores = np.array([ 0.0 for n in range(4) ])
    else:
        scores = 0
    num_cap_per_audio = len(refs[list(refs.keys())[0]])

    for i in range(num_cap_per_audio):
        if i > 0:
            for key in refs:
                refs[key].insert(0, res[key][0])
        res = {key: [refs[key].pop(),] for key in refs}
        score, _ = scorer.compute_score(refs, res)    
        
        if scorer.method() == "Bleu":
            scores += np.array(score)
        else:
            scores += score
    
    score = scores / num_cap_per_audio
    return score

def embedding_score(refs, zh):
    from audiocaptioneval.sentbert.sentencebert import SentenceBert
    scorer = SentenceBert(zh=zh)

    num_cap_per_audio = len(refs[list(refs.keys())[0]])
    scores = 0

    for i in range(num_cap_per_audio):
        res = {key: [refs[key][i],] for key in refs.keys() if len(refs[key]) == num_cap_per_audio}
        refs_i = {key: np.concatenate([refs[key][:i], refs[key][i+1:]]) for key in refs.keys() if len(refs[key]) == num_cap_per_audio}
        score, _ = scorer.compute_score(refs_i, res)    
        scores += score
    
    score = scores / num_cap_per_audio
    return score

def diversity_score(refs, zh):
    from utils.diverse_eval import diversity_evaluate

    np.random.seed(1)
    part_df = []
    for key, tokens in refs.items():
        if len(tokens) > 1:
            indice = np.random.choice(len(tokens), 1)[0]
            token = tokens[indice]
        else:
            token = tokens[0]
        if not zh:
            token = token.split(" ")
        part_df.append({"tokens": token})
    part_df = pd.DataFrame(part_df)
    return diversity_evaluate(part_df)
   
def main(eval_caption_file, output, zh=False, embedding_path=None):
    df = pd.read_json(eval_caption_file)
    if zh:
        refs = df.groupby("key")["tokens"].apply(list).to_dict()
    else:
        refs = df.groupby("key")["caption"].apply(list).to_dict()

    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.rouge.rouge import Rouge

    scorer = Bleu(zh=zh)
    bleu_scores = coco_score(copy.deepcopy(refs), scorer)
    print(bleu_scores)
    scorer = Cider(zh=zh)
    cider_score = coco_score(copy.deepcopy(refs), scorer)
    print(cider_score)
    scorer = Rouge(zh=zh)
    rouge_score = coco_score(copy.deepcopy(refs), scorer)
    print(rouge_score)

    if not zh:
        from pycocoevalcap.meteor.meteor import Meteor
        scorer = Meteor()
        meteor_score = coco_score(copy.deepcopy(refs), scorer)

        from pycocoevalcap.spice.spice import Spice
        scorer = Spice()
        spice_score = coco_score(copy.deepcopy(refs), scorer)
    
    diverse_score = diversity_score(refs, zh)

    with open(embedding_path, "rb") as f:
        ref_embeddings = pickle.load(f)

    bert_score = embedding_score(ref_embeddings, zh)

    with open(output, "w") as f:
        for n in range(4):
            f.write("BLEU-{}: {:6.3f}\n".format(n+1, bleu_scores[n]))
        f.write("CIDEr: {:6.3f}\n".format(cider_score))
        f.write("ROUGE: {:6.3f}\n".format(rouge_score))
        if not zh:
            f.write("Meteor: {:6.3f}\n".format(meteor_score))
            f.write("SPICE: {:6.3f}\n".format(spice_score))
        f.write("SentenceBert: {:6.3f}\n".format(bert_score))
        f.write("Diversity: {:6.3f}\n".format(diverse_score))


if __name__ == "__main__":
    fire.Fire(main)
