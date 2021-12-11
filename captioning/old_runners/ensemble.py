import sys
import os
import fire
import pandas as pd

import torch
from ignite.contrib.handlers import ProgressBar
from ignite.utils import convert_tensor
from ignite.engine.engine import Engine

sys.path.append(os.getcwd())
from datasets.SJTUDataSet import SJTUDatasetEval, collate_fn
import models
from utils.build_vocab import Vocabulary

device = "cpu"
if torch.cuda.is_available() and "SLURM_JOB_PARTITION" in os.environ and \
    "gpu" in os.environ["SLURM_JOB_PARTITION"]:
    device = "cuda"
    torch.backends.cudnn.deterministic = True
device = torch.device(device)

class Ensemble(object):

    def _ensemble(self, 
                 path1: str, 
                 path2: str, 
                 kaldi_stream: str,
                 kaldi_scp: str,
                 max_length: int=None):
        dump1 = torch.load(path1, map_location="cpu")
        dump2 = torch.load(path2, map_location="cpu")

        model1 = dump1["model"].to(device)
        model2 = dump2["model"].to(device)

        scaler = dump1["scaler"]
        config = dump1["config"]

        vocabulary = torch.load(config["vocab_file"])
        
        dataset = SJTUDatasetEval(
            kaldi_stream=kaldi_stream,
            kaldi_scp=kaldi_scp,
            transform=scaler.transform)
        # dataset[i]: key, feature
        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            collate_fn=collate_fn((1,)),
            batch_size=32,
            num_workers=0)

        if max_length is None:
            max_length = model1.max_length

        pbar = ProgressBar(persist=False, ascii=True)

        key2pred = {}

        def _sample(engine, batch):
            # batch: [ids, feats, feat_lens]

            ids = batch[0]
            feats = batch[1]
            feat_lens = batch[-1]
            seqs = self._sample_batch(model1, model2, feats, feat_lens, max_length)
            seqs = seqs.cpu().numpy()

            for idx, seq in enumerate(seqs):
                caption = []
                for word_id in seq:
                    word = vocabulary.idx2word[word_id]
                    if word == "<start>":
                        continue
                    elif word == "<end>":
                        break
                    else:
                        caption.append(word)
                caption = " ".join(caption)
                key2pred[ids[idx]] = [caption,]

        sampler = Engine(_sample)
        pbar.attach(sampler)
        sampler.run(dataloader)

        return key2pred

    def _sample_batch(self,
                      model1,
                      model2,
                      features,
                      feature_lens,
                      max_length):
        with torch.no_grad():
            model1.eval()
            model2.eval()
            features = convert_tensor(features.float(),
                                   device=device,
                                   non_blocking=True)

            N = features.shape[0]

            encoded1, states1 = model1.encoder(features, feature_lens)
            encoded2, states2 = model2.encoder(features, feature_lens)

            seqs = torch.zeros(N, max_length, dtype=torch.long).fill_(model1.end_idx)
            # start sampling
            for t in range(max_length):
                # prepare input word/audio embedding
                if t == 0:
                    e1_t = encoded1
                    e2_t = encoded2
                else:
                    e1_t = model1.word_embeddings(w_t)
                    e1_t = model1.dropoutlayer(e1_t)
                    e2_t = model2.word_embeddings(w_t)
                    e2_t = model2.dropoutlayer(e2_t)

                e1_t = e1_t.unsqueeze(1)
                outputs = model1.decoder(e1_t, states1)
                states1 = outputs["states"]
                probs1_t = outputs["probs"].squeeze(1)

                e2_t = e2_t.unsqueeze(1)
                outputs = model2.decoder(e2_t, states2)
                states2 = outputs["states"]
                probs2_t = outputs["probs"].squeeze(1)

                probs_t = (probs1_t + probs2_t) / 2
                logprobs = torch.log_softmax(probs_t, dim=1)
                sampled_logprobs, w_t = torch.max(logprobs.detach(), 1)

                # decide whether to stop
                if t >= 1:
                    if t == 1:
                        unfinished = w_t != model1.end_idx
                    else:
                        unfinished = unfinished * (w_t != model1.end_idx)
                    # w_t[~unfinished] = self.end_idx
                    seqs[:, t] = w_t
                    seqs[:, t][~unfinished] = model1.end_idx
                    if unfinished.sum() == 0:
                        break
                else:
                    seqs[:, t] = w_t

        return seqs


    def predict(self,
                path1: str, 
                path2: str, 
                kaldi_stream: str,
                kaldi_scp: str,
                max_length: int=None,
                output: str="prediction.csv"):

        key2pred = self._ensemble(path1, path2, kaldi_stream, 
                                  kaldi_scp, max_length)
        predictions = []

        for key, prediction in key2pred.items():
            predictions.append({"file_name": key + ".wav",
                                "caption_predicted": prediction[0]})

        pred_df = pd.DataFrame(predictions)
        pred_df.to_csv(output, index=False)

    def coco_evaluate(self,
                      path1: str, 
                      path2: str, 
                      kaldi_stream: str,
                      kaldi_scp: str,
                      caption_file: str,
                      max_length: int=None,
                      output: str="coco_scores.txt"):
        key2pred = self._ensemble(path1, path2, kaldi_stream, 
                                  kaldi_scp, max_length)

        caption_df = pd.read_json(caption_file)
        caption_df["key"] = caption_df["filename"].apply(lambda x: os.path.splitext(x)[0])
        key2refs = caption_df.groupby(["key"])["caption"].apply(list).to_dict()

        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.spice.spice import Spice

        f = open(output, "w")

        scorer = Bleu(n=4)
        score, scores = scorer.compute_score(key2refs, key2pred)
        for n in range(4):
            f.write("Bleu-{}: {:6.3f}\n".format(n + 1, score[n]))

        scorer = Rouge()
        score, scores = scorer.compute_score(key2refs, key2pred)
        f.write("ROUGE: {:6.3f}\n".format(score))

        scorer = Cider()
        score, scores = scorer.compute_score(key2refs, key2pred)
        f.write("CIDEr: {:6.3f}\n".format(score))


        scorer = Meteor()
        score, scores = scorer.compute_score(key2refs, key2pred)
        f.write("Meteor: {:6.3f}\n".format(score))

        scorer = Spice()
        score, scores = scorer.compute_score(key2refs, key2pred)
        f.write("Spice: {:6.3f}\n".format(score))

        f.close()

if __name__ == "__main__":
    fire.Fire(Ensemble)
