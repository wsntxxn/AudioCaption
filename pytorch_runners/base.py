import json
import pickle
import random
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
from tqdm import tqdm
import fire
import torch
from ignite.engine.engine import Engine
from ignite.contrib.handlers import ProgressBar

import captioning.utils.train_util as train_util
import captioning.datasets.caption_dataset as ac_dataset

class BaseRunner(object):
    """Main class to run experiments"""
    def __init__(self, seed=1):
        super(BaseRunner, self).__init__()
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
            torch.cuda.manual_seed_all(seed)
            # torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        self.device = torch.device(device)

    def _get_dataloaders(self, config):
        augments = train_util.parse_augments(config["augments"])
        if config["distributed"]:
            config["dataloader_args"]["batch_size"] //= self.world_size

        data_config = config["data"]
        vocabulary = config["vocabulary"]
        if "train" not in data_config:
            raw_feat_df = pd.read_csv(data_config["raw_feat_csv"], sep="\t")
            raw_audio_to_h5 = dict(zip(raw_feat_df["audio_id"], raw_feat_df["hdf5_path"]))
            fc_feat_df = pd.read_csv(data_config["fc_feat_csv"], sep="\t")
            fc_audio_to_h5 = dict(zip(fc_feat_df["audio_id"], fc_feat_df["hdf5_path"]))
            attn_feat_df = pd.read_csv(data_config["attn_feat_csv"], sep="\t")
            attn_audio_to_h5 = dict(zip(attn_feat_df["audio_id"], attn_feat_df["hdf5_path"]))
            caption_info = json.load(open(data_config["caption_file"], "r"))["audios"]
            val_size = int(len(caption_info) * (1 - data_config["train_percent"] / 100.))
            val_audio_idxs = np.random.choice(len(caption_info), val_size, replace=False)
            train_audio_idxs = [idx for idx in range(len(caption_info)) if idx not in val_audio_idxs]
            train_dataset = ac_dataset.CaptionDataset(
                raw_audio_to_h5,
                fc_audio_to_h5,
                attn_audio_to_h5,
                caption_info,
                vocabulary,
                transform=augments
            )
            # TODO DistributedCaptionSampler
            # train_sampler = torch.utils.data.DistributedSampler(train_dataset) if config["distributed"] else None
            train_sampler = ac_dataset.CaptionSampler(train_dataset, train_audio_idxs, True, **config["sampler_args"])
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                collate_fn=ac_dataset.collate_fn([0, 2, 3], 3),
                sampler=train_sampler,
                **config["dataloader_args"]
            )
            val_audio_ids = [caption_info[audio_idx]["audio_id"] for audio_idx in val_audio_idxs]
            val_dataset = ac_dataset.CaptionEvalDataset(
                {audio_id: raw_audio_to_h5[audio_id] for audio_id in val_audio_ids},
                {audio_id: fc_audio_to_h5[audio_id] for audio_id in val_audio_ids},
                {audio_id: attn_audio_to_h5[audio_id] for audio_id in val_audio_ids},
            )
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                collate_fn=ac_dataset.collate_fn([1, 3]),
                **config["dataloader_args"]
            )
            train_key2refs = {}
            for audio_idx in train_audio_idxs:
                audio_id = caption_info[audio_idx]["audio_id"]
                train_key2refs[audio_id] = []
                for caption in caption_info[audio_idx]["captions"]:
                    train_key2refs[audio_id].append(caption["token" if data_config["zh"] else "caption"])
            val_key2refs = {}
            for audio_idx in val_audio_idxs:
                audio_id = caption_info[audio_idx]["audio_id"]
                val_key2refs[audio_id] = []
                for caption in caption_info[audio_idx]["captions"]:
                    val_key2refs[audio_id].append(caption["token" if data_config["zh"] else "caption"])
        else:
            data = {"train": {}, "val": {}}
            for split in ["train", "val"]:
                conf_split = data_config[split]
                output = data[split]
                raw_feat_df = pd.read_csv(conf_split["raw_feat_csv"], sep="\t")
                output["raw_audio_to_h5"] = dict(zip(raw_feat_df["audio_id"], raw_feat_df["hdf5_path"]))
                fc_feat_df = pd.read_csv(conf_split["fc_feat_csv"], sep="\t")
                output["fc_audio_to_h5"] = dict(zip(fc_feat_df["audio_id"], fc_feat_df["hdf5_path"]))
                attn_feat_df = pd.read_csv(conf_split["attn_feat_csv"], sep="\t")
                output["attn_audio_to_h5"] = dict(zip(attn_feat_df["audio_id"], attn_feat_df["hdf5_path"]))
                output["caption_info"] = json.load(open(conf_split["caption_file"], "r"))["audios"]

            train_dataset = ac_dataset.CaptionDataset(
                data["train"]["raw_audio_to_h5"],
                data["train"]["fc_audio_to_h5"],
                data["train"]["attn_audio_to_h5"],
                data["train"]["caption_info"],
                vocabulary,
                transform=augments
            )
            # TODO DistributedCaptionSampler
            # train_sampler = torch.utils.data.DistributedSampler(train_dataset) if config["distributed"] else None
            train_sampler = ac_dataset.CaptionSampler(train_dataset, shuffle=True, **config["sampler_args"])
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                collate_fn=ac_dataset.collate_fn([0, 2, 3], 3),
                sampler=train_sampler,
                **config["dataloader_args"]
            )
            val_dataset = ac_dataset.CaptionEvalDataset(
                data["val"]["raw_audio_to_h5"],
                data["val"]["fc_audio_to_h5"],
                data["val"]["attn_audio_to_h5"],
            )
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                collate_fn=ac_dataset.collate_fn([1, 3]),
                **config["dataloader_args"]
            )
            train_key2refs = {}
            caption_info = data["train"]["caption_info"]
            for audio_idx in range(len(caption_info)):
                audio_id = caption_info[audio_idx]["audio_id"]
                train_key2refs[audio_id] = []
                for caption in caption_info[audio_idx]["captions"]:
                    train_key2refs[audio_id].append(
                        caption["token" if config["zh"] else "caption"])
            val_key2refs = {}
            caption_info = data["val"]["caption_info"]
            for audio_idx in range(len(caption_info)):
                audio_id = caption_info[audio_idx]["audio_id"]
                val_key2refs[audio_id] = []
                for caption in caption_info[audio_idx]["captions"]:
                    val_key2refs[audio_id].append(
                        caption["token" if config["zh"] else "caption"])

        return {
            "train_dataloader": train_dataloader,
            "train_key2refs": train_key2refs,
            "val_dataloader": val_dataloader,
            "val_key2refs": val_key2refs
        }

    @staticmethod
    def _get_model(config):
        raise NotImplementedError

    def _forward(self, model, batch, mode, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _convert_idx2sentence(word_ids, vocabulary, zh=False):
        candidate = []
        for word_id in word_ids:
            word = vocabulary[word_id]
            if word == "<end>":
                break
            elif word == "<start>":
                continue
            candidate.append(word)
        if not zh:
            candidate = " ".join(candidate)
        return candidate

    def train(self, config, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _eval_prediction(key2refs, key2pred, scorers, pretokenized=False):
        if not pretokenized:
            refs4eval = {}
            for key, refs in key2refs.items():
                refs4eval[key] = []
                for idx, ref in enumerate(refs):
                    refs4eval[key].append({
                        "audio_id": key,
                        "id": idx,
                        "caption": ref
                    })

            preds4eval = {}
            for key, preds in key2pred.items():
                preds4eval[key] = []
                for idx, pred in enumerate(preds):
                    preds4eval[key].append({
                        "audio_id": key,
                        "id": idx,
                        "caption": pred
                    })

            from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

            tokenizer = PTBTokenizer()
            key2refs = tokenizer.tokenize(refs4eval)
            key2pred = tokenizer.tokenize(preds4eval)

        output = {}
        for scorer in scorers:
            score, scores = scorer.compute_score(key2refs, key2pred)
            output[scorer.method()] = score
        return output

    def predict(self,
                experiment_path: str,
                save_type: str = "best",
                raw_feat_csv: str = None,
                fc_feat_csv: str = None,
                attn_feat_csv: str = None,
                caption_output: str = "output.json",
                return_pred: bool=False,
                **kwargs):
        """kwargs: {'max_length': int, 'method': str, 'beam_size': int}"""
        if "method" in kwargs and kwargs["method"] == "beam":
            del kwargs["method"]
            kwargs["sample_method"] = "beam"
        experiment_path = Path(experiment_path)
        checkpoint = f"{save_type}.pth"
        dump = torch.load(experiment_path / checkpoint, map_location="cpu")
        # Previous training config
        config = train_util.parse_config_or_kwargs(experiment_path / "config.yaml")

        vocabulary = dump["vocabulary"]
        config["vocabulary"] = vocabulary
        model = self._get_model(config)
        model.load_state_dict(dump["model"])

        zh = config["zh"]
        model = model.to(self.device)

        raw_feat_df = pd.read_csv(raw_feat_csv, sep="\t")
        raw_audio_to_h5 = dict(zip(raw_feat_df["audio_id"], raw_feat_df["hdf5_path"]))
        fc_feat_df = pd.read_csv(fc_feat_csv, sep="\t")
        fc_audio_to_h5 = dict(zip(fc_feat_df["audio_id"], fc_feat_df["hdf5_path"]))
        attn_feat_df = pd.read_csv(attn_feat_csv, sep="\t")
        attn_audio_to_h5 = dict(zip(attn_feat_df["audio_id"], attn_feat_df["hdf5_path"]))
        dataset = ac_dataset.CaptionEvalDataset(
            raw_audio_to_h5,
            fc_audio_to_h5,
            attn_audio_to_h5
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            collate_fn=ac_dataset.collate_fn([1, 3]),
            batch_size=kwargs.get("batch_size", 1)
        )

        model.eval()
        key2pred = {}

        with torch.no_grad(), tqdm(total=len(dataloader), ascii=True, ncols=100) as pbar:
            for batch in dataloader:
                output = self._forward(model, batch, mode = "eval", **kwargs)
                keys = batch[0]
                seqs = output["seqs"].cpu().numpy()
                for idx, seq in enumerate(seqs):
                    caption = self._convert_idx2sentence(seq, vocabulary, zh)
                    key2pred[keys[idx]] = [caption,]
                pbar.update()

        pred_data = []
        for key, pred in key2pred.items():
            pred_data.append({
                "filename": key,
                "caption": "".join(pred[0]) if zh else pred[0],
                "tokens": " ".join(pred[0]) if zh else pred[0]
            })
        json.dump({"predictions": pred_data}, open(experiment_path / caption_output, "w"), indent=4)

        if return_pred:
            return key2pred

    def evaluate(self,
                 experiment_path: str,
                 task: str,
                 save_type: str = "best",
                 raw_feat_csv: str = None,
                 fc_feat_csv: str = None,
                 attn_feat_csv: str = None,
                 caption_file: str = None,
                 caption_output: str = "eval_output.json",
                 score_output: str = "scores.txt",
                 **kwargs):
        """kwargs: {'max_length': int, 'method': str, 'beam_size': int}"""
        default_eval_data = {
            "clotho": {
                "raw_feat_csv": "data/clotho_v2/eval/lms.csv",
                "fc_feat_csv": "data/clotho_v2/eval/panns_cnn14_fc.csv",
                "attn_feat_csv": "data/clotho_v2/eval/panns_cnn14_attn.csv",
                "caption_file": "data/clotho_v2/eval/text.json",
            },
            "audiocaps": {
                "raw_feat_csv": "data/audiocaps/test/lms.csv",
                "fc_feat_csv": "data/audiocaps/test/panns_cnn14_fc.csv",
                "attn_feat_csv": "data/audiocaps/test/panns_cnn14_attn.csv",
                "caption_file": "data/audiocaps/test/text.json"
            },
        }
        if raw_feat_csv is None:
            raw_feat_csv = default_eval_data[task]["raw_feat_csv"]
        if fc_feat_csv is None:
            fc_feat_csv = default_eval_data[task]["fc_feat_csv"]
        if attn_feat_csv is None:
            attn_feat_csv = default_eval_data[task]["attn_feat_csv"]
        experiment_path = Path(experiment_path)
        # Previous training config
        config = train_util.parse_config_or_kwargs(experiment_path / "config.yaml")
        zh = config["zh"]
        key2pred = self.predict(experiment_path,
                                save_type,
                                raw_feat_csv,
                                fc_feat_csv,
                                attn_feat_csv,
                                caption_output,
                                return_pred=True,
                                **kwargs)
        if caption_file is None:
            caption_file = default_eval_data[task]["caption_file"]
        captions = json.load(open(caption_file, "r"))["audios"]
        key2refs = {}
        for audio_idx in range(len(captions)):
            audio_id = captions[audio_idx]["audio_id"]
            key2refs[audio_id] = []
            for caption in captions[audio_idx]["captions"]:
                key2refs[audio_id].append(caption["token" if zh else "caption"])

        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.spice.spice import Spice

        scorers = [Bleu(n=4, zh=zh), Rouge(zh=zh), Cider(zh=zh)]
        if not zh:
            scorers.append(Meteor())
            scorers.append(Spice())
        scores_output = self._eval_prediction(key2refs, key2pred, scorers)

        with open(str(experiment_path / score_output), "w") as f:
            spider = 0
            for name, score in scores_output.items():
                if name == "Bleu":
                    for n in range(4):
                        f.write("Bleu-{}: {:6.3f}\n".format(n + 1, score[n]))
                else:
                    f.write("{}: {:6.3f}\n".format(name, score))
                    if name in ["CIDEr", "SPICE"]:
                        spider += score
            if not zh:
                f.write("SPIDEr: {:6.3f}\n".format(spider / 2))

        # from audiocaptioneval.sentbert.sentencebert import SentenceBert
        # scorer = SentenceBert(zh=zh)
        # if caption_embedding_path is not None:
            # key2ref_embeds = np.load(caption_embedding_path, allow_pickle=True)
            # score, scores = scorer.compute_score(key2ref_embeds, key2pred)
        # else:
            # score, scores = scorer.compute_score(key2refs, key2pred)
        # f.write("SentenceBert: {:6.3f}\n".format(score))

        # from utils.diverse_eval import diversity_evaluate
        # score = diversity_evaluate(pred_df)
        # f.write("Diversity: {:6.3f}\n".format(score))

    def train_evaluate(self, config, task, **kwargs):
        experiment_path = self.train(config, **kwargs)
        for save_type in ["best", "last", "swa"]:
            self.evaluate(experiment_path, task, save_type=save_type, score_output=f"{save_type}.txt", method="beam")
        return experiment_path

    def dcase_predict(self,
                      experiment_path: str,
                      feature_file: str,
                      feature_scp: str,
                      output: str="test_prediction.csv",
                      **kwargs):
        """kwargs: {'max_length': int, 'method': str, 'beam_size': int}"""

        dump = torch.load(str(Path(experiment_path) / "saved.pth"), map_location="cpu")
        # Some scaler (sklearn standardscaler)
        scaler = dump["scaler"]
        # Also load previous training config
        config = dump["config"]

        vocabulary = torch.load(config["vocab_file"])
        model = self._get_model(config, vocabulary)
        model.load_state_dict(dump["model"])

        zh = config["zh"]
        model = model.to(self.device)
        
        dataset = SJTUDatasetEval(
            feature=feature_file,
            eval_scp=feature_scp,
            transform=None if scaler is None else [scaler.transform]
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            collate_fn=collate_fn((1,)),
            batch_size=32,
            num_workers=0
        )

        pbar = ProgressBar(persist=False, ascii=True, ncols=100)
        predictions = []

        def _sample(engine, batch):
            # batch: [keys, feats, feat_lens]
            with torch.no_grad():
                model.eval()
                keys = batch[0]
                output = self._forward(model, batch, mode="eval", **kwargs)
                seqs = output["seqs"].cpu().numpy()
                for idx, seq in enumerate(seqs):
                    caption = self._convert_idx2sentence(seq, vocabulary, zh)
                    predictions.append({"file_name": keys[idx], "caption_predicted": caption})

        sample_engine = Engine(_sample)
        pbar.attach(sample_engine)
        sample_engine.run(dataloader)

        pred_df = pd.DataFrame(predictions)
        pred_df.to_csv(str(Path(experiment_path) / output), index=False)

    def ensemble(self,
                 exp_path_file: str,
                 feature_file: str,
                 feature_scp: str,
                 prediction: str,
                 dcase_format: bool = False,
                 label: str = None,
                 score_output: str = "scores.txt",
                 **kwargs):
        from models.seq_train_model import ScstWrapper

        config = None
        vocabulary = None
        scaler = None
        zh = None
        models = []
        exp_paths = []
        with open(exp_path_file, "r") as reader:
            for line in reader.readlines():
                if not line.startswith("#"):
                    exp_paths.append(line.strip())

        for path in exp_paths:
            dump = torch.load(str(Path(path) / "saved.pth"), map_location="cpu")
            if config is None:
                config = dump["config"]
                vocabulary = torch.load(config["vocab_file"])
                scaler = dump["scaler"]
                zh = config["zh"]
            model = self._get_model(config, vocabulary)
            model.load_state_dict(dump["model"])
            if isinstance(model, ScstWrapper):
                model = model.model
            model = model.to(self.device)
            models.append(model)

        dataset = SJTUDatasetEval(
            feature=feature_file,
            eval_scp=feature_scp,
            transform=None if scaler is None else [scaler.transform]
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            collate_fn=collate_fn((1,)),
            batch_size=kwargs.get("batch_size", 32),
            num_workers=0
        )

        key2pred = {}
        pred_data = []

        def _inference(engine, batch):
            keys = batch[0]
            output = self._ensemble_batch(models, batch, **kwargs)
            seqs = output["seqs"].cpu().numpy()
            for idx, seq in enumerate(seqs):
                caption = self._convert_idx2sentence(seq, vocabulary, zh)
                key2pred[keys[idx]] = [caption,]

        pbar = ProgressBar(persist=False, ascii=True, ncols=100)
        sampler = Engine(_inference)
        pbar.attach(sampler)
        sampler.run(dataloader)
        
        for key, pred in key2pred.items():
            if dcase_format:
                pred_data.append({
                    "file_name": key,
                    "caption_predicted": pred[0]
                })
            else:
                pred_data.append({
                    "filename": key,
                    "caption": "".join(pred[0]) if zh else pred[0],
                    "tokens": pred[0] if zh else pred[0].split()
                })
        pred_df = pd.DataFrame(pred_data)
        if dcase_format:
            pred_df.to_csv(prediction, index=False)
        else:
            pred_df.to_json(prediction)

        if label is None:
            return

        label_df = pd.read_json(label, dtype={"audio_key": str})
        if zh:
            key2refs = label_df.groupby("audio_key")["tokens"].apply(list).to_dict()
        else:
            key2refs = label_df.groupby("audio_key")["caption"].apply(list).to_dict()

        refs4eval = {}
        for key, refs in key2refs.items():
            refs4eval[key] = []
            for idx, ref in enumerate(refs):
                refs4eval[key].append({
                    "audio_id": key,
                    "id": idx,
                    "caption": ref
                })

        preds4eval = {}
        for key, preds in key2pred.items():
            preds4eval[key] = []
            for idx, pred in enumerate(preds):
                preds4eval[key].append({
                    "audio_id": key,
                    "id": idx,
                    "caption": pred
                })

        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.spice.spice import Spice
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

        tokenizer = PTBTokenizer()
        key2refs = tokenizer.tokenize(refs4eval)
        key2pred = tokenizer.tokenize(preds4eval)

        f = open(score_output, "w")

        scorer = Bleu(n=4, zh=zh)
        score, scores = scorer.compute_score(key2refs, key2pred)
        for n in range(4):
            f.write("Bleu-{}: {:6.3f}\n".format(n + 1, score[n]))

        scorer = Rouge(zh=zh)
        score, scores = scorer.compute_score(key2refs, key2pred)
        f.write("ROUGE: {:6.3f}\n".format(score))

        scorer = Cider(zh=zh)
        score, scores = scorer.compute_score(key2refs, key2pred)
        f.write("CIDEr: {:6.3f}\n".format(score))
        cider = score

        if not zh:
            scorer = Meteor()
            score, scores = scorer.compute_score(key2refs, key2pred)
            f.write("Meteor: {:6.3f}\n".format(score))

            scorer = Spice()
            score, scores = scorer.compute_score(key2refs, key2pred)
            f.write("Spice: {:6.3f}\n".format(score))
            spice = score
            f.write("Spider: {:6.3f}\n".format((cider + spice) / 2))

        # from audiocaptioneval.sentbert.sentencebert import SentenceBert
        # scorer = SentenceBert(zh=zh)
        # if caption_embedding_path is not None:
            # key2ref_embeds = np.load(caption_embedding_path, allow_pickle=True)
            # score, scores = scorer.compute_score(key2ref_embeds, key2pred)
        # else:
            # score, scores = scorer.compute_score(key2refs, key2pred)
        # f.write("SentenceBert: {:6.3f}\n".format(score))

        from utils.diverse_eval import diversity_evaluate
        score = diversity_evaluate(pred_df)
        f.write("Diversity: {:6.3f}\n".format(score))

        f.close()

    def _ensemble_batch(self, models, batch, **kwargs):
        method = kwargs.get("method", "greedy")
        max_length = kwargs.get("max_length", 20)
        start_idx = models[0].start_idx
        end_idx = models[0].end_idx
        num_models = len(models)
        with torch.no_grad():
            feats = batch[1]
            feat_lens = batch[-1]

            feats = feats.float().to(self.device)
            for model in models:
                model.eval()
            
            encoded = []
            for model in models:
                encoded.append(model.encoder(feats, feat_lens))

            if method == "beam":
                return self._ensemble_batch_beam_search(models, encoded, **kwargs)

            N = feats.shape[0]
            seqs = torch.empty(N, max_length, dtype=torch.long).fill_(end_idx)
            
            decoder_inputs = [{} for _ in range(num_models)]
            outputs = [{} for _ in range(num_models)]

            for t in range(max_length):
                # prepare input word (shared by all models)
                if t == 0:
                    w_t = torch.tensor([start_idx,] * N).long()
                else:
                    w_t = seqs[:, t-1]

                # prepare decoder input
                if t == 0:
                    for model_idx, model in enumerate(models):
                        cur_input = decoder_inputs[model_idx]
                        cur_input["state"] = model.decoder.init_hidden(N).to(self.device)
                        cur_input["enc_mem"] = encoded[model_idx]["audio_embeds"]
                        cur_input["enc_mem_lens"] = encoded[model_idx]["audio_embeds_lens"]
                else:
                    for model_idx, model in enumerate(models):
                        cur_input = decoder_inputs[model_idx]
                        cur_input["state"] = outputs[model_idx]["states"]

                for model_idx in range(num_models):
                    decoder_inputs[model_idx]["word"] = w_t.unsqueeze(1)

                # forward each model
                for model_idx, model in enumerate(models):
                    outputs[model_idx] = model.decoder(**decoder_inputs[model_idx])

                # ensemble word probabilities and greedy decoder word
                probs_t = torch.stack(
                    [torch.softmax(output["logits"].squeeze(1), -1) for output in outputs]).mean(dim=0)
                sampled_probs, w = torch.max(probs_t.detach(), dim=1)
                w = w.detach().long()
                seqs[:, t] = w

                # decide whether to stop
                unfinished_t = seqs[:, t] != end_idx
                if t == 0:
                    unfinished = unfinished_t
                else:
                    unfinished *= unfinished_t
                seqs[:, t][~unfinished] = end_idx
                if unfinished.sum() == 0:
                    break
        return {"seqs": seqs}
        
    def _ensemble_batch_beam_search(self, models, encoded, **kwargs):
        # encoded: [{"audio_embeds": ..., "audio_embeds_lens": ...}, {"audio_embeds": ..., "audio_embeds_lens": ...}, ...]
        beam_size = kwargs.get("beam_size", 5)
        N = len(encoded[0]["audio_embeds_lens"])
        max_length = kwargs.get("max_length", 20)
        start_idx = models[0].start_idx
        end_idx = models[0].end_idx
        vocab_size = models[0].vocab_size
        num_models = len(models)
        seqs = torch.empty(N, max_length, dtype=torch.long).fill_(end_idx)
        
        decoder_inputs = [{} for _ in range(num_models)]
        outputs = [{} for _ in range(num_models)]
        for i in range(N):
            top_k_logprobs = torch.zeros(beam_size).to(self.device)
            for t in range(max_length):
                # prepare input word (shared by all models)
                if t == 0:
                    w_t = torch.tensor([start_idx,] * beam_size).long()
                else:
                    w_t = next_word_inds

                # prepare decoder input
                if t == 0:
                    for model_idx, model in enumerate(models):
                        cur_input = decoder_inputs[model_idx]
                        cur_input["state"] = model.decoder.init_hidden(beam_size).to(self.device)
                        cur_input["enc_mem"] = encoded[model_idx]["audio_embeds"][i].unsqueeze(0).repeat(beam_size, 1, 1)
                        cur_input["enc_mem_lens"] = encoded[model_idx]["audio_embeds_lens"][i].repeat(beam_size)
                else:
                    for model_idx, model in enumerate(models):
                        cur_input = decoder_inputs[model_idx]
                        cur_input["state"] = outputs[model_idx]["states"][:, prev_word_inds, :].contiguous()

                for model_idx in range(num_models):
                    decoder_inputs[model_idx]["word"] = w_t.unsqueeze(1)

                # forward each model
                for model_idx, model in enumerate(models):
                    outputs[model_idx] = model.decoder(**decoder_inputs[model_idx])

                # ensemble word probabilities and greedy decoder word
                probs_t = torch.stack(
                    [torch.softmax(output["logits"].squeeze(1), -1) for output in outputs]).mean(dim=0)

                # merge current probability with previous timesteps to select words with the highest probs
                logprobs_t = torch.log(probs_t)
                logprobs_t = top_k_logprobs.unsqueeze(1).expand_as(logprobs_t) + logprobs_t
                if t == 0: # for the first step, all k seqs have the same probs
                    top_k_logprobs, top_k_words = logprobs_t[0].topk(beam_size, 0, True, True)
                else: # unroll and find top logprobs, and their unrolled indices
                    top_k_logprobs, top_k_words = logprobs_t.view(-1).topk(beam_size, 0, True, True)
                prev_word_inds = top_k_words // vocab_size
                next_word_inds = top_k_words % vocab_size

                # generate new beam
                if t == 0:
                    seqs_i = next_word_inds.unsqueeze(1)
                else:
                    seqs_i = torch.cat([seqs_i[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)
            seqs[i] = seqs_i[0]
        return {"seqs": seqs}




if __name__ == "__main__":
    fire.Fire(BaseRunner)
