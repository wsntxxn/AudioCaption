import json
import pickle
import random
from pathlib import Path
from typing import List, Union, Dict
import numpy as np
import pandas as pd
from tqdm import tqdm
import fire
import torch

import captioning.utils.train_util as train_util
import captioning.datasets as dataset_module


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

    def _get_dataloaders(self):
        dataloaders, key2refses = {}, {}
        for split in ["train", "val"]:
            data_config = self.config["data"][split]
            dataset_config = data_config["dataset"]
            dataset = getattr(dataset_module, dataset_config["type"])(
                **dataset_config["args"])
            collate_config = data_config["collate_fn"]
            collate_fn = getattr(dataset_module, collate_config["type"])(
                **collate_config["args"])
            if "batch_sampler" in data_config:
                bs_config = data_config["batch_sampler"]
                batch_sampler = getattr(dataset_module, bs_config["type"])(
                    **bs_config["args"])
            else:
                batch_sampler = None
            dataloader = torch.utils.data.DataLoader(
                dataset=dataset, collate_fn=collate_fn,
                batch_sampler=batch_sampler, **data_config["dataloader_args"])
            dataloaders[split] = dataloader

            key2refs = {}
            try:
                caption_info = dataset.caption_info
            except AttributeError:
                caption_info = json.load(open(data_config["caption"]))["audios"]

            for audio_idx in range(len(caption_info)):
                audio_id = caption_info[audio_idx]["audio_id"]
                key2refs[audio_id] = []
                for caption in caption_info[audio_idx]["captions"]:
                    key2refs[audio_id].append(caption[
                        "tokens" if self.config["zh"] else "caption"])
            key2refses[split] = key2refs

        return {
            "dataloader": dataloaders,
            "key2refs": key2refses
        }

    def _get_model(self):
        raise NotImplementedError

    def _forward(self, model, batch, training, **kwargs):
        raise NotImplementedError

    def _decode_to_sentence(self, word_ids, vocabulary):
        words = []
        for word_id in word_ids:
            word = vocabulary[word_id]
            if word == "<end>":
                break
            elif word == "<start>":
                continue
            words.append(word)
        sentence = " ".join(words)
        return sentence

    def train(self, config, **kwargs):
        raise NotImplementedError

    def _eval_prediction(self, key2refs, key2pred, scorers,
            pretokenized=False, per_audio=False):
        output = {}
        if per_audio:
            output["per_audio"] = {}
        for scorer in scorers:
            if scorer.method() == "Fense":
                score, scores = scorer.compute_score(key2refs, key2pred)
                output[scorer.method()] = score
                if per_audio:
                    output["per_audio"][scorer.method()] = dict(zip(
                        key2refs.keys(), scores))

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

        for scorer in scorers:
            if scorer.method() != "Fense":
                score, scores = scorer.compute_score(key2refs, key2pred)
                output[scorer.method()] = score
                if per_audio:
                    if scorer.method() == "Bleu":
                        output["per_audio"][scorer.method()] = dict(zip(
                            key2refs.keys(), scores[3]))
                    elif scorer.method() == "SPICE":
                        scores = np.array([item["All"]["f"] for item in scores])
                        output["per_audio"][scorer.method()] = dict(zip(
                            sorted(key2refs.keys()), scores))
                    else:
                        output["per_audio"][scorer.method()] = dict(zip(
                            key2refs.keys(), scores))
        return output

    def evaluate_prediction(self,
                            caption_file,
                            system_output,
                            score_output=None,
                            zh=False,
                            system_output_index=None,
                            per_audio=False):
        captions = json.load(open(caption_file, "r"))["audios"]
        key2refs = {}
        for audio_idx in range(len(captions)):
            audio_id = captions[audio_idx]["audio_id"]
            key2refs[audio_id] = []
            for caption in captions[audio_idx]["captions"]:
                key2refs[audio_id].append(caption["tokens" if zh else "caption"])
        key2pred = {}
        key2idx = {}
        predictions = json.load(open(system_output, "r"))["predictions"]
        for idx, pred_item in enumerate(predictions):
            if system_output_index is not None:
                pred = pred_item["tokens"][system_output_index]
            else:
                pred = pred_item["tokens"]
            audio_id = pred_item["filename"]
            key2idx[audio_id] = idx
            key2pred[audio_id] = [pred,]
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.spice.spice import Spice
        from fense.fense import Fense
        scorers = [Bleu(n=4), Rouge(), Cider()]
        if not zh:
            scorers.append(Meteor())
            scorers.append(Spice())
            scorers.append(Fense())
        scores_output = self._eval_prediction(key2refs, key2pred, scorers,
            pretokenized=zh, per_audio=per_audio)
        
        if per_audio:
            for name, key2score in scores_output["per_audio"].items():
                for key in key2refs.keys():
                    predictions[key2idx[key]][name] = f"{key2score[key]:.3f}"

            for key in key2refs.keys():
                spider = (scores_output["per_audio"]["CIDEr"][key] + 
                    scores_output["per_audio"]["SPICE"][key]) / 2
                predictions[key2idx[key]]["SPIDEr"] = f"{spider:.3f}"

            if score_output:
                score_output = Path(score_output)
                if not score_output.parent.exists():
                    score_output.parent.mkdir(parents=True)
            json.dump({"predictions": predictions}, open(score_output, "w"),
                      indent=4)
        else:
            spider = 0
            for name, score in scores_output.items():
                if name == "Bleu":
                    for n in range(4):
                        print("Bleu-{}: {:6.3f}".format(n + 1, score[n]))
                else:
                    print("{}: {:6.3f}".format(name, score))
                    if name in ["CIDEr", "SPICE"]:
                        spider += score
            if not zh:
                print("SPIDEr: {:6.3f}".format(spider / 2))

            if score_output:
                score_output = Path(score_output)
                if not score_output.parent.exists():
                    score_output.parent.mkdir(parents=True)
                with open(score_output, "w") as f:
                    for name, score in scores_output.items():
                        if name == "Bleu":
                            for n in range(4):
                                print(f"Bleu-{n + 1}: {score[n]:6.3f}",
                                      file=f)
                        else:
                            print(f"{name}: {score:6.3f}", file=f)
                    if not zh:
                        print(f"SPIDEr: {spider / 2:6.3f}", file=f)

    def _inference(self, model, dataloader, vocabulary, **inference_args):
        model.eval()
        key2pred = {}
        with torch.no_grad(), tqdm(total=len(dataloader),
            ncols=100, ascii=True, leave=False) as pbar:
            for batch in dataloader:
                output = self._forward(model, batch, training=False,
                                       **inference_args)
                keys = batch["audio_id"]
                seqs = output["seq"].cpu().numpy()
                for (idx, seq) in enumerate(seqs):
                    candidate = self._decode_to_sentence(seq, vocabulary)
                    key2pred[keys[idx]] = [candidate,]
                pbar.update()
        return key2pred

    def predict(self,
                experiment_path: str,
                eval_config: Union[str, Dict],
                return_pred: bool = False,
                **kwargs):
        experiment_path = Path(experiment_path)
        if not isinstance(eval_config, dict):
            eval_config = train_util.parse_config_or_kwargs(eval_config, **kwargs)
        resume_path = experiment_path / eval_config['resume']
        checkpoint = torch.load(resume_path, "cpu")
        # Previous training config
        self.config = train_util.parse_config_or_kwargs(
            experiment_path / "config.yaml")

        self.vocabulary = checkpoint["vocabulary"]
        model = self._get_model()
        model.load_state_dict(checkpoint["model"])
        model = model.to(self.device)

        dataset_config = eval_config["data"]["test"]["dataset"]
        dataset = getattr(dataset_module, dataset_config["type"])(
            **dataset_config["args"])
        collate_config = eval_config["data"]["test"]["collate_fn"]
        collate_fn = getattr(dataset_module, collate_config["type"])(
            **collate_config["args"])
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset, collate_fn=collate_fn,
            **eval_config["data"]["test"]["dataloader_args"])

        key2pred = self._inference(model, dataloader, self.vocabulary,
            **eval_config["inference_args"])

        pred_data = []
        for key, pred in key2pred.items():
            pred_data.append({
                "filename": key,
                "tokens": pred[0]
            })
        output_file = experiment_path / eval_config["caption_output"]
        if not output_file.parent.exists():
            output_file.parent.mkdir(parents=True)
        json.dump(
            {"predictions": pred_data}, open(output_file , "w"), indent=4)

        if return_pred:
            return key2pred

    def evaluate(self,
                 experiment_path: str,
                 eval_config: str,
                 **kwargs):
        experiment_path = Path(experiment_path)
        eval_config = train_util.parse_config_or_kwargs(eval_config, **kwargs)

        key2pred = self.predict(experiment_path,
                                eval_config,
                                return_pred=True)
        zh = self.config["zh"]
        caption_file = eval_config["data"]["test"]["caption"]
        captions = json.load(open(caption_file, "r"))["audios"]
        key2refs = {}
        for audio_idx in range(len(captions)):
            audio_id = captions[audio_idx]["audio_id"]
            key2refs[audio_id] = []
            for caption in captions[audio_idx]["captions"]:
                key2refs[audio_id].append(caption["tokens" if zh else "caption"])

        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.spice.spice import Spice
        from fense.fense import Fense

        scorers = [Bleu(n=4), Rouge(), Cider()]
        if not zh:
            scorers.append(Meteor())
            scorers.append(Spice())
            scorers.append(Fense())
        result = self._eval_prediction(key2refs, key2pred, scorers)

        output_filename = experiment_path / eval_config["score_output"]
        if not output_filename.parent.exists():
            output_filename.parent.mkdir(parents=True)
        with open(output_filename, "w") as f:
            spider = 0
            for name, score in result.items():
                if name == "Bleu":
                    for n in range(4):
                        f.write("Bleu-{}: {:6.3f}\n".format(n + 1, score[n]))
                else:
                    f.write("{}: {:6.3f}\n".format(name, score))
                    if name in ["CIDEr", "SPICE"]:
                        spider += score
            if not zh:
                f.write("SPIDEr: {:6.3f}\n".format(spider / 2))

    def train_evaluate(self, config, task, **kwargs):
        experiment_path = self.train(config, **kwargs)
        for save_type in ["best", "last", "swa"]:
            self.evaluate(experiment_path, task, save_type=save_type,
                score_output=f"{save_type}.txt", caption_output=f"{save_type}.json",
                method="beam")
        return experiment_path


if __name__ == "__main__":
    fire.Fire(BaseRunner)
