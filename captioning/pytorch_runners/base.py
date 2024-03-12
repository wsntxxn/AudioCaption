import sys
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


class BaseRunner(object):
    """Main class to run experiments"""
    def __init__(self,):
        super(BaseRunner, self).__init__()
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
            # torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        self.device = torch.device(device)

    def _get_dataloaders(self):
        dataloaders, key2refses = {}, {}
        for split in ["train", "val"]:
            data_config = self.config["data"][split]
            dataset = train_util.init_obj_from_dict(data_config["dataset"])
            collate_fn = train_util.init_obj_from_dict(data_config["collate_fn"])
            if "batch_sampler" in data_config:
                batch_sampler = train_util.init_obj_from_dict(
                    data_config["batch_sampler"], dataset=dataset)
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

    def _get_model(self, print_fn=sys.stdout.write):
        raise NotImplementedError

    def _forward(self, batch, training=False):
        raise NotImplementedError

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

    def _inference(self, dataloader):
        self.model.eval()
        key2pred = {}
        with torch.no_grad(), tqdm(total=len(dataloader),
            ncols=100, ascii=True, leave=False) as pbar:
            for batch in dataloader:
                output = self._forward(batch, training=False)
                keys = batch["audio_id"]
                seqs = output["seq"].cpu().numpy()
                for (idx, seq) in enumerate(self.tokenizer.decode(seqs)):
                    key2pred[keys[idx]] = [seq,]
                pbar.update()
        return key2pred

    def load_tokenizer(self):
        cfg = self.config["data"]["train"]["collate_fn"]["tokenizer"]
        tokenizer = train_util.init_obj_from_dict(cfg)
        return tokenizer

    def save_checkpoint(self, ckpt_path):
        model_dict = self.model.state_dict()
        ckpt = {
            "model": { k: model_dict[k] for k in self.saving_keys },
            "epoch": self.epoch,
            "metric_monitor": self.metric_monitor.state_dict(),
            "not_improve_cnt": self.not_improve_cnt,
        }
        if self.tokenizer.__class__.__name__ == "DictTokenizer":
            ckpt["tokenizer"] = self.tokenizer.state_dict()
        if self.include_optim_in_ckpt:
            ckpt["optimizer"] = self.optimizer.state_dict()
            ckpt["lr_scheduler"] = self.lr_scheduler.state_dict()
        torch.save(ckpt, ckpt_path)

    def resume_checkpoint(self, finetune=False):
        ckpt = torch.load(self.config["resume"], "cpu")
        train_util.load_pretrained_model(self.model, ckpt)
        if not hasattr(self, "tokenizer"):
            self.tokenizer = self.load_tokenizer()
            if not self.tokenizer.loaded:
                self.tokenizer.load_state_dict(ckpt["tokenizer"])
            self.model.set_index(self.tokenizer.bos, self.tokenizer.eos, self.tokenizer.pad)
        if not finetune:
            self.epoch = ckpt["statistics"]["epoch"]
            self.metric_monitor.load_state_dict(ckpt["metric_monitor"])
            self.not_improve_cnt = ckpt["not_improve_cnt"]
            if self.optimizer.__class__.__name__ == "Adam":
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.lr_scheduler.load_state_dict(ckpt["lr_scheduler"])

    def predict(self,
                experiment_path: Union[str, Path],
                eval_config: Union[str, Dict],
                return_pred: bool = False,
                **kwargs):
        experiment_path = Path(experiment_path)
        if not isinstance(eval_config, dict):
            eval_config = train_util.parse_config_or_kwargs(eval_config, **kwargs)
        resume_path = experiment_path / eval_config['resume']
        self.config = train_util.parse_config_or_kwargs(
            experiment_path / "config.yaml")
        self.config["resume"] = resume_path

        self.model = self._get_model()
        self.resume_checkpoint(finetune=True)
        self.model = self.model.to(self.device)

        dataset_config = eval_config["data"]["test"]["dataset"]
        dataset = train_util.init_obj_from_dict(dataset_config)
        collate_config = eval_config["data"]["test"]["collate_fn"]
        collate_fn = train_util.init_obj_from_dict(collate_config)
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset, collate_fn=collate_fn,
            **eval_config["data"]["test"]["dataloader_args"])
        
        self.config["inference_args"] = eval_config["inference_args"]

        key2pred = self._inference(dataloader)

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

    def train_evaluate(self, train_config, eval_config, **kwargs):
        experiment_path = self.train(train_config, **kwargs)
        self.evaluate(experiment_path, eval_config)
        return experiment_path


if __name__ == "__main__":
    fire.Fire(BaseRunner)
