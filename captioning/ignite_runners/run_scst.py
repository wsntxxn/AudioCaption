# coding=utf-8
#!/usr/bin/env python3
import os
from pathlib import Path
import pickle
import sys
import logging
import datetime
import random
import uuid
from pprint import pformat

from tqdm import tqdm
import fire
import numpy as np
import pandas as pd
import torch
from ignite.engine.engine import Engine, Events
from ignite.utils import convert_tensor
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.contrib.handlers import ProgressBar
from ignite.metrics import Accuracy, Loss, RunningAverage, Average

import captioning.models
import captioning.utils.train_util as train_util
from captioning.utils.build_vocab import Vocabulary
from captioning.ignite_runners.run import Runner as XeRunner


class Runner(XeRunner):
    """Main class to run experiments"""

    def _get_model(self, config, outputfun=sys.stdout):
        basemodel = super()._get_model(config, outputfun)
        model = getattr(captioning.models.rl_model, config["modelwrapper"])(basemodel)
        return model

    def _forward(self, model, batch, mode, **kwargs):
        assert mode in ("train", "validation", "eval")

        if mode == "train":
            raw_feats = batch[0]
            fc_feats = batch[1]
            attn_feats = batch[2]
            raw_feat_lens = batch[-3]
            attn_feat_lens = batch[-2]
        else:
            raw_feats = batch[1]
            fc_feats = batch[2]
            attn_feats = batch[3]
            raw_feat_lens = batch[-2]
            attn_feat_lens = batch[-1]

        raw_feats = convert_tensor(raw_feats.float(),
                                   device=self.device,
                                   non_blocking=True)
        fc_feats = convert_tensor(fc_feats.float(),
                                  device=self.device,
                                  non_blocking=True)
        attn_feats = convert_tensor(attn_feats.float(),
                                    device=self.device,
                                    non_blocking=True)
        
        input_dict = {
            "mode": "train" if mode == "train" else "inference",
            "raw_feats": raw_feats,
            "raw_feat_lens": raw_feat_lens,
            "fc_feats": fc_feats,
            "attn_feats": attn_feats,
            "attn_feat_lens": attn_feat_lens
        }

        if mode == "train":
            for item in ["key2refs", "vocabulary", "scorer"]:
                assert item in kwargs, f"missing {item} in scst"
            input_dict["keys"] = batch[4]
            input_dict.update(kwargs)
            # output = model(feats, feat_lens, keys, kwargs["key2refs"], kwargs["vocabulary"],
                           # max_length=max(cap_lens)-1, scorer=kwargs["scorer"])
            output = model(input_dict)
        else:
            input_dict.update(kwargs)
            output = model(input_dict)
        return output

    def train(self, config, **kwargs):
        """Trains a model on the given configurations.
        :param config:str: A training configuration. Note that all parameters in the config can also be manually adjusted with --ARG=VALUE
        :param **kwargs: parameters to overwrite yaml config
        """

        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.spider.spider import Spider

        conf = train_util.parse_config_or_kwargs(config, **kwargs)
        conf["seed"] = self.seed
        outputdir = Path(conf["outputpath"]) / conf["modelwrapper"] / \
            conf["remark"] / "seed_{}".format(self.seed)
        outputdir.mkdir(parents=True, exist_ok=True)

        logger = train_util.genlogger(str(outputdir / "train.log"))
        if "SLURM_JOB_ID" in os.environ:
            logger.info(f"Slurm job id: {os.environ['SLURM_JOB_ID']}")
            logger.info(f"Slurm node: {os.environ['SLURM_JOB_NODELIST']}")
        logger.info("Storing files in: {}".format(outputdir))
        train_util.pprint_dict(conf, logger.info)

        vocabulary = pickle.load(open(conf["data"]["vocab_file"], "rb"))
        conf["vocabulary"] = vocabulary
        dataloaders = self._get_dataloaders(conf)
        train_dataloader = dataloaders["train_dataloader"]
        val_dataloader = dataloaders["val_dataloader"]
        train_key2refs = dataloaders["train_key2refs"]
        val_key2refs = dataloaders["val_key2refs"]
        conf["data"]["raw_feat_dim"] = train_dataloader.dataset.raw_feat_dim
        conf["data"]["fc_feat_dim"] = train_dataloader.dataset.fc_feat_dim
        conf["data"]["attn_feat_dim"] = train_dataloader.dataset.attn_feat_dim

        model = self._get_model(conf, logger.info)
        model = model.to(self.device)
        train_util.pprint_dict(model, logger.info, formatter="pretty")
        optimizer = getattr(
            torch.optim, conf["optimizer"]
        )(model.parameters(), **conf["optimizer_args"])
        train_util.pprint_dict(optimizer, logger.info, formatter="pretty")

        crtrn_imprvd = train_util.criterion_improver(conf["improvecriterion"])

        scorer_dict = {"cider": Cider(), "spider": Spider()}
        if "train_scorer" not in conf:
            conf["train_scorer"] = "cider"
        train_scorer = scorer_dict[conf["train_scorer"]]
        def _train_batch(engine, batch):
            model.train()
            with torch.enable_grad(), torch.autograd.set_detect_anomaly(True):
                optimizer.zero_grad()
                output = self._forward(model, batch, "train",
                                       key2refs=train_key2refs,
                                       scorer=train_scorer,
                                       vocabulary=vocabulary)
                output["loss"].backward()
                optimizer.step()
                return output

        trainer = Engine(_train_batch)
        RunningAverage(output_transform=lambda x: x["loss"]).attach(trainer, "running_loss")
        pbar = ProgressBar(persist=False, ascii=True)
        pbar.attach(trainer, ["running_loss"])

        metrics = {
            "loss": Average(output_transform=lambda x: x["loss"]),
            "reward": Average(output_transform=lambda x: x["reward"].reshape(-1, 1)),
        }

        for name, metric in metrics.items():
            metric.attach(trainer, name)

        # stochastic weight averaging
        if conf["swa"]:
            swa_model = torch.optim.swa_utils.AveragedModel(model)
            @trainer.on(Events.EPOCH_COMPLETED)
            def update_swa(engine):
                if engine.state.epoch >= conf["swa_start"]:
                    swa_model.update_parameters(model)

        key2pred = {}

        def _inference(engine, batch):
            model.eval()
            keys = batch[0]
            with torch.no_grad():
                output = self._forward(model, batch, "validation", sample_method="beam", beam_size=3)
                seqs = output["seqs"].cpu().numpy()
                for idx, seq in enumerate(seqs):
                    candidate = self._convert_idx2sentence(seq, vocabulary.idx2word)
                    key2pred[keys[idx]] = [candidate,]
                return output

        evaluator = Engine(_inference)
        pbar.attach(evaluator)

        def eval_val(engine):
            scorer = Cider()
            score_output = self._eval_prediction(val_key2refs, key2pred, [scorer])
            engine.state.metrics["score"] = score_output["CIDEr"]
            key2pred.clear()
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, eval_val)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_results(engine):
            train_results = engine.state.metrics
            evaluator.run(val_dataloader)
            val_results = evaluator.state.metrics
            output_str_list = [
                "Validation Results - Epoch : {:<4}".format(engine.state.epoch)
            ]
            for metric in metrics:
                output = train_results[metric]
                if isinstance(output, torch.Tensor):
                    output = output.item()
                output_str_list.append("{} {:<5.2g} ".format(
                    metric, output))
            for metric in ["score"]:
                output = val_results[metric]
                if isinstance(output, torch.Tensor):
                    output = output.item()
                output_str_list.append("{} {:5<.2g} ".format(
                    metric, output))
            lr = optimizer.param_groups[0]["lr"]
            output_str_list.append(f"lr {lr:5<.2g} ")
            logger.info(" ".join(output_str_list))

        # saving best model
        @evaluator.on(Events.EPOCH_COMPLETED)
        def save_model(engine):
            dump = {
                "model": model.state_dict() if not conf["distributed"] else model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "vocabulary": vocabulary.idx2word
            }
            if crtrn_imprvd(engine.state.metrics["score"]):
                torch.save(dump, outputdir / "best.pth")
            torch.save(dump, outputdir / "last.pth")


        # dump configuration
        del conf["vocabulary"]
        train_util.store_yaml(conf, outputdir / "config.yaml")

        trainer.run(train_dataloader, max_epochs=conf["epochs"])

        # stochastic weight averaging
        if conf["swa"]:
            torch.save({
                "model": swa_model.module.state_dict(),
                "vocabulary": vocabulary.idx2word
            }, outputdir / "swa.pth")

        return outputdir


if __name__ == "__main__":
    fire.Fire(Runner)
