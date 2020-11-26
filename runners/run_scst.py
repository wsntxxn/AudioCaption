# coding=utf-8
#!/usr/bin/env python3
import os
import re
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
import sklearn.preprocessing as pre
import torch
from ignite.engine.engine import Engine, Events
from ignite.utils import convert_tensor
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.contrib.handlers import ProgressBar
from ignite.metrics import Accuracy, Loss, RunningAverage, Average

sys.path.append(os.getcwd())
import models
import utils.train_util as train_util
from utils.build_vocab import Vocabulary
from datasets.SJTUDataSet import SJTUDataset, collate_fn
from runners.base_runner import BaseRunner


class ScstRunner(BaseRunner):
    """Main class to run experiments"""

    @staticmethod
    def _get_model(config, vocabulary):
        embed_size = config["basemodel_args"]["embed_size"]
        encodermodel = getattr(
            models.encoder, config["encodermodel"])(
            inputdim=config["inputdim"], 
            embed_size=embed_size,
            **config["encodermodel_args"])
        if config["decodermodel"] == "RNNBahdanauAttnDecoder":
            input_size = embed_size * 2
        else:
            input_size = embed_size
        decodermodel = getattr(
            models.decoder, config["decodermodel"])(
            vocab_size=len(vocabulary),
            input_size=input_size,
            **config["decodermodel_args"])

        basemodel = getattr(models, config["basemodel"])(
            encodermodel, decodermodel, **config["basemodel_args"])

        if config["load_pretrained"]:
            dump = torch.load(
                config["pretrained"], map_location="cpu")
            basemodel.load_state_dict(dump["model"].state_dict(), strict=False)

        model = getattr(models.SeqTrainModel, config["modelwrapper"])(
                basemodel, vocabulary)

        return model

    def _forward(self, model, batch, mode, **kwargs):
        assert mode in ("train", "sample")

        if mode == "sample":
            # SJTUDataSetEval
            feats = batch[1]
            feat_lens = batch[-1]

            feats = convert_tensor(feats.float(),
                                   device=self.device,
                                   non_blocking=True)
            sampled = model(feats, feat_lens, mode="sample", **kwargs)
            return sampled

        # mode is "train"
        assert "train_mode" in kwargs, "need to provide training mode (XE or scst)"
        assert kwargs["train_mode"] in ("XE", "scst"), "unknown training mode"

        feats = batch[0]
        caps = batch[1]
        keys = batch[2]
        feat_lens = batch[-2]
        cap_lens = batch[-1]
        feats = convert_tensor(feats.float(),
                               device=self.device,
                               non_blocking=True)
        caps = convert_tensor(caps.long(),
                              device=self.device,
                              non_blocking=True)

        
        assert "key2refs" in kwargs, "missing references"
        output = model(feats, feat_lens, keys, kwargs["key2refs"], 
        # output = model(feats, feat_lens, keys, caps, 
                       max_length=max(cap_lens), scorer=kwargs["scorer"])
        
        return output

    def train(self, config, **kwargs):
        """Trains a model on the given configurations.
        :param config:str: A training configuration. Note that all parameters in the config can also be manually adjusted with --ARG=VALUE
        :param **kwargs: parameters to overwrite yaml config
        """

        from pycocoevalcap.cider.cider import Cider

        config_parameters = train_util.parse_config_or_kwargs(config, **kwargs)
        config_parameters["seed"] = self.seed
        zh = config_parameters["zh"]
        outputdir = os.path.join(
            config_parameters["outputpath"], config_parameters["modelwrapper"],
            "{}_{}".format(
                datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%m'),
                uuid.uuid1().hex))

        # Early init because of creating dir
        checkpoint_handler = ModelCheckpoint(
            outputdir,
            "run",
            n_saved=1,
            require_empty=False,
            create_dir=True,
            score_function=lambda engine: -engine.state.metrics["loss"],
            score_name="loss")

        logger = train_util.genlogger(os.path.join(outputdir, "train.log"))
        # print passed config parameters
        logger.info("Storing files in: {}".format(outputdir))
        train_util.pprint_dict(config_parameters, logger.info)

        vocabulary = torch.load(config_parameters["vocab_file"])
        train_loader, val_loader, info = self._get_dataloaders(config_parameters, vocabulary)
        config_parameters["inputdim"] = info["inputdim"]
        logger.info("<== Estimating Scaler ({}) ==>".format(info["scaler"].__class__.__name__))
        logger.info(
                "Feature: {} Input dimension: {} Vocab Size: {}".format(
                config_parameters["feature_file"], info["inputdim"], len(vocabulary)))
        train_key2refs = info["train_key2refs"]
        # train_scorer = BatchCider(train_key2refs)
        val_key2refs = info["val_key2refs"]
        # cv_scorer = BatchCider(cv_key2refs)

        model = self._get_model(config_parameters, vocabulary)
        model = model.to(self.device)
        train_util.pprint_dict(model, logger.info, formatter="pretty")
        optimizer = getattr(
            torch.optim, config_parameters["optimizer"]
        )(model.parameters(), **config_parameters["optimizer_args"])
        train_util.pprint_dict(optimizer, logger.info, formatter="pretty")

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            # optimizer, **config_parameters["scheduler_args"])
        crtrn_imprvd = train_util.criterion_improver(config_parameters["improvecriterion"])

        def _train_batch(engine, batch):
            model.train()
            with torch.enable_grad():
                optimizer.zero_grad()
                train_scorer = Cider(zh=zh)
                output = self._forward(model, batch, "train", train_mode="scst", 
                                       key2refs=train_key2refs, scorer=train_scorer)
                output["loss"].backward()
                optimizer.step()
                return output

        trainer = Engine(_train_batch)
        RunningAverage(output_transform=lambda x: x["loss"]).attach(trainer, "running_loss")
        pbar = ProgressBar(persist=False, ascii=True)
        pbar.attach(trainer, ["running_loss"])

        key2pred = {}

        def _inference(engine, batch):
            model.eval()
            keys = batch[2]
            with torch.no_grad():
                cv_scorer = Cider(zh=zh)
                output = self._forward(model, batch, "train", train_mode="scst",
                                       key2refs=val_key2refs, scorer=cv_scorer)
                seqs = output["sampled_seqs"].cpu().numpy()
                for idx, seq in enumerate(seqs):
                    if keys[idx] in key2pred:
                        continue
                    candidate = self._convert_idx2sentence(seq, vocabulary, zh=zh)
                    key2pred[keys[idx]] = [candidate,]
                return output

        evaluator = Engine(_inference)
        RunningAverage(output_transform=lambda x: x["loss"]).attach(trainer, "running_loss")

        metrics = {
            "loss": Average(output_transform=lambda x: x["loss"]),
            "reward": Average(output_transform=lambda x: x["reward"].reshape(-1, 1)),
            # "score": Average(output_transform=lambda x: x["score"].reshape(-1, 1)),
        }

        for name, metric in metrics.items():
            metric.attach(trainer, name)
            metric.attach(evaluator, name)

        RunningAverage(output_transform=lambda x: x["loss"]).attach(evaluator, "running_loss")
        pbar.attach(evaluator, ["running_loss"])

        # @trainer.on(Events.STARTED)
        # def log_initial_result(engine):
            # evaluator.run(cvloader, max_epochs=1)
            # logger.info("Initial Results - loss: {:<5.2f}\tscore: {:<5.2f}".format(evaluator.state.metrics["loss"], evaluator.state.metrics["score"].item()))


        trainer.add_event_handler(
              Events.EPOCH_COMPLETED, train_util.log_results, evaluator, val_loader,
              logger.info, ["loss", "reward"], ["loss", "reward", "score"])

        def eval_cv(engine, key2pred, key2refs, scorer):
            score, scores = scorer.compute_score(key2refs, key2pred)
            engine.state.metrics["score"] = score
            key2pred.clear()

        evaluator.add_event_handler(
            Events.EPOCH_COMPLETED, eval_cv, key2pred, val_key2refs, Cider(zh=zh))

        evaluator.add_event_handler(
            Events.EPOCH_COMPLETED, train_util.save_model_on_improved, crtrn_imprvd,
            "score", {
                "model": model,
                "config": config_parameters,
                "scaler": info["scaler"]
            }, os.path.join(outputdir, "saved.pth"))

        evaluator.add_event_handler(
            Events.EPOCH_COMPLETED, checkpoint_handler, {
                "model": model,
            }
        )

        trainer.run(train_loader, max_epochs=config_parameters["epochs"])
        return outputdir


if __name__ == "__main__":
    fire.Fire(ScstRunner)
