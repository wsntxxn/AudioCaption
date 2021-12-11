# coding=utf-8
#!/usr/bin/env python3
import os
import sys
import datetime
import random
import uuid

import fire
import numpy as np
import pandas as pd
import sklearn.preprocessing as pre
import torch
from ignite.engine.engine import Engine, Events
from ignite.metrics import Accuracy, Loss, RunningAverage, Average
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.contrib.handlers import ProgressBar
from ignite.utils import convert_tensor

sys.path.append(os.getcwd())
import models
import utils.train_util as train_util
from utils.build_vocab import Vocabulary
from runners.base_runner import BaseRunner
from datasets.SJTUDataSet import SJTUDataset, SJTUDatasetEval, collate_fn

class Runner(BaseRunner):

    def _get_model(self, config, vocabulary):
        vocab_size = len(vocabulary)
        if config["encodermodel"] == "E2EASREncoder":
            encodermodel = models.encoder.load_espnet_encoder(config["pretrained_encoder"], pretrained=config["load_encoder_params"])
        else:
            encodermodel = getattr(
                models.encoder, config["encodermodel"])(
                inputdim=config["inputdim"],
                **config["encodermodel_args"])
            if "pretrained_encoder" in config:
                encoder_state_dict = torch.load(
                    config["pretrained_encoder"],
                    map_location="cpu")
                encodermodel.load_state_dict(encoder_state_dict, strict=False)

        decodermodel = getattr(
            models.decoder, config["decodermodel"])(
            vocab_size=vocab_size,
            enc_mem_size=config["encodermodel_args"]["embed_size"],
            **config["decodermodel_args"])
        if "pretrained_word_embedding" in config:
            embeddings = np.load(config["pretrained_word_embedding"])
            decodermodel.load_word_embeddings(
                embeddings, 
                tune=config["tune_word_embedding"], 
                projection=True
            )
        model = getattr(
            models, config["model"])(encodermodel, decodermodel, **config["model_args"])
        if "load_pretrained" in config and config["load_pretrained"]:
            pretrained_state_dict = torch.load(config["pretrained"], map_location="cpu")["model"]
            model.load_state_dict(pretrained_state_dict)
        return model

    def _forward(self, model, batch, mode, **kwargs):
        assert mode in ("train", "validation", "eval")

        if mode == "eval":
            feats = batch[1]
            feat_lens = batch[-1]
        else:
            feats = batch[0]
            caps = batch[1]
            feat_lens = batch[-2]
            cap_lens = batch[-1]

        feats = convert_tensor(feats.float(),
                               device=self.device,
                               non_blocking=True)

        if mode == "train":
            caps = convert_tensor(caps.long(),
                                  device=self.device,
                                  non_blocking=True)
            # pack labels to remove padding from caption labels
            #  targets = torch.nn.utils.rnn.pack_padded_sequence(
                #  caps[:, 1:], cap_lens - 1, batch_first=True).data
            targets = torch.nn.utils.rnn.pack_padded_sequence(
                caps, cap_lens, batch_first=True).data

            output = model(feats, feat_lens, caps, cap_lens, **kwargs)

            #  packed_logits = torch.nn.utils.rnn.pack_padded_sequence(
                #  output["logits"], cap_lens - 1, batch_first=True).data
            packed_logits = torch.nn.utils.rnn.pack_padded_sequence(
                output["logits"], cap_lens, batch_first=True).data
            packed_logits = convert_tensor(
                packed_logits, device=self.device, non_blocking=True)

            output["packed_logits"] = packed_logits
            output["targets"] = targets
        else:
            output = model(feats, feat_lens, **kwargs)

        return output

    def train(self, config, **kwargs):
        """Trains a model on the given configurations.
        :param config: A training configuration. Note that all parameters in the config can also be manually adjusted with --ARG=VALUE
        :param **kwargs: parameters to overwrite yaml config
        """

        conf = train_util.parse_config_or_kwargs(config, **kwargs)
        conf["seed"] = self.seed

        zh = conf["zh"]
        vocabulary = torch.load(conf["vocab_file"])
        train_loader, val_loader, info = self._get_dataloaders(conf, vocabulary)
        conf["inputdim"] = info["inputdim"]
        val_key2refs = info["val_key2refs"]

        model = self._get_model(conf, vocabulary)
        model = model.to(self.device)
        optimizer = getattr(
            torch.optim, conf["optimizer"]
        )(model.parameters(), **conf["optimizer_args"])


        criterion = torch.nn.CrossEntropyLoss().to(self.device)
        crtrn_imprvd = train_util.criterion_improver(conf['improvecriterion'])

        train_dataiter = iter(train_loader)
        batch = next(train_dataiter)
        model.train()
        with torch.enable_grad():
            optimizer.zero_grad()
            output = self._forward(
                model, batch, "train",
                ss_ratio=conf["ss_args"]["ss_ratio"]
            )
            loss = criterion(output["packed_logits"], output["targets"]).to(self.device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        print(output["logits"])



if __name__ == "__main__":
    fire.Fire(Runner)
