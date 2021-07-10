# coding=utf-8
#!/usr/bin/env python3
import os
import sys
import datetime
import random
import uuid
import re
from pprint import pformat

from tqdm import tqdm
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
from datasets.SJTUDataSet import SJTUSentenceDataset, collate_fn
import utils.score_util as score_util
from runners.run import Runner as XeRunner

class Runner(XeRunner):

    @staticmethod
    def _get_dataloaders(config, vocabulary):
        scaler = getattr(
            pre, config["scaler"])(
            **config["scaler_args"])
        inputdim = -1
        if "caption_file" in config:
            caption_df = pd.read_json(config["caption_file"], dtype={"key": str})
            train_keys = np.random.choice(
                caption_df["key"].unique(), 
                int(len(caption_df["key"].unique()) * (config["train_percent"] / 100.)), 
                replace=False
            )
            train_df = caption_df[caption_df["key"].apply(lambda x: x in train_keys)]
            val_df = caption_df[~caption_df.index.isin(train_df.index)]
        else:
            train_df = pd.read_json(config["caption_file_train"], dtype={"key": str})
            val_df = pd.read_json(config["caption_file_val"], dtype={"key": str})
            caption_df = pd.concat((train_df, val_df))
        sentence_embedding = np.load(config["sentence_embedding"], allow_pickle=True)

        for batch in tqdm(
            torch.utils.data.DataLoader(
                SJTUSentenceDataset(
                    feature=config["feature_file"],
                    caption_df=caption_df,
                    vocabulary=vocabulary,
                    sentence_embedding=sentence_embedding,
                ),
                collate_fn=collate_fn([0, 1]),
                **config["dataloader_args"]
            ), 
            ascii=True,
            ncols=100
        ):
            feat = batch[0]
            feat_lens = batch[-2]
            packed_feat = torch.nn.utils.rnn.pack_padded_sequence(
                feat, feat_lens, batch_first=True, enforce_sorted=False).data
            scaler.partial_fit(packed_feat)
            inputdim = feat.shape[-1]

        augments = train_util.parse_augments(config["augments"])

        trainloader = torch.utils.data.DataLoader(
            SJTUSentenceDataset(
                feature=config["feature_file"],
                caption_df=train_df,
                vocabulary=vocabulary,
                sentence_embedding=sentence_embedding,
                transform=[augments, scaler.transform]
            ),
            shuffle=True,
            collate_fn=collate_fn([0, 1]),
            **config["dataloader_args"]
        )

        if config["zh"]:
            train_key2refs = train_df.groupby("key")["tokens"].apply(list).to_dict()
            val_key2refs = val_df.groupby("key")["tokens"].apply(list).to_dict()
        else:
            train_key2refs = train_df.groupby("key")["caption"].apply(list).to_dict()
            val_key2refs = val_df.groupby("key")["caption"].apply(list).to_dict()
        val_loader = torch.utils.data.DataLoader(
            SJTUSentenceDataset(
                feature=config["feature_file"],
                caption_df=val_df,
                vocabulary=vocabulary,
                sentence_embedding=sentence_embedding,
                transform=scaler.transform,
            ),
            shuffle=False,
            collate_fn=collate_fn([0, 1]),
            **config["dataloader_args"]
        )
        return trainloader, val_loader, {
            "scaler": scaler, "inputdim": inputdim, 
            "train_key2refs": train_key2refs, "val_key2refs": val_key2refs}

    def _forward(self, model, batch, mode, **kwargs):
        assert mode in ("train", "validation", "eval")

        if mode == "eval":
            feats = batch[1]
            feat_lens = batch[-1]
        else:
            feats = batch[0]
            caps = batch[1]
            sent_embeds = batch[2]
            feat_lens = batch[-2]
            cap_lens = batch[-1]

        feats = convert_tensor(feats.float(),
                               device=self.device,
                               non_blocking=True)

        if mode == "eval":
            output = model(feats, feat_lens, **kwargs)
        else:
            sent_embeds = convert_tensor(sent_embeds.float(),
                                         device=self.device,
                                         non_blocking=True)
            caps = convert_tensor(caps.long(),
                                  device=self.device,
                                  non_blocking=True)
            # pack labels to remove padding from caption labels
            word_targets = torch.nn.utils.rnn.pack_padded_sequence(
                caps[:, 1:], cap_lens - 1, batch_first=True).data

            if mode == "train":
                output = model(feats, feat_lens, caps, cap_lens, **kwargs)
            else:
                output = model(feats, feat_lens, **kwargs)

            packed_logits = torch.nn.utils.rnn.pack_padded_sequence(
                output["logits"], cap_lens - 1, batch_first=True).data
            packed_logits = convert_tensor(
                packed_logits, device=self.device, non_blocking=True)

            output["packed_logits"] = packed_logits
            output["word_targets"] = word_targets
            output["sentence_targets"] = sent_embeds

        return output

    def train(self, config, **kwargs):
        """Trains a model on the given configurations.
        :param config: A training configuration. Note that all parameters in the config can also be manually adjusted with --ARG=VALUE
        :param **kwargs: parameters to overwrite yaml config
        """

        from pycocoevalcap.cider.cider import Cider

        conf = train_util.parse_config_or_kwargs(config, **kwargs)
        conf["seed"] = self.seed
        outputdir = os.path.join(
            conf["outputpath"], conf["model"],
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
            score_function=lambda engine: engine.state.metrics["score"],
            score_name="loss")

        logger = train_util.genlogger(os.path.join(outputdir, "train.log"))
        # print passed config parameters
        logger.info("Storing files in: {}".format(outputdir))
        train_util.pprint_dict(conf, logger.info)

        zh = conf["zh"]
        vocabulary = torch.load(conf["vocab_file"])
        train_loader, val_loader, info = self._get_dataloaders(conf, vocabulary)
        conf["inputdim"] = info["inputdim"]
        val_key2refs = info["val_key2refs"]
        logger.info("<== Estimating Scaler ({}) ==>".format(info["scaler"].__class__.__name__))
        logger.info(
            "Feature: {} Input dimension: {} Vocab Size: {}".format(
                conf["feature_file"], info["inputdim"], len(vocabulary)))

        model = self._get_model(conf, len(vocabulary))
        model = model.to(self.device)
        train_util.pprint_dict(model, logger.info, formatter="pretty")
        optimizer = getattr(
            torch.optim, conf["optimizer"]
        )(model.parameters(), **conf["optimizer_args"])
        train_util.pprint_dict(optimizer, logger.info, formatter="pretty")


        XE_criterion = torch.nn.CrossEntropyLoss().to(self.device)
        seq_criterion = torch.nn.CosineEmbeddingLoss().to(self.device)
        crtrn_imprvd = train_util.criterion_improver(conf['improvecriterion'])

        def _train_batch(engine, batch):
            model.train()
            with torch.enable_grad():
                optimizer.zero_grad()
                output = self._forward(
                    model, batch, "train", ss_ratio=conf["ss_args"]["ss_ratio"])
                XE_loss = XE_criterion(output["packed_logits"], output["word_targets"])
                seq_loss = seq_criterion(output["seq_outputs"], output["sentence_targets"], torch.ones(batch[0].shape[0]).to(self.device))
                loss = XE_loss + seq_loss * conf["seq_loss_ratio"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                output["XE_loss"] = XE_loss.item()
                output["seq_loss"] = seq_loss.item()
                output["loss"] = loss.item()
                return output

        trainer = Engine(_train_batch)
        RunningAverage(output_transform=lambda x: x["loss"]).attach(trainer, "running_loss")
        pbar = ProgressBar(persist=False, ascii=True, ncols=100)
        pbar.attach(trainer, ["running_loss"])

        key2pred = {}

        def _inference(engine, batch):
            model.eval()
            keys = batch[3]
            with torch.no_grad():
                output = self._forward(model, batch, "validation")
                output["seq_loss"] = seq_criterion(output["seq_outputs"], output["sentence_targets"], torch.ones(len(keys)).to(self.device))
                seqs = output["seqs"].cpu().numpy()
                for (idx, seq) in enumerate(seqs):
                    if keys[idx] in key2pred:
                        continue
                    candidate = self._convert_idx2sentence(seq, vocabulary, zh)
                    key2pred[keys[idx]] = [candidate,]
                return output

        metrics = {
            "loss": Average(output_transform=lambda x: x["loss"]),
            "XE_loss": Average(output_transform=lambda x: x["XE_loss"]),
            "seq_loss": Average(output_transform=lambda x: x["seq_loss"]),
        }

        evaluator = Engine(_inference)

        def eval_val(engine, key2pred, key2refs):
            scorer = Cider(zh=zh)
            score, scores = scorer.compute_score(key2refs, key2pred)
            engine.state.metrics["score"] = score
            key2pred.clear()

        evaluator.add_event_handler(
            Events.EPOCH_COMPLETED, eval_val, key2pred, val_key2refs)

        for name, metric in metrics.items():
            metric.attach(trainer, name)

        metrics["seq_loss"].attach(evaluator, "seq_loss")
            
        trainer.add_event_handler(
              Events.EPOCH_COMPLETED, train_util.log_results, evaluator, val_loader,
              logger.info, metrics.keys(), ["seq_loss", "score"])

        if conf["ss"]:
            trainer.add_event_handler(
                Events.GET_BATCH_COMPLETED, train_util.update_ss_ratio, conf, len(train_loader))


        evaluator.add_event_handler(
            Events.EPOCH_COMPLETED, train_util.save_model_on_improved, crtrn_imprvd,
            "score", {
                "model": model.state_dict(),
                "config": conf,
                "scaler": info["scaler"]
        }, os.path.join(outputdir, "saved.pth"))

        scheduler = getattr(torch.optim.lr_scheduler, conf["scheduler"])(
            optimizer, **conf["scheduler_args"])
        evaluator.add_event_handler(
            Events.EPOCH_COMPLETED, train_util.update_lr,
            scheduler, "score")

        evaluator.add_event_handler(
            Events.EPOCH_COMPLETED, checkpoint_handler, {
                "model": model,
            }
        )

        trainer.run(train_loader, max_epochs=conf["epochs"])
        return outputdir


if __name__ == "__main__":
    fire.Fire(Runner)
