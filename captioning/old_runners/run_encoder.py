# coding=utf-8
#!/usr/bin/env python3
import os
import sys
import datetime
import random
import uuid
import re
from pprint import pformat
from contextlib import contextmanager

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
sys.path.append("/mnt/lustre/sjtu/home/xnx98/utils")
import models
import utils.train_util as train_util
from utils.build_vocab import Vocabulary
from datasets.SJTUDataSet import SJTUSentenceDataset, collate_fn
import utils.score_util as score_util
from runners.base_runner import BaseRunner


class Runner(BaseRunner):
    """Main class to run experiments"""
    def __init__(self, seed=1):
        super(Runner, self).__init__(seed)

    @staticmethod
    def _get_dataloaders(config, vocabulary):
        scaler = getattr(
            pre, config["scaler"])(
            **config["scaler_args"])
        inputdim = -1
        caption_df = pd.read_json(config["caption_file"])

        sentence_embedding = np.load(config["sentence_embedding"], allow_pickle=True)

        for batch in tqdm(
            torch.utils.data.DataLoader(
                SJTUSentenceDataset(
                    kaldi_stream=config["feature_stream"],
                    caption_df=caption_df,
                    vocabulary=vocabulary,
                    sentence_embedding=sentence_embedding,
                ),
                collate_fn=collate_fn([0, 1]),
                **config["dataloader_args"]
            ),
            ascii=True
        ):
            feat = batch[0]
            feat = feat.reshape(-1, feat.shape[-1])
            scaler.partial_fit(feat)
            inputdim = feat.shape[-1]
        assert inputdim > 0, "Reading inputstream failed"

        augments = train_util.parse_augments(config["augments"])
        train_df = caption_df.sample(frac=config["train_percent"] / 100.,
                                     random_state=0)
        trainloader = torch.utils.data.DataLoader(
            SJTUSentenceDataset(
                kaldi_stream=config["feature_stream"],
                caption_df=train_df,
                vocabulary=vocabulary,
                sentence_embedding=sentence_embedding,
                transform=[scaler.transform, augments]
            ),
            shuffle=True,
            collate_fn=collate_fn([0, 1]),
            **config["dataloader_args"]
        )

        cv_df = caption_df[~caption_df.index.isin(train_df.index)]
        cv_key2refs = cv_df.groupby("key")["tokens"].apply(list).to_dict()
        cvloader = torch.utils.data.DataLoader(
            SJTUSentenceDataset(
                kaldi_stream=config["feature_stream"],
                caption_df=cv_df,
                vocabulary=vocabulary,
                sentence_embedding=sentence_embedding,
                transform=scaler.transform,
            ),
            shuffle=False,
            collate_fn=collate_fn([0, 1]),
            **config["dataloader_args"]
        )
        return trainloader, cvloader, {"scaler": scaler, "inputdim": inputdim, "cv_key2refs": cv_key2refs}

    @staticmethod
    def _get_model(config, vocab_size):
        embed_size = config["model_args"]["embed_size"]
        encodermodel = getattr(
            models.encoder, config["encodermodel"])(
            inputdim=config["inputdim"],
            embed_size=embed_size,
            **config["encodermodel_args"])
        if "pretrained_encoder" in config:
            pretrained_encoder = torch.load(
                config["pretrained_encoder"],
                map_location="cpu")["model"]
            encodermodel.load_state_dict(pretrained_encoder.state_dict())

        model = encodermodel

        return model

    def _forward(self, model, batch, mode="train", **kwargs):
        assert mode in ("train", "eval")

        if mode == "eval":
            feats = batch[1]
            feat_lens = batch[-1]

            feats = convert_tensor(feats.float(),
                                   device=self.device,
                                   non_blocking=True)
            embeds, _ = model(feats, feat_lens)
            return {"embedding_output": embeds}


        feats = batch[0]
        sent_embeds = batch[2]
        keys = batch[3]
        feat_lens = batch[-2]
        feats = convert_tensor(feats.float(),
                               device=self.device,
                               non_blocking=True)
        sent_embeds = convert_tensor(sent_embeds.float(),
                                     device=self.device,
                                     non_blocking=True)
        output = {}

        embeds, _ = model(feats, feat_lens)
        output["embedding_output"] = embeds

        if "negative_sampling" in kwargs and kwargs["negative_sampling"]:
            batch_size = feats.shape[0]
            embedding_targets = torch.empty_like(sent_embeds).to(self.device)
            embedding_labels = torch.ones(batch_size).to(self.device)
            embedding_targets[:batch_size // 3, :] = sent_embeds[:batch_size // 3, :]
            for i in range(batch_size // 3, 2 * batch_size // 3):
                key = keys[i]
                neg_idx = np.random.choice(range(batch_size), 1)[0]
                while keys[neg_idx] == key:
                    neg_idx = np.random.choice(range(batch_size), 1)[0]
                embedding_targets[i, :] = sent_embeds[neg_idx, :]
            for i in range(2 * batch_size // 3, batch_size):
                key = keys[i]
                neg_idx = np.random.choice(range(batch_size), 1)[0]
                while keys[neg_idx] == key:
                    neg_idx = np.random.choice(range(batch_size), 1)[0]
                embedding_targets[i, :] = output["embedding_output"][neg_idx, :]
            embedding_labels[batch_size // 3:] = -1
            output["embedding_labels"] = embedding_labels.to(self.device)
            output["embedding_targets"] = embedding_targets
        else:
            output["embedding_targets"] = sent_embeds

        return output

    def train(self, config, **kwargs):
        """Trains a model on the given configurations.
        :param config: A training configuration. Note that all parameters in the config can also be manually adjusted with --ARG=VALUE
        :param **kwargs: parameters to overwrite yaml config
        """

        config_parameters = train_util.parse_config_or_kwargs(config, **kwargs)
        outputdir = os.path.join(
            config_parameters["outputpath"], config_parameters["encodermodel"],
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
        trainloader, cvloader, info = self._get_dataloaders(config_parameters, vocabulary)
        cv_key2refs = info["cv_key2refs"]
        config_parameters["inputdim"] = info["inputdim"]
        logger.info("<== Estimating Scaler ({}) ==>".format(info["scaler"].__class__.__name__))
        logger.info(
            "Stream: {} Input dimension: {} Vocab Size: {}".format(
                config_parameters["feature_stream"], info["inputdim"], len(vocabulary)))

        model = self._get_model(config_parameters, len(vocabulary))
        if "pretrained_word_embedding" in config_parameters:
            embeddings = np.load(config_parameters["pretrained_word_embedding"])
            model.load_word_embeddings(embeddings, tune=config_parameters["tune_word_embedding"], projection=True)
        model = model.to(self.device)
        train_util.pprint_dict(model, logger.info, formatter="pretty")
        optimizer = getattr(
            torch.optim, config_parameters["optimizer"]
        )(model.parameters(), **config_parameters["optimizer_args"])
        train_util.pprint_dict(optimizer, logger.info, formatter="pretty")

        criterion = getattr(torch.nn, config_parameters["train_loss"])(**config_parameters["train_loss_args"]).to(self.device)
        crtrn_imprvd = train_util.criterion_improver(config_parameters['improvecriterion'])
        tf_ratio = config_parameters["teacher_forcing_ratio"]

        def _train_batch(engine, batch):
            model.train()
            tf = True if random.random() < tf_ratio else False
            with torch.enable_grad():
                optimizer.zero_grad()
                if config_parameters["train_loss"] == "CosineEmbeddingLoss":
                    output = self._forward(model, batch, negative_sampling=True)
                    loss = criterion(
                        output["embedding_output"], 
                        output["embedding_targets"], 
                        output["embedding_labels"]
                    )
                else:
                    output = self._forward(model, batch)
                    loss = criterion(output["embedding_output"], output["embedding_targets"]).to(self.device)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                output["loss"] = loss.item()
                return output

        trainer = Engine(_train_batch)
        RunningAverage(output_transform=lambda x: x["loss"]).attach(trainer, "running_loss")
        pbar = ProgressBar(persist=False, ascii=True)
        pbar.attach(trainer, ["running_loss"])

        # key2pred = {}

        def _inference(engine, batch):
            model.eval()
            # keys = batch[3]
            with torch.no_grad():
                output = self._forward(model, batch, tf=config_parameters["teacher_forcing_on_validation"], negative_sampling=True)
                if config_parameters["train_loss"] == "CosineEmbeddingLoss":
                    output["loss"] = criterion(
                        output["embedding_output"], 
                        output["embedding_targets"], 
                        output["embedding_labels"]
                    )
                else:
                    output["loss"] = criterion(output["embedding_output"], output["embedding_targets"]).item()
                return output

        metrics = {
            "loss": Average(output_transform=lambda x: x["loss"]),
        }

        evaluator = Engine(_inference)

        for name, metric in metrics.items():
            metric.attach(trainer, name)
            metric.attach(evaluator, name)

        trainer.add_event_handler(
              Events.EPOCH_COMPLETED, train_util.log_results, evaluator, cvloader,
              logger.info, metrics.keys(), metrics.keys())

        evaluator.add_event_handler(
            Events.EPOCH_COMPLETED, train_util.save_model_on_improved, crtrn_imprvd,
            "loss", {
                "model": model,
                "config": config_parameters,
                "scaler": info["scaler"]
        }, os.path.join(outputdir, "saved.pth"))

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **config_parameters["scheduler_args"])

        # evaluator.add_event_handler(
        # Events.EPOCH_COMPLETED, train_util.update_reduce_on_plateau,
        # scheduler, "score")

        # evaluator.add_event_handler(
        # Events.EPOCH_COMPLETED, checkpoint_handler, {
        # "model": model,
        # }
        # )

        early_stop_handler = EarlyStopping(
            patience=config_parameters["early_stop"],
            score_function=lambda engine: -engine.state.metrics["loss"],
            trainer=trainer)
        evaluator.add_event_handler(Events.COMPLETED, early_stop_handler)

        trainer.run(trainloader, max_epochs=config_parameters["epochs"])
        return outputdir

    def evaluate(self,
                 experiment_path: str,
                 kaldi_stream: str,
                 caption_embedding_path=None,
                 embedding_output: str = "embedding.pkl",
                 result_output: str = "result.txt"):
        import pickle
        from datasets.SJTUDataSet import SJTUDatasetEval, collate_fn

        dump = torch.load(os.path.join(experiment_path, "saved.pth"),
                          map_location="cpu")
        model = dump["model"]
        # Some scaler (sklearn standardscaler)
        scaler = dump["scaler"]
        # Also load previous training config
        config = dump["config"]
        model = model.to(self.device)

        dataset = SJTUDatasetEval(
            kaldi_stream=kaldi_stream,
            transform=scaler.transform)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            collate_fn=collate_fn((1,)),
            batch_size=32,
            num_workers=0)

        model.eval()

        key2pred_embed = {}
        key2ref_embeds = np.load(caption_embedding_path, allow_pickle=True)

        def _predict(engine, batch):
            with torch.no_grad():
                model.eval()
                keys = batch[0]
                output = self._forward(model, batch, mode="eval")
                pred_embeddings = output["embedding_output"]
                for idx, key in enumerate(keys):
                    if key in key2pred_embed:
                        continue
                    key2pred_embed[key] = pred_embeddings[idx].cpu().numpy()

        predict_engine = Engine(_predict)
        predict_engine.run(dataloader)

        with open(os.path.join(experiment_path, embedding_output), "wb") as writer:
            pickle.dump(key2pred_embed, writer)

        losses = []
        criterion = getattr(torch.nn, config["train_loss"])()

        for key in key2pred_embed.keys():
            pred_embed = key2pred_embed[key].reshape(1, -1)
            ref_embeds = key2ref_embeds[key]
            for i in range(ref_embeds.shape[0]):
                if config["train_loss"] == "CosineEmbeddingLoss":
                    loss = criterion(
                        torch.as_tensor(pred_embed),
                        torch.as_tensor(ref_embeds[i].reshape(1, -1)),
                        torch.ones(1)
                    )
                else:
                    loss = criterion(
                        torch.as_tensor(pred_embed),
                        torch.as_tensor(ref_embeds[i].reshape(1, -1))
                    )
                losses.append(loss)

        with open(os.path.join(experiment_path, result_output), "w") as writer:
            writer.write("Loss on evaluation data: {:.3g}\n".format(np.mean(losses)))


if __name__ == "__main__":
    fire.Fire(Runner)
