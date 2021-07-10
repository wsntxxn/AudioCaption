# coding=utf-8
#!/usr/bin/env python3
import os
import sys
import pickle
import datetime
import uuid
from pathlib import Path

import fire
import numpy as np
import torch
from ignite.engine.engine import Engine, Events
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import ProgressBar
from ignite.utils import convert_tensor

sys.path.append(os.getcwd())
import models
import utils.train_util as train_util
from utils.build_vocab import Vocabulary
from runners.base_runner import BaseRunner

class Runner(BaseRunner):

    @staticmethod
    def _get_model(config, vocab_size):
        if config["encodermodel"] == "E2EASREncoder":
            encodermodel = models.encoder.load_espnet_encoder(config["pretrained_encoder"], pretrained=config["load_encoder_params"])
        else:
            encodermodel = getattr(
                models.encoder, config["encodermodel"])(
                inputdim=config["input_dim"],
                **config["encodermodel_args"])
            if "pretrained_encoder" in config:
                encoder_state_dict = torch.load(
                    config["pretrained_encoder"],
                    map_location="cpu")
                if "model" in encoder_state_dict:
                    encoder_state_dict = encoder_state_dict["model"]
                encodermodel.load_state_dict(encoder_state_dict, strict=False)

        decodermodel = getattr(
            models.decoder, config["decodermodel"])(
            vocab_size=vocab_size,
            enc_mem_size=config["encodermodel_args"]["embed_size"],
            **config["decodermodel_args"])
        if "pretrained_decoder" in config:
            decoder_state_dict = torch.load(
                config["pretrained_decoder"],
                map_location="cpu")
            if "model" in decoder_state_dict:
                decoder_state_dict = decoder_state_dict["model"]
            decodermodel.load_state_dict(decoder_state_dict, strict=False)
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
            if Path(config["pretrained"]).exists():
                pretrained_state_dict = torch.load(config["pretrained"], map_location="cpu")
                if "model" in pretrained_state_dict:
                    pretrained_state_dict = pretrained_state_dict["model"]
                model.load_state_dict(pretrained_state_dict)
            else:
                print("Loading pretrained model {} failed, model is random initialized!".format(config["pretrained"]))
        return model

    def _forward(self, model, batch, mode, **kwargs):
        assert mode in ("train", "validation", "eval")

        if mode == "train":
            feats = batch[0]
            caps = batch[1]
            feat_lens = batch[-2]
            cap_lens = batch[-1]
        else:
            feats = batch[1]
            feat_lens = batch[-1]

        feats = convert_tensor(feats.float(),
                               device=self.device,
                               non_blocking=True)

        if mode == "train":
            caps = convert_tensor(caps.long(),
                                  device=self.device,
                                  non_blocking=True)
            # pack labels to remove padding from caption labels
            targets = torch.nn.utils.rnn.pack_padded_sequence(
                    caps[:, 1:], cap_lens - 1, batch_first=True).data

            output = model(feats, feat_lens, caps, cap_lens, **kwargs)

            packed_logits = torch.nn.utils.rnn.pack_padded_sequence(
                output["logits"], cap_lens - 1, batch_first=True).data
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
        from pycocoevalcap.cider.cider import Cider

        conf = train_util.parse_config_or_kwargs(config, **kwargs)
        conf["seed"] = self.seed
        
        assert "distributed" in conf

        if conf["distributed"]:
            torch.distributed.init_process_group(backend="nccl")
            self.local_rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
            assert kwargs["local_rank"] == self.local_rank
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
            # self.group = torch.distributed.new_group()

        if not conf["distributed"] or not self.local_rank:
            outputdir = str(
                Path(conf["outputpath"]) / 
                conf["model"] /
                # "{}_{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%m"),
                               # uuid.uuid1().hex)
                conf["remark"] /
                "seed_{}".format(self.seed)
            )

            Path(outputdir).mkdir(parents=True, exist_ok=True)
            # # Early init because of creating dir
            # checkpoint_handler = ModelCheckpoint(
                # outputdir,
                # "run",
                # n_saved=1,
                # require_empty=False,
                # create_dir=False,
                # score_function=lambda engine: engine.state.metrics["score"],
                # score_name="score")

            logger = train_util.genlogger(str(Path(outputdir) / "train.log"))
            # print passed config parameters
            if "SLURM_JOB_ID" in os.environ:
                logger.info("Slurm job id: {}".format(os.environ["SLURM_JOB_ID"]))
            logger.info("Storing files in: {}".format(outputdir))
            train_util.pprint_dict(conf, logger.info)

        zh = conf["zh"]
        vocabulary = pickle.load(open(conf["vocab_file"], "rb"))
        dataloaders = self._get_dataloaders(conf, vocabulary)
        train_dataloader = dataloaders["train_dataloader"]
        val_dataloader = dataloaders["val_dataloader"]
        val_key2refs = dataloaders["val_key2refs"]
        data_dim = train_dataloader.dataset.data_dim
        conf["input_dim"] = data_dim
        if not conf["distributed"] or not self.local_rank:
            feature_data = conf["h5_csv"] if "h5_csv" in conf else conf["train_h5_csv"]
            logger.info(
                "Feature: {} Input dimension: {} Vocab Size: {}".format(
                    feature_data, data_dim, len(vocabulary)))

        model = self._get_model(conf, len(vocabulary))
        model = model.to(self.device)
        if conf["distributed"]:
            model = torch.nn.parallel.distributed.DistributedDataParallel(
                model, device_ids=[self.local_rank,], output_device=self.local_rank,
                find_unused_parameters=True)
        optimizer = getattr(
            torch.optim, conf["optimizer"]
        )(model.parameters(), **conf["optimizer_args"])

        if not conf["distributed"] or not self.local_rank:
            train_util.pprint_dict(model, logger.info, formatter="pretty")
            train_util.pprint_dict(optimizer, logger.info, formatter="pretty")

        if conf["label_smoothing"]:
            criterion = train_util.LabelSmoothingLoss(len(vocabulary), smoothing=conf["smoothing"])
        else:
            criterion = torch.nn.CrossEntropyLoss().to(self.device)
        crtrn_imprvd = train_util.criterion_improver(conf['improvecriterion'])

        def _train_batch(engine, batch):
            if conf["distributed"]:
                train_dataloader.sampler.set_epoch(engine.state.epoch)
            model.train()
            with torch.enable_grad():
                optimizer.zero_grad()
                output = self._forward(
                    model, batch, "train",
                    ss_ratio=conf["ss_args"]["ss_ratio"]
                )
                loss = criterion(output["packed_logits"], output["targets"]).to(self.device)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), conf["max_grad_norm"])
                optimizer.step()
                output["loss"] = loss.item()
                return output

        trainer = Engine(_train_batch)
        RunningAverage(output_transform=lambda x: x["loss"]).attach(trainer, "running_loss")
        pbar = ProgressBar(persist=False, ascii=True, ncols=100)
        pbar.attach(trainer, ["running_loss"])

        key2pred = {}

        def _inference(engine, batch):
            model.eval()
            keys = batch[0]
            with torch.no_grad():
                output = self._forward(model, batch, "validation")
                seqs = output["seqs"].cpu().numpy()
                for (idx, seq) in enumerate(seqs):
                    candidate = self._convert_idx2sentence(seq, vocabulary, zh)
                    key2pred[keys[idx]] = [candidate,]
                return output

        metrics = {
            "loss": Loss(criterion, output_transform=lambda x: (x["packed_logits"], x["targets"])),
            "accuracy": Accuracy(output_transform=lambda x: (x["packed_logits"], x["targets"])),
        }
        for name, metric in metrics.items():
            metric.attach(trainer, name)

        evaluator = Engine(_inference)

        def eval_val(engine, key2pred, key2refs):
            scorer = Cider(zh=zh)
            score_output = self._eval_prediction(key2refs, key2pred, [scorer])
            engine.state.metrics["score"] = score_output["CIDEr"]
            key2pred.clear()

        evaluator.add_event_handler(
            Events.EPOCH_COMPLETED, eval_val, key2pred, val_key2refs)

        pbar.attach(evaluator)

        # Learning rate scheduler
        if "scheduler" in conf:
            try:
                scheduler = getattr(torch.optim.lr_scheduler, conf["scheduler"])(
                    optimizer, **conf["scheduler_args"])
            except AttributeError:
                import utils.lr_scheduler
                if conf["scheduler"] == "ExponentialDecayScheduler":
                    conf["scheduler_args"]["total_iters"] = len(train_dataloader) * conf["epochs"]
                scheduler = getattr(utils.lr_scheduler, conf["scheduler"])(
                    optimizer, **conf["scheduler_args"])
            if scheduler.__class__.__name__ in ["StepLR", "ReduceLROnPlateau", "ExponentialLR", "MultiStepLR"]:
                evaluator.add_event_handler(
                    Events.EPOCH_COMPLETED, train_util.update_lr,
                    scheduler, "score")
            else:
                trainer.add_event_handler(
                    Events.ITERATION_COMPLETED, train_util.update_lr, scheduler, None)
        
        # Scheduled sampling
        if conf["ss"]:
            trainer.add_event_handler(
                Events.GET_BATCH_COMPLETED, train_util.update_ss_ratio, conf, len(train_dataloader))

        #########################
        # Events for main process: mostly logging and saving
        #########################
        if not conf["distributed"] or not self.local_rank:
            # logging training and validation loss and metrics
            trainer.add_event_handler(
                Events.EPOCH_COMPLETED, train_util.log_results, optimizer, evaluator, val_dataloader,
                logger.info, metrics.keys(), ["score"])
            # saving best model
            evaluator.add_event_handler(
                Events.EPOCH_COMPLETED, train_util.save_model_on_improved, crtrn_imprvd,
                "score", {
                    "model": model.state_dict() if not conf["distributed"] else model.module.state_dict(),
                    # "config": conf,
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": scheduler.state_dict()
                }, str(Path(outputdir) / "saved.pth")
            )
            # regular checkpoint
            checkpoint_handler = ModelCheckpoint(
                outputdir,
                "run",
                n_saved=1,
                require_empty=False,
                create_dir=False,
                score_function=lambda engine: engine.state.metrics["score"],
                score_name="score")
            evaluator.add_event_handler(
                Events.EPOCH_COMPLETED, checkpoint_handler, {
                    "model": model,
                }
            )
            # dump configuration
            train_util.store_yaml(conf, str(Path(outputdir) / "config.yaml"))

        #########################
        # Start training
        #########################
        trainer.run(train_dataloader, max_epochs=conf["epochs"])
        if not conf["distributed"] or not self.local_rank:
            return outputdir


if __name__ == "__main__":
    fire.Fire(Runner)
