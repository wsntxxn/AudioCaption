# coding=utf-8
#!/usr/bin/env python3
import os
import sys
import pickle
import json
import datetime
import uuid
from pathlib import Path

import fire
from ignite.metrics.accumulation import Average
import numpy as np
import torch
from ignite.engine.engine import Engine, Events
from ignite.metrics import RunningAverage
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import ProgressBar
from ignite.utils import convert_tensor
from torch.utils.tensorboard import SummaryWriter

import captioning.models
import captioning.models.encoder
import captioning.models.decoder
import captioning.losses.loss as losses
import captioning.metrics.metric as metrics
import captioning.utils.train_util as train_util
from captioning.utils.build_vocab import Vocabulary
from captioning.ignite_runners.base import BaseRunner
import captioning.datasets.caption_dataset as ac_dataset

class Runner(BaseRunner):

    def _get_dataloaders(self, config):
        augments = train_util.parse_augments(config["augments"])
        if config["distributed"]:
            config["dataloader_args"]["batch_size"] //= self.world_size

        data_config = config["data"]
        vocabulary = config["vocabulary"]
        if "train" not in data_config:
            raw_audio_to_h5 = train_util.load_dict_from_csv(data_config["raw_feat_csv"],
                                                            ("audio_id", "hdf5_path"))
            fc_audio_to_h5 = train_util.load_dict_from_csv(data_config["fc_feat_csv"],
                                                           ("audio_id", "hdf5_path"))
            attn_audio_to_h5 = train_util.load_dict_from_csv(data_config["attn_feat_csv"],
                                                             ("audio_id", "hdf5_path"))
            cap_id_to_condition = train_util.load_dict_from_csv(data_config["caption_condition"],
                                                                ("cap_id", "specificity"))
            caption_info = json.load(open(data_config["caption_file"], "r"))["audios"]
            val_size = int(len(caption_info) * (1 - data_config["train_percent"] / 100.))
            val_audio_idxs = np.random.choice(len(caption_info), val_size, replace=False)
            train_audio_idxs = [idx for idx in range(len(caption_info)) if idx not in val_audio_idxs]
            train_dataset = ac_dataset.CaptionConditionDataset(
                raw_audio_to_h5,
                fc_audio_to_h5,
                attn_audio_to_h5,
                caption_info,
                cap_id_to_condition,
                vocabulary,
                transform=augments
            )
            # TODO DistributedCaptionSampler
            # train_sampler = torch.utils.data.DistributedSampler(train_dataset) if config["distributed"] else None
            train_sampler = ac_dataset.CaptionSampler(train_dataset, train_audio_idxs, True, **config["sampler_args"])
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                collate_fn=ac_dataset.collate_fn([0, 2, 4], 4),
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
                output["raw_audio_to_h5"] = train_util.load_dict_from_csv(
                    conf_split["raw_feat_csv"], ("audio_id", "hdf5_path"))
                output["fc_audio_to_h5"] = train_util.load_dict_from_csv(
                    conf_split["fc_feat_csv"], ("audio_id", "hdf5_path"))
                output["attn_audio_to_h5"] = train_util.load_dict_from_csv(
                    conf_split["attn_feat_csv"], ("audio_id", "hdf5_path"))
                output["caption_info"] = json.load(open(conf_split["caption_file"], "r"))["audios"]
            cap_id_to_condition = train_util.load_dict_from_csv(data_config["train"]["caption_condition"],
                                                                ("cap_id", "specificity"))

            train_dataset = ac_dataset.CaptionConditionDataset(
                data["train"]["raw_audio_to_h5"],
                data["train"]["fc_audio_to_h5"],
                data["train"]["attn_audio_to_h5"],
                data["train"]["caption_info"],
                cap_id_to_condition,
                vocabulary,
                transform=augments
            )
            # TODO DistributedCaptionSampler
            # train_sampler = torch.utils.data.DistributedSampler(train_dataset) if config["distributed"] else None
            train_sampler = ac_dataset.CaptionSampler(train_dataset, shuffle=True, **config["sampler_args"])
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                collate_fn=ac_dataset.collate_fn([0, 2, 4], 4),
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
            "val_key2refs": val_key2refs,
            "specificity_norm": 2.0 / max(cap_id_to_condition.values()),
            "mean_condition": np.mean(list(cap_id_to_condition.values()))
        }

    @staticmethod
    def _get_model(config, outputfun=sys.stdout):
        vocabulary = pickle.load(open(config["data"]["vocab_file"], "rb"))
        encoder = getattr(
            captioning.models.encoder, config["encoder"])(
            config["data"]["raw_feat_dim"],
            config["data"]["fc_feat_dim"],
            config["data"]["attn_feat_dim"],
            **config["encoder_args"]
        )
        if "pretrained_encoder" in config:
            train_util.load_pretrained_model(encoder, 
                                             config["pretrained_encoder"],
                                             outputfun)

        decoder = getattr(
            captioning.models.decoder, config["decoder"])(
            vocab_size=len(vocabulary),
            **config["decoder_args"]
        )
        if "pretrained_word_embedding" in config:
            embeddings = np.load(config["pretrained_word_embedding"])
            decoder.load_word_embeddings(
                embeddings,
                freeze=config["freeze_word_embedding"]
            )
        if "pretrained_decoder" in config:
            train_util.load_pretrained_model(decoder,
                                             config["pretrained_decoder"],
                                             outputfun)
        model = getattr(
            captioning.models, config["model"])(
            encoder, decoder, **config["model_args"]
        )
        if "pretrained" in config:
            train_util.load_pretrained_model(model, 
                                             config["pretrained"],
                                             outputfun)
        return model

    def _forward(self, model, batch, mode, **kwargs):
        assert mode in ("train", "validation", "eval")

        if mode == "train":
            raw_feats = batch[0]
            fc_feats = batch[1]
            attn_feats = batch[2]
            conditions = batch[3]
            caps = batch[4]
            raw_feat_lens = batch[-3]
            attn_feat_lens = batch[-2]
            cap_lens = batch[-1]
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
            caps = convert_tensor(caps.long(),
                                  device=self.device,
                                  non_blocking=True)

            input_dict["caps"] = caps
            input_dict["cap_lens"] = cap_lens
            input_dict["conditions"] = conditions.to(self.device) * self.specificity_norm
            input_dict["ss_ratio"] = kwargs["ss_ratio"]
            output = model(input_dict)

            output["targets"] = caps[:, 1:]
            output["lens"] = torch.as_tensor(cap_lens - 1)
            output["conditions"] = conditions
        else:
            input_dict.update(kwargs)
            conditions = torch.empty(raw_feats.size(0)).fill_(kwargs["condition"]).to(self.device)
            input_dict["conditions"] = conditions * self.specificity_norm
            output = model(input_dict)

        return output

    def train(self, config, **kwargs):
        """Trains a model on the given configurations.
        :param config: A training configuration. Note that all parameters in the config can also be manually adjusted with --ARG=VALUE
        :param **kwargs: parameters to overwrite yaml config
        """
        from pycocoevalcap.cider.cider import Cider

        conf = train_util.parse_config_or_kwargs(config, **kwargs)
        conf["seed"] = self.seed
        
        #########################
        # Distributed training initialization
        #########################
        if conf["distributed"]:
            torch.distributed.init_process_group(backend="nccl")
            self.local_rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
            assert kwargs["local_rank"] == self.local_rank
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
            # self.group = torch.distributed.new_group()

        #########################
        # Create checkpoint directory
        #########################
        if not conf["distributed"] or not self.local_rank:
            outputdir = Path(conf["outputpath"]) / conf["model"] / \
                conf["remark"] / f"seed_{self.seed}"
                # "{}_{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%m"),
                               # uuid.uuid1().hex)

            outputdir.mkdir(parents=True, exist_ok=True)
            logger = train_util.genlogger(str(outputdir / "train.log"))
            # print passed config parameters
            if "SLURM_JOB_ID" in os.environ:
                logger.info(f"Slurm job id: {os.environ['SLURM_JOB_ID']}")
                logger.info(f"Slurm node: {os.environ['SLURM_JOB_NODELIST']}")
            logger.info(f"Storing files in: {outputdir}")
            train_util.pprint_dict(conf, logger.info)

        #########################
        # Create dataloaders
        #########################
        zh = conf["zh"]
        vocabulary = pickle.load(open(conf["data"]["vocab_file"], "rb"))
        conf["vocabulary"] = vocabulary
        dataloaders = self._get_dataloaders(conf)
        train_dataloader = dataloaders["train_dataloader"]
        val_dataloader = dataloaders["val_dataloader"]
        val_key2refs = dataloaders["val_key2refs"]
        specificity_norm = dataloaders["specificity_norm"]
        val_condition = dataloaders["mean_condition"]
        self.specificity_norm = specificity_norm
        conf["data"]["raw_feat_dim"] = train_dataloader.dataset.raw_feat_dim
        conf["data"]["fc_feat_dim"] = train_dataloader.dataset.fc_feat_dim
        conf["data"]["attn_feat_dim"] = train_dataloader.dataset.attn_feat_dim
        conf["data"]["specificity_norm"] = specificity_norm


        #########################
        # Initialize model
        #########################
        if not conf["distributed"] or not self.local_rank:
            model = self._get_model(conf, logger.info)
        else:
            model = self._get_model(conf)
        model = model.to(self.device)
        if conf["distributed"]:
            model = torch.nn.parallel.distributed.DistributedDataParallel(
                model, device_ids=[self.local_rank,], output_device=self.local_rank,
                find_unused_parameters=True)
        if not conf["distributed"] or not self.local_rank:
            train_util.pprint_dict(model, logger.info, formatter="pretty")
            num_params = 0
            for param in model.parameters():
                num_params += param.numel()
            logger.info(f"{num_params} parameters in total")

        #########################
        # Create loss function and saving criterion
        #########################
        optimizer = getattr(torch.optim, conf["optimizer"])(
            model.parameters(), **conf["optimizer_args"])

        loss_fn = getattr(losses, conf["loss"])(**conf["loss_args"])
        word_to_specificity = train_util.load_dict_from_csv(
            conf["data"]["word_specificity"], ("word", "specificity"))
        word_specificity = torch.zeros(len(vocabulary)).to(self.device)
        for word, specificity in word_to_specificity.items():
            word_specificity[vocabulary.word2idx[word]] = specificity
        loss_fn = losses.SpecificityLossWrapper(
            loss_fn, word_specificity, conf["sentence_reduce"], 0.0)

        if not conf["distributed"] or not self.local_rank:
            train_util.pprint_dict(optimizer, logger.info, formatter="pretty")
            crtrn_imprvd = train_util.criterion_improver(conf["improvecriterion"])

        #########################
        # Tensorboard record
        #########################
        if not conf["distributed"] or not self.local_rank:
            tensorboard_writer = SummaryWriter(outputdir / "run")

        #########################
        # Define training engine
        #########################
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
                loss, word_loss, condition_loss = loss_fn(output)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), conf["max_grad_norm"])
                optimizer.step()
                output["loss"] = loss.item()
                output["word_loss"] = word_loss.item()
                output["condition_loss"] = condition_loss.item()
                if not conf["distributed"] or not self.local_rank:
                    tensorboard_writer.add_scalar(
                        "loss/train", loss.item(), engine.state.iteration)
                    tensorboard_writer.add_scalar(
                        "word_loss/train", word_loss.item(), engine.state.iteration)
                    tensorboard_writer.add_scalar(
                        "condition_loss/train", condition_loss.item(), engine.state.iteration)

                return output

        trainer = Engine(_train_batch)
        RunningAverage(output_transform=lambda x: x["loss"]).attach(trainer, "running_loss")
        pbar = ProgressBar(persist=False, ascii=True, ncols=100)
        pbar.attach(trainer, ["running_loss"])
        train_metrics = {
            "loss": Average(lambda x: x["loss"]),
            "word_loss": Average(lambda x: x["word_loss"]),
            "condition_loss": Average(lambda x: x["condition_loss"]),
            "accuracy": metrics.Accuracy(),
        }
        for name, metric in train_metrics.items():
            metric.attach(trainer, name)
        # scheduled sampling
        if conf["ss"]:
            @trainer.on(Events.ITERATION_STARTED)
            def update_ss_ratio(engine):
                num_iter = len(train_dataloader)
                num_epoch = conf["epochs"]
                mode = conf["ss_args"]["ss_mode"]
                if mode == "exponential":
                    conf["ss_args"]["ss_ratio"] *= 0.01 ** (1.0 / num_epoch / num_iter)
                elif mode == "linear":
                    conf["ss_args"]["ss_ratio"] -= (1.0 - conf["ss_args"]["final_ss_ratio"]) / num_epoch / num_iter
        # stochastic weight averaging
        if conf["swa"]:
            swa_model = torch.optim.swa_utils.AveragedModel(model)
            @trainer.on(Events.EPOCH_COMPLETED)
            def update_swa(engine):
                if engine.state.epoch >= conf["swa_start"]:
                    swa_model.update_parameters(model)

        # set condition alpha when epoch > condition_start
        @trainer.on(Events.EPOCH_STARTED)
        def set_condition_alpha(engine):
            if engine.state.epoch > conf["condition_start"]:
                loss_fn.alpha = conf["condition_alpha"]

        #########################
        # Define inference engine
        #########################
        key2pred = {}

        def _inference(engine, batch):
            model.eval()
            keys = batch[0]
            with torch.no_grad():
                output = self._forward(model, batch, "validation",
                                       sample_method="beam", beam_size=3,
                                       condition=val_condition)
                seqs = output["seqs"].cpu().numpy()
                for (idx, seq) in enumerate(seqs):
                    candidate = self._convert_idx2sentence(seq, vocabulary.idx2word)
                    key2pred[keys[idx]] = [candidate,]
                return output

        evaluator = Engine(_inference)

        @evaluator.on(Events.EPOCH_COMPLETED)
        def eval_val(engine):
            scorer = Cider()
            score_output = self._eval_prediction(val_key2refs, key2pred, [scorer])
            engine.state.metrics["score"] = score_output["CIDEr"]
            key2pred.clear()

        pbar.attach(evaluator)

        #########################
        # Create learning rate scheduler
        #########################
        try:
            scheduler = getattr(torch.optim.lr_scheduler, conf["scheduler"])(
                optimizer, **conf["scheduler_args"])
        except AttributeError:
            import captioning.utils.lr_scheduler as lr_scheduler
            if conf["scheduler"] == "ExponentialDecayScheduler":
                conf["scheduler_args"]["total_iters"] = len(train_dataloader) * conf["epochs"]
            if "warmup_iters" not in conf["scheduler_args"]:
                warmup_iters = len(train_dataloader) * conf["epochs"] // 5
                conf["scheduler_args"]["warmup_iters"] = warmup_iters
            if not conf["distributed"] or not self.local_rank:
                logger.info(f"Warm up iterations: {conf['scheduler_args']['warmup_iters']}")
            scheduler = getattr(lr_scheduler, conf["scheduler"])(
                optimizer, **conf["scheduler_args"])

        def update_lr(engine, metric=None):
            if scheduler.__class__.__name__ == "ReduceLROnPlateau":
                assert metric is not None, "need validation metric for ReduceLROnPlateau"
                val_result = engine.state.metrics[metric]
                scheduler.step(val_result)
            else:
                scheduler.step()

        if scheduler.__class__.__name__ in ["StepLR", "ReduceLROnPlateau", "ExponentialLR", "MultiStepLR"]:
            evaluator.add_event_handler(Events.EPOCH_COMPLETED, update_lr, "score")
        else:
            trainer.add_event_handler(Events.ITERATION_STARTED, update_lr)

        #########################
        # Events for main process: mostly logging and saving
        #########################


        if not conf["distributed"] or not self.local_rank:
            # logging training and validation loss and metrics
            @trainer.on(Events.EPOCH_COMPLETED)
            def log_results(engine):
                train_results = engine.state.metrics
                evaluator.run(val_dataloader)
                val_results = evaluator.state.metrics
                output_str_list = [
                    "Validation Results - Epoch : {:<4}".format(engine.state.epoch)
                ]
                for metric in train_metrics:
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
                    tensorboard_writer.add_scalar(f"{metric}/val", output, engine.state.epoch)
                lr = optimizer.param_groups[0]["lr"]
                output_str_list.append(f"lr {lr:5<.2g} ")

                logger.info(" ".join(output_str_list))
            # saving best model
            @evaluator.on(Events.EPOCH_COMPLETED)
            def save_model(engine):
                dump = {
                    "model": model.state_dict() if not conf["distributed"] else model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": scheduler.state_dict(),
                    "vocabulary": vocabulary.idx2word
                }
                if crtrn_imprvd(engine.state.metrics["score"]):
                    torch.save(dump, outputdir / "best.pth")
                torch.save(dump, outputdir / "last.pth")
            # dump configuration
            train_util.store_yaml(conf, outputdir / "config.yaml")

        #########################
        # Start training
        #########################
        trainer.run(train_dataloader, max_epochs=conf["epochs"])

        # stochastic weight averaging
        if conf["swa"]:
            torch.save({
                "model": swa_model.module.state_dict(),
                "vocabulary": vocabulary.idx2word
            }, outputdir / "swa.pth")


        if not conf["distributed"] or not self.local_rank:
            return outputdir

    def predict(self,
                experiment_path: str,
                save_type: str = "best",
                raw_feat_csv: str = None,
                fc_feat_csv: str = None,
                attn_feat_csv: str = None,
                caption_output: str = "output.json",
                return_pred: bool=False,
                **kwargs):
        import pandas as pd

        """kwargs: {'max_length': int, 'method': str, 'beam_size': int}"""
        if "method" in kwargs and kwargs["method"] == "beam":
            del kwargs["method"]
            kwargs["sample_method"] = "beam"
        experiment_path = Path(experiment_path)
        checkpoint = f"{save_type}.pth"
        dump = torch.load(experiment_path / checkpoint, map_location="cpu")
        # Previous training config
        config = train_util.parse_config_or_kwargs(experiment_path / "config.yaml")

        self.specificity_norm = config["data"]["specificity_norm"]
        vocabulary = dump["vocabulary"]
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

        def _sample(engine, batch):
            with torch.no_grad():
                model.eval()
                keys = batch[0]
                output = self._forward(model, batch, "eval",
                                       condition=kwargs["specificity"],
                                       **kwargs)
                seqs = output["seqs"].cpu().numpy()

                for idx, seq in enumerate(seqs):
                    caption = self._convert_idx2sentence(seq, vocabulary)
                    key2pred[keys[idx]] = [caption,]

        pbar = ProgressBar(persist=False, ascii=True, ncols=100)
        sampler = Engine(_sample)
        pbar.attach(sampler)
        sampler.run(dataloader)

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

if __name__ == "__main__":
    fire.Fire(Runner)
