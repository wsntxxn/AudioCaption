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
from captioning.runners.base_runner import BaseRunner

class Runner(BaseRunner):

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
            caps = batch[3]
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
            # pack labels to remove padding from caption labels
            # targets = torch.nn.utils.rnn.pack_padded_sequence(
                    # caps[:, 1:], cap_lens - 1, batch_first=True).data

            input_dict["caps"] = caps
            input_dict["cap_lens"] = cap_lens
            input_dict["ss_ratio"] = kwargs["ss_ratio"]
            output = model(input_dict)

            # packed_logits = torch.nn.utils.rnn.pack_padded_sequence(
                # output["logits"], cap_lens - 1, batch_first=True).data
            # packed_logits = convert_tensor(
                # packed_logits, device=self.device, non_blocking=True)

            # output["packed_logits"] = packed_logits
            # output["targets"] = targets
            output["targets"] = caps[:, 1:]
            output["lens"] = torch.as_tensor(cap_lens - 1)
        else:
            input_dict.update(kwargs)
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
        dataloaders = self._get_dataloaders(conf)
        train_dataloader = dataloaders["train_dataloader"]
        val_dataloader = dataloaders["val_dataloader"]
        val_key2refs = dataloaders["val_key2refs"]
        conf["data"]["raw_feat_dim"] = train_dataloader.dataset.raw_feat_dim
        conf["data"]["fc_feat_dim"] = train_dataloader.dataset.fc_feat_dim
        conf["data"]["attn_feat_dim"] = train_dataloader.dataset.attn_feat_dim

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
        criterion = getattr(losses, conf["loss"])(**conf["loss_args"])
        # if conf["label_smoothing"]:
            # criterion = losses.LabelSmoothingLoss(len(vocabulary), smoothing=conf["smoothing"])
        # else:
            # criterion = torch.nn.CrossEntropyLoss().to(self.device)
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
                # loss = criterion(output["packed_logits"], output["targets"]).to(self.device)
                loss = criterion(output).to(self.device)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), conf["max_grad_norm"])
                optimizer.step()
                output["loss"] = loss.item()
                if not conf["distributed"] or not self.local_rank:
                    tensorboard_writer.add_scalar("loss/train", loss.item(), engine.state.iteration)
                return output

        trainer = Engine(_train_batch)
        RunningAverage(output_transform=lambda x: x["loss"]).attach(trainer, "running_loss")
        pbar = ProgressBar(persist=False, ascii=True, ncols=100)
        pbar.attach(trainer, ["running_loss"])
        train_metrics = {
            "loss": losses.Loss(criterion),
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


        #########################
        # Define inference engine
        #########################
        key2pred = {}

        def _inference(engine, batch):
            model.eval()
            keys = batch[0]
            with torch.no_grad():
                output = self._forward(model, batch, "validation", sample_method="beam", beam_size=3)
                # output = self._forward(model, batch, "validation")
                seqs = output["seqs"].cpu().numpy()
                for (idx, seq) in enumerate(seqs):
                    candidate = self._convert_idx2sentence(seq, vocabulary.idx2word, zh)
                    key2pred[keys[idx]] = [candidate,]
                return output

        evaluator = Engine(_inference)

        @evaluator.on(Events.EPOCH_COMPLETED)
        def eval_val(engine):
            scorer = Cider(zh=zh)
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
        elif scheduler.__class__.__name__ == "NoamScheduler":
            trainer.add_event_handler(Events.ITERATION_STARTED, update_lr)
        else:
            trainer.add_event_handler(Events.ITERATION_COMPLETED, update_lr)

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
                    torch.save(dump, outputdir / "saved.pth")
                torch.save(dump, outputdir / "last.pth")
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
            train_util.store_yaml(conf, outputdir / "config.yaml")

        #########################
        # Start training
        #########################
        trainer.run(train_dataloader, max_epochs=conf["epochs"])

        # stochastic weight averaging
        if conf["swa"]:
            torch.save({
                "model": swa_model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": scheduler.state_dict(),
                "vocabulary": vocabulary.idx2word
            }, outputdir / "swa.pth")

        if not conf["distributed"] or not self.local_rank:
            return outputdir


if __name__ == "__main__":
    fire.Fire(Runner)
