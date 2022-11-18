# coding=utf-8
#!/usr/bin/env python3
import os
import sys
from pathlib import Path

import fire
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

import captioning.models
import captioning.models.encoder
import captioning.models.decoder
import captioning.losses.loss as losses
import captioning.utils.train_util as train_util
from captioning.utils.build_vocab import Vocabulary
from captioning.pytorch_runners.base import BaseRunner
from pycocoevalcap.cider.cider import Cider


class Runner(BaseRunner):

    def _get_model(self, print_fn=sys.stdout.write):
        encoder_cfg = self.config["model"]["encoder"]
        encoder = train_util.init_obj(
            captioning.models.encoder,
            encoder_cfg
        )
        if "pretrained" in encoder_cfg:
            pretrained = encoder_cfg["pretrained"]
            train_util.load_pretrained_model(encoder,
                                             pretrained,
                                             print_fn)
        decoder_cfg = self.config["model"]["decoder"]
        if "vocab_size" not in decoder_cfg["args"]:
            decoder_cfg["args"]["vocab_size"] = len(self.vocabulary)
        decoder = train_util.init_obj(
            captioning.models.decoder,
            decoder_cfg
        )
        if "word_embedding" in decoder_cfg:
            decoder.load_word_embedding(**decoder_cfg["word_embedding"])
        if "pretrained" in decoder_cfg:
            pretrained = decoder_cfg["pretrained"]
            train_util.load_pretrained_model(decoder,
                                             pretrained,
                                             print_fn)
        model = train_util.init_obj(captioning.models, self.config["model"],
            encoder=encoder, decoder=decoder)
        return model

    def _forward(self, batch, training=True):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                if k == "cap":
                    batch[k] = v.long().to(self.device)
                else:
                    batch[k] = v.float().to(self.device)

        input_dict = {
            "mode": "train" if training else "inference",
        }
        input_dict.update(batch)

        if training:
            input_dict["ss_ratio"] = self.ss_ratio
            if "specaug" in self.config:
                input_dict["specaug"] = self.config["specaug"]
            output = self.model(input_dict)
            output["tgt"] = batch["cap"][:, 1:]
            output["tgt_len"] = torch.as_tensor(batch["cap_len"] - 1)
        else:
            input_dict["specaug"] = False
            input_dict.update(self.config["inference_args"])
            output = self.model(input_dict)

        return output

    def _update_ss_ratio(self):
        if not self.config["scheduled_sampling"]["use"]:
            return
        ss_cfg = self.config["scheduled_sampling"]
        total_iters = self.iterations
        if ss_cfg["mode"] == "exponential":
            self.ss_ratio *= 0.01 ** (1.0 / total_iters)
        elif ss_cfg["mode"] == "linear":
            self.ss_ratio -= (1.0 - ss_cfg["final_ratio"]) / total_iters
        else:
            raise Exception(f"mode {ss_cfg['mode']} not supported")
    
    def _train_epoch(self):
        total_loss, nsamples = 0, 0
        self.model.train()

        for iteration in trange(self.epoch_length, ascii=True, ncols=100,
                                desc=f"Epoch {self.epoch}/{self.epochs}"):

            try:
                batch = next(self.train_iter)
            except StopIteration:
                self.train_iter = iter(self.train_dataloader)
                batch = next(self.train_iter)

            #####################################################################
            # Update scheduled sampling ratio
            #####################################################################
            self._update_ss_ratio()
            self.tb_writer.add_scalar("scheduled_sampling_prob",
                self.ss_ratio, self.iteration)

            #####################################################################
            # Update learning rate
            #####################################################################
            if self.lr_update_interval == "iteration":
                self.lr_scheduler.step()
                self.tb_writer.add_scalar("lr",
                    self.optimizer.param_groups[0]["lr"], self.iteration)

            #####################################################################
            # Forward and backward
            #####################################################################
            self.optimizer.zero_grad()
            output = self._forward(batch, training=True)
            loss = self.loss_fn(output)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            #####################################################################
            # Write the loss summary
            #####################################################################
            self.tb_writer.add_scalar("loss/train", loss.item(), self.iteration)
            cap_len = batch["cap_len"]
            nsample = sum(cap_len - 1)
            total_loss += loss.item() * nsample
            nsamples += nsample
            
            self.iteration += 1

        return {
            "loss": total_loss / nsamples
        }

    def _eval_epoch(self):
        key2pred = self._inference(self.val_dataloader)
        scorer = Cider()
        result = self._eval_prediction(self.val_key2refs, key2pred, [scorer])
        result = result[scorer.method()]
        return { "score": result }

    def save_checkpoint(self, ckpt_path):
        model_dict = self.model.state_dict()
        ckpt = {
            "model": { k: model_dict[k] for k in self.saving_keys },
            "epoch": self.epoch,
            "metric_monitor": self.metric_monitor.state_dict(),
            "not_improve_cnt": self.not_improve_cnt,
            "vocabulary": self.vocabulary.idx2word
        }
        if self.include_optim_in_ckpt:
            ckpt["optimizer"] = self.optimizer.state_dict()
            ckpt["lr_scheduler"] = self.lr_scheduler.state_dict()
        torch.save(ckpt, ckpt_path)

    def resume_checkpoint(self, finetune=False):
        ckpt = torch.load(self.config["resume"], "cpu")
        train_util.load_pretrained_model(self.model, ckpt)
        if not hasattr(self, "vocabulary"):
            self.vocabulary = ckpt["vocabulary"]
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
    
    def train(self, config, **kwargs):
        """Trains a model on the given configurations.
        :param config: A training configuration. Note that all parameters in the config can also be manually adjusted with --ARG=VALUE
        :param **kwargs: parameters to overwrite yaml config
        """

        self.config = train_util.parse_config_or_kwargs(config, **kwargs)
        self.config["seed"] = self.seed

        #####################################################################
        # Create checkpoint directory
        #####################################################################
        exp_dir = Path(self.config["outputpath"]) / \
            self.config["model"]["type"] / self.config["remark"] / \
            f"seed_{self.seed}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        self.logger = train_util.init_logger(str(exp_dir / "train.log"))
        # print passed config parameters
        if "SLURM_JOB_ID" in os.environ:
            self.logger.info(f"Slurm job id: {os.environ['SLURM_JOB_ID']}")
            self.logger.info(f"Slurm node: {os.environ['SLURM_JOB_NODELIST']}")
        self.logger.info(f"Storing files in: {exp_dir}")
        train_util.pprint_dict(self.config, self.logger.info)

        #####################################################################
        # Create dataloaders
        #####################################################################
        dataloaders = self._get_dataloaders()
        self.train_dataloader = dataloaders["dataloader"]["train"]
        self.val_dataloader = dataloaders["dataloader"]["val"]
        self.val_key2refs = dataloaders["key2refs"]["val"]
        self.logger.info(f"the training dataset has "
            f"{len(self.train_dataloader.dataset)} samples")
        self.vocabulary = self.train_dataloader.dataset.vocabulary
        
        self.__dict__.update(self.config["trainer"])

        if not hasattr(self, "epoch_length"):
            self.epoch_length = len(self.train_dataloader)
        self.iterations = self.epoch_length * self.epochs

        #####################################################################
        # Build model
        #####################################################################
        model = self._get_model(self.logger.info)
        self.model = model.to(self.device)
        swa_model = train_util.AveragedModel(model)
        train_util.pprint_dict(self.model, self.logger.info, formatter="pretty")
        num_params = 0
        num_trainable_params = 0
        self.saving_keys = []
        for name, param in model.named_parameters():
            num_params += param.numel()
            if param.requires_grad:
                num_trainable_params += param.numel()
                self.saving_keys.append(name)
        for name, buffer in model.named_buffers():
            self.saving_keys.append(name)

        self.logger.info(f"{num_params} parameters in total")
        self.logger.info(f"{num_trainable_params} trainable parameters in total")

        #####################################################################
        # Build loss function and optimizer
        #####################################################################
        self.optimizer = train_util.init_obj(torch.optim,
            self.config["optimizer"], params=self.model.parameters())
        self.loss_fn = train_util.init_obj(losses, self.config["loss"])
        train_util.pprint_dict(self.optimizer, self.logger.info,
                               formatter="pretty")

        #####################################################################
        # Tensorboard record
        #####################################################################
        self.tb_writer = SummaryWriter(exp_dir / "run")

        #####################################################################
        # Create learning rate scheduler
        #####################################################################
        try:
            self.lr_scheduler = train_util.init_obj(torch.optim.lr_scheduler,
                self.config["lr_scheduler"], optimizer=self.optimizer)
        except AttributeError:
            import captioning.utils.lr_scheduler as lr_scheduler
            lr_scheduler_config = self.config["lr_scheduler"]
            if lr_scheduler_config["type"] in [
                    "ExponentialDecayScheduler", "CosineWithWarmup"]:
                lr_scheduler_config["args"]["total_iters"] = self.iterations
            if "warmup_iters" not in lr_scheduler_config["args"]:
                warmup_iters = self.iterations // 5
                lr_scheduler_config["args"]["warmup_iters"] = warmup_iters
            else:
                warmup_iters = lr_scheduler_config["args"]["warmup_iters"]
            self.logger.info(f"Warm up iterations: {warmup_iters}")
            self.lr_scheduler = train_util.init_obj(lr_scheduler,
                lr_scheduler_config, optimizer=self.optimizer)

        if "inference_args" not in self.config:
            self.config["inference_args"] = {
                "sample_method": "beam",
                "beam_size": 3
            }

        #####################################################################
        # Dump configuration
        #####################################################################
        train_util.store_yaml(self.config, exp_dir / "config.yaml")

        #####################################################################
        # Start training
        #####################################################################
        
        metric_mode = self.monitor_metric["mode"]
        metric_name = self.monitor_metric["name"]
        self.metric_monitor = train_util.MetricImprover(metric_mode)

        self.ss_ratio = 1.0
        self.iteration = 1

        self.train_iter = iter(self.train_dataloader)

        self.not_improve_cnt = 0

        if not hasattr(self, "early_stop"):
            self.early_stop = self.epochs

        self.epoch = 1
        
        if "resume" in self.config:
            assert "finetune" in self.__dict__, "finetune not being set"
            self.resume_checkpoint(finetune=self.finetune)

        for _ in range(self.epochs):
            
            train_output = self._train_epoch()
            val_result = self._eval_epoch()
            val_score = val_result[metric_name]

            #####################################################################
            # Stochastic weight averaging
            #####################################################################
            if self.config["swa"]["use"]:
                if self.epoch >= self.config["swa"]["start"]:
                    swa_model.update_parameters(self.model)

            #####################################################################
            # Update learning rate
            #####################################################################
            if self.lr_update_interval == "epoch":
                if self.lr_scheduler.__class__.__name__ == "ReduceLROnPlateau":
                    self.lr_scheduler.step(val_score)
                else:
                    self.lr_scheduler.step()

            #####################################################################
            # Log results
            #####################################################################
            lr = self.optimizer.param_groups[0]["lr"]
            train_loss = train_output["loss"]
            output_str = f"epoch: {self.epoch}  train_loss: {train_loss:.2g}" \
                         f"  val_score: {val_score:.2g}  lr: {lr:.2g}"
            self.logger.info(output_str)
            self.tb_writer.add_scalar("score/val", val_score, self.epoch)

            #####################################################################
            # Save checkpoint
            #####################################################################

            if self.metric_monitor(val_score):
                self.not_improve_cnt = 0
                self.save_checkpoint(exp_dir / "best.pth")
            else:
                self.not_improve_cnt += 1

            if self.epoch % self.save_interval == 0:
                self.save_checkpoint(exp_dir / "last.pth")

            if self.not_improve_cnt == self.early_stop:
                break
            
            self.epoch += 1

        #####################################################################
        # Stochastic weight averaging
        #####################################################################
        if self.config["swa"]["use"]:
            model_dict = swa_model.module.state_dict()
            torch.save({
                "model": { k: model_dict[k] for k in self.saving_keys },
                "vocabulary": self.vocabulary.idx2word
            }, exp_dir / "swa.pth")

        return exp_dir


if __name__ == "__main__":
    fire.Fire(Runner)
