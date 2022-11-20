# coding=utf-8
#!/usr/bin/env python3
import os
import sys
import pickle
import json
from pathlib import Path

import fire
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

import captioning.models
import captioning.models.encoder
import captioning.models.decoder
import captioning.losses.loss as losses
import captioning.utils.train_util as train_util
from captioning.utils.build_vocab import Vocabulary
from captioning.pytorch_runners.run import Runner as BaseRunner
import captioning.datasets as dataset_module

class Runner(BaseRunner):

    def _get_augment_data(self):
        data_config = self.config["data"]["augmentation"]
        dataset = train_util.init_obj(dataset_module, data_config["dataset"])
        collate_fn = train_util.init_obj(dataset_module, data_config["collate_fn"])
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset, collate_fn=collate_fn,
            **data_config["dataloader_args"]
        )
        return dataloader

    def _update_aug_factor(self):
        self.aug_discount = self.iteration / self.iterations * \
            self.config["max_aug_discount"]

    def _train_epoch(self):
        loss_history = {"real": [], "aug": []}
        self.model.train()
        for iteration in trange(self.epoch_length, ascii=True, ncols=100,
                                desc=f"Epoch {self.epoch}/{self.epochs}"):

            try:
                batch = next(self.train_iter)
            except StopIteration:
                self.train_iter = iter(self.train_dataloader)
                batch = next(self.train_iter)

            try:
                aug_batch = next(self.aug_iter)
            except StopIteration:
                self.aug_iter = iter(self.aug_dataloader)
                aug_batch = next(self.aug_iter)

            #####################################################################
            # Update scheduled sampling ratio
            #####################################################################
            self._update_ss_ratio()
            self.tb_writer.add_scalar("scheduled_sampling_prob",
                self.ss_ratio, self.iteration)

            #####################################################################
            # Update augmentation discount factor
            #####################################################################
            self._update_aug_factor()
            self.tb_writer.add_scalar("augmentation_discount",
                self.aug_discount, iteration)

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
            loss_real = self.loss_fn(output)

            output_aug = self._forward(aug_batch, training=True)
            loss_aug = self.loss_fn(output_aug)
            
            loss = loss_real + loss_aug * self.aug_discount
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            #####################################################################
            # Write the loss summary
            #####################################################################
            self.tb_writer.add_scalar("loss/train", loss.item(), self.iteration)
            loss_history["real"].append(loss_real.item())
            loss_history["aug"].append(loss_aug.item())
            
            self.iteration += 1

        return {
            "loss_real": np.mean(loss_history["real"]),
            "loss_aug": np.mean(loss_history["aug"])
        }


    def train(self, config, **kwargs):
        from pycocoevalcap.cider.cider import Cider

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

        self.aug_dataloader = self._get_augment_data()
        self.logger.info(f"Augmentation data size: "
                         f"{len(self.aug_dataloader.dataset)}")
        self.aug_iter = iter(self.aug_dataloader)

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
            loss_real = train_output["loss_real"]
            loss_aug = train_output["loss_aug"]
            output_str = f"epoch: {self.epoch}  real_loss: {loss_real:.2g}" \
                         f"  aug_loss: {loss_aug:.2g}" \
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
