# coding=utf-8
#!/usr/bin/env python3
import os
import sys
from pathlib import Path

import fire
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import wandb
from tqdm import tqdm, trange
from pycocoevalcap.cider.cider import Cider
import captioning.utils.train_util as train_util

from python_scripts.train_eval.base import BaseRunner


class Runner(BaseRunner):

    def _get_dataloaders(self):
        dataloaders = {}
        for split in ["train", "val"]:
            data_config = self.config["data"][split]
            dataset = train_util.init_obj_from_dict(data_config["dataset"])
            collate_fn = train_util.init_obj_from_dict(data_config["collate_fn"])
            if "batch_sampler" in data_config:
                batch_sampler = train_util.init_obj_from_dict(
                    data_config["batch_sampler"], dataset=dataset)
            else:
                batch_sampler = None
            dataloader = torch.utils.data.DataLoader(
                dataset=dataset, collate_fn=collate_fn,
                batch_sampler=batch_sampler, **data_config["dataloader_args"])
            dataloaders[split] = dataloader

        return dataloaders

    def save_checkpoint(self, ckpt_path):
        model_dict = self.model.state_dict()
        ckpt = {
            "model": { k: model_dict[k] for k in self.saving_keys },
            "epoch": self.epoch,
            "metric_monitor": self.metric_monitor.state_dict(),
            "not_improve_cnt": self.not_improve_cnt,
        }
        if self.include_optim_in_ckpt:
            ckpt["optimizer"] = self.optimizer.state_dict()
            ckpt["lr_scheduler"] = self.lr_scheduler.state_dict()
        torch.save(ckpt, ckpt_path)

    def _forward(self, batch, training=True):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.float().to(self.device)

        # student forward
        input_dict = batch.copy()

        if training:
            if "specaug" in self.config:
                input_dict["specaug"] = self.config["specaug"]

            # teacher forward
            with torch.no_grad():
                tchr_enc_output = self.teacher(batch["teacher_wav"])

            input_dict["tchr_output"] = tchr_enc_output
            output = self.model(input_dict)
        else:
            input_dict["specaug"] = False
            output = self.model(input_dict)

        return output

    def _get_teacher(self, print_fn=sys.stdout.write):
        sys.path.append("/mnt/fast/nobackup/scratch4weeks/xx00336/workspace/wavcaps/captioning")
        from models.bart_captioning import BartCaptionModel
        checkpoint_path = self.config["teacher"]
        ckpt = torch.load(checkpoint_path, "cpu")
        model = BartCaptionModel(ckpt["config"])
        model.load_state_dict(ckpt["model"])
        model.eval()
        return model.encoder
    
    def _train_epoch(self):
        epoch_loss = []
        self.model.train()
        self.loss_fn.train()

        for iteration in trange(self.epoch_length, ascii=True, ncols=100,
                                desc=f"Epoch {self.epoch}/{self.epochs} (train)"):

            try:
                batch = next(self.train_iter)
            except StopIteration:
                self.train_iter = iter(self.train_dataloader)
                batch = next(self.train_iter)

            #####################################################################
            # Update learning rate
            #####################################################################
            if self.lr_update_interval == "iteration":
                self.lr_scheduler.step()
                if wandb.run is not None:
                    wandb.log({"lr": self.optimizer.param_groups[0]["lr"]},
                              step=self.iteration)
                else:
                    self.tb_writer.add_scalar("lr",
                        self.optimizer.param_groups[0]["lr"], self.iteration)

            #####################################################################
            # Forward and backward
            #####################################################################
            self.optimizer.zero_grad()
            output = self._forward(batch, training=True)
            output["step"] = self.iteration
            loss = self.loss_fn(output)

            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                #####################################################################
                # Write the loss summary
                #####################################################################
                if wandb.run is not None:
                    wandb.log({"train/loss": loss.item()}, step=self.iteration)
                else:
                    self.tb_writer.add_scalar("loss/train", loss.item(), self.iteration)
                epoch_loss.append(loss.item())
            else:
                # import pdb; pdb.set_trace()
                pass

            self.iteration += 1

        return {
            "loss": np.mean(epoch_loss),
        }

    def _eval_epoch(self):
        eval_loss = []
        self.model.eval()
        self.loss_fn.eval()

        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, ascii=True, ncols=100,
                              desc=f"Epoch {self.epoch}/{self.epochs} (val)"):

                output = self._forward(batch, training=True)
                loss = self.loss_fn(output)
                eval_loss.append(loss.item())

        return np.mean(eval_loss)

    
    def train(self, config, **kwargs):
        """Trains a model on the given configurations.
        :param config: A training configuration. Note that all parameters in the config can also be manually adjusted with --ARG=VALUE
        :param **kwargs: parameters to overwrite yaml config
        """

        self.config = train_util.parse_config_or_kwargs(config, **kwargs)
        train_util.set_seed(self.config["seed"])

        #####################################################################
        # Create checkpoint directory
        #####################################################################
        exp_dir = Path(self.config["experiment_path"])
        exp_dir.mkdir(parents=True, exist_ok=True)
        self.logger = train_util.init_logger(str(exp_dir / "train.log"))
        # print passed config parameters
        if "SLURM_JOB_ID" in os.environ:
            self.logger.info(f"Slurm job id: {os.environ['SLURM_JOB_ID']}")
            self.logger.info(f"Slurm node: {os.environ['SLURM_JOB_NODELIST']}")
        elif "CONDOR_JOB_ID" in os.environ:
            self.logger.info(f"Condor job id: {os.environ['CONDOR_JOB_ID']}")
        self.logger.info(f"Storing files in: {exp_dir}")
        train_util.pprint_dict(self.config, self.logger.info)

        #####################################################################
        # Create dataloaders
        #####################################################################
        dataloaders = self._get_dataloaders()
        self.train_dataloader = dataloaders["train"]
        self.val_dataloader = dataloaders["val"]
        self.logger.info(f"the training dataset has "
            f"{len(self.train_dataloader.dataset)} samples")
        
        self.__dict__.update(self.config["trainer"])

        if not hasattr(self, "epoch_length"):
            self.epoch_length = len(self.train_dataloader)
        self.iterations = self.epoch_length * self.epochs

        #####################################################################
        # Build model
        #####################################################################

        self.teacher = self._get_teacher(self.logger.info).to(self.device)
        self.model = train_util.init_model_from_config(
            self.config["model"], self.logger.info).to(self.device)
        train_util.pprint_dict(self.model, self.logger.info, formatter="pretty")
        num_params = 0
        num_trainable_params = 0
        self.saving_keys = []
        for name, param in self.model.named_parameters():
            num_params += param.numel()
            if param.requires_grad:
                num_trainable_params += param.numel()
                self.saving_keys.append(name)
        for name, buffer in self.model.named_buffers():
            self.saving_keys.append(name)

        self.logger.info(f"{num_params} parameters in total")
        self.logger.info(f"{num_trainable_params} trainable parameters in total")

        #####################################################################
        # Build loss function and optimizer
        #####################################################################
        self.optimizer = train_util.init_obj_from_dict(
            self.config["optimizer"], params=self.model.parameters())
        self.loss_fn = train_util.init_obj_from_dict(self.config["loss"])
        train_util.pprint_dict(self.optimizer, self.logger.info,
                               formatter="pretty")

        #####################################################################
        # Tensorboard or wandb record
        #####################################################################
        if "wandb" in self.config:
            wandb.init(project=self.config["wandb"]["project"],
                       config=self.config,
                       name=self.config["wandb"]["name"],
                       dir=exp_dir)
        else:
            self.tb_writer = SummaryWriter(exp_dir / "run")

        #####################################################################
        # Create learning rate scheduler
        #####################################################################
        lr_sdl_cfg = self.config["lr_scheduler"]
        lr_sdl_type = lr_sdl_cfg["type"].split(".")[-1]
        if lr_sdl_type in ["ExponentialDecayScheduler",
                           "CosineWithWarmup"]:
            lr_sdl_cfg["args"]["total_iters"] = self.iterations
            if "warmup_iters" not in lr_sdl_cfg["args"]:
                warmup_iters = self.iterations // 5
                lr_sdl_cfg["args"]["warmup_iters"] = warmup_iters
            else:
                warmup_iters = lr_sdl_cfg["args"]["warmup_iters"]
            self.logger.info(f"Warm up iterations: {warmup_iters}")

        self.lr_scheduler = train_util.init_obj_from_dict(
            self.config["lr_scheduler"], optimizer=self.optimizer)

        #####################################################################
        # Dump configuration
        #####################################################################
        train_util.store_yaml(self.config, exp_dir / "config.yaml")

        #####################################################################
        # Start training
        #####################################################################
        
        self.metric_monitor = train_util.MetricImprover("min")
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
            val_loss = self._eval_epoch()

            #####################################################################
            # Update learning rate
            #####################################################################
            if self.lr_update_interval == "epoch":
                if self.lr_scheduler.__class__.__name__ == "ReduceLROnPlateau":
                    self.lr_scheduler.step(val_loss)
                else:
                    self.lr_scheduler.step()

            #####################################################################
            # Log results
            #####################################################################
            lr = self.optimizer.param_groups[0]["lr"]
            train_loss = train_output["loss"]
            output_str = f"epoch: {self.epoch}  train_loss: {train_loss:.2g}" \
                         f"  val_loss: {val_loss:.2g}  lr: {lr:.2g}"
            self.logger.info(output_str)
            if wandb.run is not None:
                wandb.log({"val/loss": val_loss}, step=self.iteration)
            else:
                self.tb_writer.add_scalar("loss/val", val_loss, self.epoch)

            #####################################################################
            # Save checkpoint
            #####################################################################

            if self.metric_monitor(val_loss):
                self.not_improve_cnt = 0
                self.save_checkpoint(exp_dir / "best.pth")
            else:
                self.not_improve_cnt += 1

            if self.epoch % self.save_interval == 0:
                self.save_checkpoint(exp_dir / "last.pth")

            if self.not_improve_cnt == self.early_stop:
                break
            
            self.epoch += 1


        if wandb.run is not None:
            wandb.finish()

        return exp_dir


    def debug(self, config, **kwargs):
        self.config = train_util.parse_config_or_kwargs(config)
        dataloaders = self._get_dataloaders()
        self.train_dataloader = dataloaders["train"]
        self.model = train_util.init_model_from_config(self.config["model"],
                                                       print).to(self.device)
        self.teacher = self._get_teacher(print).to(self.device)
        self.loss_fn = train_util.init_obj_from_dict(self.config["loss"])
        self.__dict__.update(self.config["trainer"])

        self.iteration = 1
        train_iter = iter(self.train_dataloader)
        for _ in range(10):
            batch = next(train_iter)
            output = self._forward(batch, training=True)
            loss = self.loss_fn(output)
            loss.backward()
        print("forward and backward done")



if __name__ == "__main__":
    fire.Fire(Runner)
