# coding=utf-8
#!/usr/bin/env python3
import os
import sys
import json
from pathlib import Path

import fire
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

import captioning.models
import captioning.models.encoder
import captioning.models.decoder
import captioning.datasets as dataset_module
import captioning.losses.loss as losses
import captioning.utils.train_util as train_util
from captioning.utils.build_vocab import Vocabulary
from captioning.pytorch_runners.run import Runner as BaseRunner
from pycocoevalcap.cider.cider import Cider



class DummyLogger:

    def info(self, *args):
        pass


class Runner(BaseRunner):

    def dist_init(self):
        rank = int(os.environ["SLURM_PROCID"])
        self.local_rank = int(os.environ["SLURM_LOCALID"])
        self.world_size = int(os.environ["SLURM_NTASKS"])
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="file://" + self.config["sync_file"],
            world_size=self.world_size,
            rank=rank)
        self.is_main_rank = rank == 0

    def _get_dataloaders(self):
        dataloaders, key2refses = {}, {}
        for split in ["train", "val"]:
            data_config = self.config["data"][split]
            dataset_config = data_config["dataset"]
            dataset = getattr(dataset_module, dataset_config["type"])(
                **dataset_config["args"])
            collate_config = data_config["collate_fn"]
            collate_fn = getattr(dataset_module, collate_config["type"])(
                **collate_config["args"])
            if "batch_sampler" in data_config:
                bs_config = data_config["batch_sampler"].copy()
                bs_config["args"]["batch_size"] //= self.world_size
                batch_sampler = getattr(dataset_module, bs_config["type"])(
                    dataset,
                    **bs_config["args"])
                batch_sampler = dataset_module.DistributedBatchSampler(
                    dataset, batch_sampler)
                sampler = None
            else:
                batch_sampler = None
                if split == "train":
                    sampler = torch.utils.data.distributed.DistributedSampler(
                        dataset, shuffle=True)
                    data_config["dataloader_args"]["batch_size"] //= \
                        self.world_size
                    del data_config["dataloader_args"]["shuffle"]
                else:
                    sampler = None
            dataloader = torch.utils.data.DataLoader(
                dataset=dataset, collate_fn=collate_fn,
                sampler=sampler, batch_sampler=batch_sampler,
                **data_config["dataloader_args"])
            dataloaders[split] = dataloader

            key2refs = {}
            try:
                caption_info = dataset.caption_info
            except AttributeError:
                caption_info = json.load(open(data_config["caption"]))["audios"]

            for audio_idx in range(len(caption_info)):
                audio_id = caption_info[audio_idx]["audio_id"]
                key2refs[audio_id] = []
                for caption in caption_info[audio_idx]["captions"]:
                    key2refs[audio_id].append(caption[
                        "tokens" if self.config["zh"] else "caption"])
            key2refses[split] = key2refs

        return {
            "dataloader": dataloaders,
            "key2refs": key2refses
        }

    def _get_model(self, print_fn=sys.stdout.write):
        model = super()._get_model(print_fn=print_fn)
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device(f"cuda:{self.local_rank}")
        model = model.to(self.device)
        if self.config["model"]["sync_bn"]:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[self.local_rank], output_device=self.local_rank,
            find_unused_parameters=True)
        return model

    def _train_epoch(self):
        total_loss, nsamples = 0, 0
        self.model.train()

        if self.is_main_rank:
            iter_range = trange(self.epoch_length, ascii=True, ncols=100,
                                desc=f"Epoch {self.epoch}/{self.epochs}")
        else:
            iter_range = range(self.epoch_length)

        for iteration in iter_range:

            try:
                batch = next(self.train_iter)
            except StopIteration:
                self.train_iter = iter(self.train_dataloader)
                batch = next(self.train_iter)

            #####################################################################
            # Update scheduled sampling ratio
            #####################################################################
            self._update_ss_ratio()
            if self.is_main_rank:
                self.tb_writer.add_scalar("scheduled_sampling_prob",
                    self.ss_ratio, self.iteration)

            #####################################################################
            # Update learning rate
            #####################################################################
            if self.lr_update_interval == "iteration":
                self.lr_scheduler.step()
                if self.is_main_rank:
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
            if self.is_main_rank:
                self.tb_writer.add_scalar("loss/train", loss.item(),
                                          self.iteration)
            cap_len = batch["cap_len"]
            nsample = sum(cap_len - 1)
            total_loss += loss.item() * nsample
            nsamples += nsample
            
            self.iteration += 1

        return {
            "loss": total_loss / nsamples
        }

    def save_checkpoint(self, ckpt_path):
        model_dict = self.model.module.state_dict()
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

    def train(self, config, **kwargs):
        """Trains a model on the given configurations.
        :param config: A training configuration. Note that all parameters in the config can also be manually adjusted with --ARG=VALUE
        :param **kwargs: parameters to overwrite yaml config
        """

        self.config = train_util.parse_config_or_kwargs(config, **kwargs)
        self.config["seed"] = self.seed

        self.dist_init()

        #####################################################################
        # Create checkpoint directory
        #####################################################################
        if self.is_main_rank:
            exp_dir = Path(self.config["outputpath"]) / \
                self.config["model"]["type"] / self.config["remark"] / \
                f"seed_{self.seed}"
            exp_dir.mkdir(parents=True, exist_ok=True)
            logger = train_util.init_logger(str(exp_dir / "train.log"))
            logger.info(f"Storing files in: {exp_dir}")
        else:
            logger = DummyLogger()

        self.logger = logger
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
        self.model = self._get_model(self.logger.info)
        swa_model = train_util.AveragedModel(self.model)
        train_util.pprint_dict(self.model, self.logger.info, formatter="pretty")
        num_params = 0
        num_trainable_params = 0
        self.saving_keys = []
        for name, param in self.model.module.named_parameters():
            num_params += param.numel()
            if param.requires_grad:
                num_trainable_params += param.numel()
                self.saving_keys.append(name)
        for name, buffer in self.model.module.named_buffers():
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
        if self.is_main_rank:
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
            if lr_scheduler_config["type"] in ["ExponentialDecayScheduler",
                "CosineWithWarmup"]:
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
        if self.is_main_rank:
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

            if hasattr(self.train_dataloader.sampler, "set_epoch"):
                self.train_dataloader.sampler.set_epoch(self.epoch)

            train_output = self._train_epoch()
            val_result = self._eval_epoch()
            val_score = val_result[metric_name]

            #####################################################################
            # Stochastic weight averaging
            #####################################################################
            if self.config["swa"]:
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

            if self.is_main_rank:
                #################################################################
                # Log results
                #################################################################
                lr = self.optimizer.param_groups[0]["lr"]
                train_loss = train_output["loss"]
                output_str = f"epoch: {self.epoch}  train_loss: {train_loss:.2g}" \
                             f"  val_score: {val_score:.2g}  lr: {lr:.2g}"
                self.logger.info(output_str)
                self.tb_writer.add_scalar(f"score/val", val_score, self.epoch)

                #################################################################
                # Save checkpoint
                #################################################################
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


        if self.is_main_rank:
            #####################################################################
            # Stochastic weight averaging
            #####################################################################
            if self.config["swa"]["use"]:
                model_dict = swa_model.module.module.state_dict()
                torch.save({
                    "model": { k: model_dict[k] for k in self.saving_keys },
                    "vocabulary": self.vocabulary.idx2word
                }, exp_dir / "swa.pth")

            return exp_dir


if __name__ == "__main__":
    fire.Fire(Runner)
