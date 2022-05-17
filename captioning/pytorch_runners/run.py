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
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

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
        encoder = getattr(captioning.models.encoder,
                          self.config["model"]["encoder"]["type"])(
            **self.config["model"]["encoder"]["args"]
        )
        if "pretrained" in self.config["model"]["encoder"]:
            pretrained = self.config["model"]["encoder"]["pretrained"]
            train_util.load_pretrained_model(encoder,
                                             pretrained,
                                             print_fn)
        self.config["model"]["decoder"]["args"]["vocab_size"] = len(
            self.vocabulary)
        decoder = getattr(captioning.models.decoder,
                          self.config["model"]["decoder"]["type"])(
            **self.config["model"]["decoder"]["args"]
        )
        if "word_embedding" in self.config["model"]["decoder"]:
            decoder.load_word_embedding(**self.config["model"]["decoder"][
                "word_embedding"])
        if "pretrained" in self.config["model"]["decoder"]:
            pretrained = self.config["model"]["decoder"]["pretrained"]
            train_util.load_pretrained_model(decoder,
                                             pretrained,
                                             print_fn)
        model = getattr(
            captioning.models, self.config["model"]["type"])(
            encoder, decoder, **self.config["model"]["args"]
        )
        if "pretrained" in self.config["model"]:
            train_util.load_pretrained_model(model,
                                             self.config["model"]["pretrained"],
                                             print_fn)
        return model

    def _forward(self, model, batch, training=True, **kwargs):
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
            input_dict["ss_ratio"] = kwargs.get("ss_ratio", 1.0)
            output = model(input_dict)
            output["tgt"] = batch["cap"][:, 1:]
            output["tgt_len"] = torch.as_tensor(batch["cap_len"] - 1)
        else:
            input_dict.update(kwargs)
            output = model(input_dict)

        return output

    def _update_ss_ratio(self):
        mode = self.config["ss_args"]["ss_mode"]
        total_iters = self.iterations
        if mode == "exponential":
            self.ss_ratio *= 0.01 ** (1.0 / total_iters)
        elif mode == "linear":
            self.ss_ratio -= (1.0 - self.config["ss_args"][
                "final_ss_ratio"]) / total_iters
        else:
            raise Exception(f"mode {mode} not supported")
    
    def _train_epoch(self, model, dataloader):
        total_loss, nsamples = 0, 0
        model.train()
        for batch in tqdm(dataloader, ascii=True, ncols=100):
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
            output = self._forward(model, batch, "train",
                                   ss_ratio=self.ss_ratio)
            loss = self.loss_fn(output)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           self.config["max_grad_norm"])
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

    def _eval_epoch(self, model, dataloader, key2refs, **inference_args):
        if "sample_method" not in inference_args:
            inference_args["sample_method"] = "beam"
            inference_args["beam_size"] = 3
        key2pred = self._inference(model, dataloader, self.vocabulary.idx2word,
            **inference_args)
        scorer = Cider()
        result = self._eval_prediction(key2refs, key2pred, [scorer])[
            scorer.method()]
        return result
    
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
        outputdir = Path(self.config["outputpath"]) / \
            self.config["model"]["type"] / self.config["remark"] / \
            f"seed_{self.seed}"
        outputdir.mkdir(parents=True, exist_ok=True)
        logger = train_util.init_logger(str(outputdir / "train.log"))
        self.logger = logger
        # print passed config parameters
        if "SLURM_JOB_ID" in os.environ:
            logger.info(f"Slurm job id: {os.environ['SLURM_JOB_ID']}")
            logger.info(f"Slurm node: {os.environ['SLURM_JOB_NODELIST']}")
        self.logger.info(f"Storing files in: {outputdir}")
        train_util.pprint_dict(self.config, self.logger.info)

        #####################################################################
        # Create dataloaders
        #####################################################################
        dataloaders = self._get_dataloaders()
        train_dataloader = dataloaders["dataloader"]["train"]
        val_dataloader = dataloaders["dataloader"]["val"]
        val_key2refs = dataloaders["key2refs"]["val"]
        self.logger.info(f"the training dataset has "
            f"{len(train_dataloader.dataset)} samples")
        self.vocabulary = train_dataloader.dataset.vocabulary
        self.iterations = len(train_dataloader) * self.config["epochs"]

        #####################################################################
        # Build model
        #####################################################################
        model = self._get_model(self.logger.info)
        model = model.to(self.device)
        swa_model = torch.optim.swa_utils.AveragedModel(model)
        train_util.pprint_dict(model, logger.info, formatter="pretty")
        num_params = 0
        for param in model.parameters():
            num_params += param.numel()
        self.logger.info(f"{num_params} parameters in total")

        #####################################################################
        # Build loss function and optimizer
        #####################################################################
        self.optimizer = getattr(torch.optim, self.config["optimizer"]["type"])(
            model.parameters(), **self.config["optimizer"]["args"])
        self.loss_fn = getattr(losses, self.config["loss"]["type"])(
            **self.config["loss"]["args"])
        train_util.pprint_dict(self.optimizer, self.logger.info,
                               formatter="pretty")

        #####################################################################
        # Tensorboard record
        #####################################################################
        self.tb_writer = SummaryWriter(outputdir / "run")

        #####################################################################
        # Create learning rate scheduler
        #####################################################################
        try:
            self.lr_scheduler = getattr(torch.optim.lr_scheduler,
                self.config["lr_scheduler"]["type"])(self.optimizer,
                    **self.config["lr_scheduler"]["args"])
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
            self.lr_scheduler = getattr(lr_scheduler, lr_scheduler_config["type"])(
                self.optimizer, **lr_scheduler_config["args"])

        if self.lr_scheduler.__class__.__name__ in ["StepLR", "ReduceLROnPlateau",
                "ExponentialLR", "MultiStepLR"]:
            self.lr_update_interval = "epoch"
        else:
            self.lr_update_interval = "iteration"

        #####################################################################
        # Dump configuration
        #####################################################################
        train_util.store_yaml(self.config, outputdir / "config.yaml")

        #####################################################################
        # Start training
        #####################################################################
        
        best_val_result = -np.inf

        self.ss_ratio = self.config["ss_args"]["ss_ratio"]
        self.iteration = 1

        for epoch in range(1, self.config["epochs"] + 1):
            
            train_output = self._train_epoch(model, train_dataloader)
            val_result = self._eval_epoch(model, val_dataloader, val_key2refs,
                                          **self.config["inference_args"])

            #####################################################################
            # Stochastic weight averaging
            #####################################################################
            if self.config["swa"]:
                if epoch >= self.config["swa_start"]:
                    swa_model.update_parameters(model)

            #####################################################################
            # Update learning rate
            #####################################################################
            if self.lr_update_interval == "epoch":
                if self.lr_scheduler.__class__.__name__ == "ReduceLROnPlateau":
                    self.lr_scheduler.step(val_result)
                else:
                    self.lr_scheduler.step()

            #####################################################################
            # Log results
            #####################################################################
            lr = self.optimizer.param_groups[0]["lr"]
            train_loss = train_output['loss']
            output_str = f"epoch: {epoch}  train_loss: {train_loss:.2g}" \
                         f"  val_score: {val_result:.2g}  lr: {lr:.2g}"
            self.logger.info(output_str)
            self.tb_writer.add_scalar(f"score/val", val_result, epoch)

            #####################################################################
            # Save checkpoint
            #####################################################################
            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "vocabulary": self.vocabulary.idx2word
            }
            if val_result > best_val_result:
                best_val_result = val_result
                torch.save(checkpoint, outputdir / "best.pth")
            if epoch % self.config["save_interval"] == 0:
                torch.save(checkpoint, outputdir / "last.pth")


        #####################################################################
        # Stochastic weight averaging
        #####################################################################
        if self.config["swa"]:
            torch.save({
                "model": swa_model.module.state_dict(),
                "vocabulary": self.vocabulary.idx2word
            }, outputdir / "swa.pth")

        return outputdir


if __name__ == "__main__":
    fire.Fire(Runner)
