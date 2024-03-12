# coding=utf-8
#!/usr/bin/env python3
import os
import sys
from pathlib import Path

import fire
import torch
from torch.utils.tensorboard import SummaryWriter
import wandb
from tqdm import tqdm, trange

import captioning.utils.train_util as train_util
from captioning.pytorch_runners.base import BaseRunner
from pycocoevalcap.cider.cider import Cider


class Runner(BaseRunner):

    def _forward(self, batch, training=True):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                if k == "cap":
                    batch[k] = v.long().to(self.device)
                else:
                    batch[k] = v.float().to(self.device)

        # student forward
        input_dict = {
            "mode": "train" if training else "inference",
        }
        input_dict.update(batch)

        if training:
            input_dict["ss_ratio"] = self.ss_ratio
            if "specaug" in self.config:
                input_dict["specaug"] = self.config["specaug"]
            output = {}
            # teacher forward
            with torch.no_grad():
                tchr_output = {}
                if "token" in self.config["kd_type"]:
                    text_input_dict = {
                        "input_ids": batch["cap"],
                        "attention_mask": batch["attention_mask"],
                    }
                    tchr_output = self.teacher(audio=batch["teacher_wav"],
                                               text=text_input_dict,
                                               return_status=True)
                    output["tchr_logit"] = tchr_output["logits"]
                if "seq" in self.config["kd_type"]:
                    if all([aid in self.aid_to_tchr_seq for aid in batch["audio_id"]]):
                        tchr_seq = [self.aid_to_tchr_seq[aid] for aid in batch["audio_id"]]
                    else:
                        tchr_output = self.teacher.generate(
                            samples=batch["teacher_wav"],
                            num_beams=3
                        )
                        tchr_seq = []
                        for aid, seq in zip(batch["audio_id"], tchr_output["caption"]):
                            self.aid_to_tchr_seq[aid] = seq
                            tchr_seq.append(seq)
                    tokenized_seq = self.tokenizer(tchr_seq)
                    for k, v in tokenized_seq.items():
                        if isinstance(v, torch.Tensor):
                            if k == "cap":
                                tokenized_seq[k] = v.long().to(self.device)
                            else:
                                tokenized_seq[k] = v.float().to(self.device)
                    output["tchr_tgt"] = tokenized_seq["cap"][:, 1:]
                    output["tchr_tgt_len"] = tokenized_seq["cap_len"] - 1

                if "enc" in self.config["kd_type"]:
                    if "encoder_output" in tchr_output:
                        tchr_enc_output = tchr_output["encoder_output"]
                    else:
                        tchr_enc_output = self.teacher.forward_encoder(batch["teacher_wav"])

            if "seq" in self.config["kd_type"]:
                input_dict.update(tokenized_seq)
                tchr_output = self.model(input_dict)
                output["tchr_cap_logit"] = tchr_output["logit"]

            input_dict.update(batch)
            if "enc" in self.config["kd_type"]:
                input_dict["tchr_output"] = tchr_enc_output

            stdnt_output = self.model(input_dict)
            output.update(stdnt_output)
            
            output["tgt"] = batch["cap"][:, 1:]
            output["tgt_len"] = torch.as_tensor(batch["cap_len"] - 1)
        else:
            input_dict["specaug"] = False
            input_dict.update(self.config["inference_args"])
            output = self.model(input_dict)

        return output

    def _forward_unsup(self, batch): 
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.float().to(self.device)

        # student forward
        input_dict = { "mode": "train" }
        input_dict.update(batch)
        input_dict["ss_ratio"] = self.ss_ratio

        if "specaug" in self.config:
            input_dict["specaug"] = self.config["specaug"]
        output = {}
        # teacher forward
        with torch.no_grad():
            tchr_output = {}
            if "token" in self.config["kd_type"]:
                text_input_dict = {
                    "input_ids": batch["cap"],
                    "attention_mask": batch["attention_mask"],
                }
                tchr_output = self.teacher(audio=batch["teacher_wav"],
                                           text=text_input_dict,
                                           return_status=True)
                output["tchr_logit"] = tchr_output["logits"]
            if "seq" in self.config["kd_type"]:
                tchr_output = self.teacher.generate(
                    samples=batch["teacher_wav"],
                    num_beams=3
                )
                tchr_seq = []
                for aid, seq in zip(batch["audio_id"], tchr_output["caption"]):
                    self.aid_to_tchr_seq[aid] = seq
                    tchr_seq.append(seq)
                tokenized_seq = self.tokenizer(tchr_seq)
                for k, v in tokenized_seq.items():
                    if isinstance(v, torch.Tensor):
                        if k == "cap":
                            tokenized_seq[k] = v.long().to(self.device)
                        else:
                            tokenized_seq[k] = v.float().to(self.device)
                output["tchr_tgt"] = tokenized_seq["cap"][:, 1:]
                output["tchr_tgt_len"] = tokenized_seq["cap_len"] - 1

            if "enc" in self.config["kd_type"]:
                if "encoder_output" in tchr_output:
                    tchr_enc_output = tchr_output["encoder_output"]
                else:
                    tchr_enc_output = self.teacher.forward_encoder(batch["teacher_wav"])

        if "seq" in self.config["kd_type"]:
            input_dict.update(tokenized_seq)
            tchr_output = self.model(input_dict)
            output["tchr_cap_logit"] = tchr_output["logit"]

        input_dict.update(batch)
        if "enc" in self.config["kd_type"]:
            input_dict["tchr_output"] = tchr_enc_output
        input_dict["unsup"] = True
        stdnt_output = self.model(input_dict)
        output.update(stdnt_output)

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

    def _get_model(self, print_fn=sys.stdout.write):
        model = train_util.init_model_from_config(self.config["model"], print_fn)
        if model.__class__.__name__ == "ScstWrapper":
            self.rl_train = True
        else:
            self.rl_train = False
        if hasattr(self, "tokenizer"):
            model.set_index(self.tokenizer.bos, self.tokenizer.eos, self.tokenizer.pad)

        if "kd_pretrained" in self.config:
            prt_ckpt_path = self.config["kd_pretrained"]
            prt_cfg_path = Path(prt_ckpt_path).parent / "config.yaml"
            prt_cfg = train_util.load_config(prt_cfg_path)
            prt_model = train_util.init_model_from_config(prt_cfg["model"], print_fn)
            prt_ckpt = torch.load(prt_ckpt_path, "cpu")
            train_util.load_pretrained_model(prt_model, prt_ckpt, print_fn)
            if prt_model.__class__.__name__ == "ContraEncoderKdWrapper":
                model.encoder.load_state_dict(prt_model.model.state_dict())
                
        return model

    def _get_teacher(self, print_fn=sys.stdout.write):
        sys.path.append("/mnt/fast/nobackup/scratch4weeks/xx00336/workspace/wavcaps/captioning")
        from models.bart_captioning import BartCaptionModel
        checkpoint_path = self.config["teacher"]
        ckpt = torch.load(checkpoint_path, "cpu")
        model = BartCaptionModel(ckpt["config"])
        model.load_state_dict(ckpt["model"])
        model.eval()
        return model

    
    def _train_epoch(self):
        losses = []
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
            if wandb.run is not None:
                wandb.log({"scheduled_sampling_prob": self.ss_ratio},
                          step=self.iteration)
            else:
                self.tb_writer.add_scalar("scheduled_sampling_prob",
                    self.ss_ratio, self.iteration)

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
            output["verbose"] = False
            if self.rl_train:
                loss = output["loss"]
            else:
                output["step"] = self.iteration
                loss = self.loss_fn(output)

            try:
                unsup_batch = next(self.unsup_iter)
            except StopIteration:
                self.unsup_iter = iter(self.unsup_dataloader)
                unsup_batch = next(self.unsup_iter)
            output = self._forward_unsup(unsup_batch)
            output["verbose"] = False
            unsup_loss = self.unsup_loss_fn(output)

            if not torch.isnan(loss) and not torch.isnan(unsup_loss):
                tot_loss = loss + unsup_loss
                tot_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                #####################################################################
                # Write the loss summary
                #####################################################################
                if wandb.run is not None:
                    wandb.log({"train/loss": loss.item()}, step=self.iteration)
                    wandb.log({"train/unsup_loss": unsup_loss.item()}, step=self.iteration)
                else:
                    self.tb_writer.add_scalar("loss/train", loss.item(), self.iteration)
                    self.tb_writer.add_scalar("unsup_loss/train", unsup_loss.item(), self.iteration)
                losses.append(tot_loss.item())
            else:
                # import pdb; pdb.set_trace()
                pass

            self.iteration += 1

        return {
            "loss": sum(losses) / len(losses),
        }

    def _eval_epoch(self):
        key2pred = self._inference(self.val_dataloader)
        scorer = Cider()
        result = self._eval_prediction(self.val_key2refs, key2pred, [scorer])
        result = result[scorer.method()]
        return { "score": result }


    def _get_dataloaders(self):
        dataloaders = super()._get_dataloaders()
        unsup_config = self.config["data"]["unsupervised"]
        dataset = train_util.init_obj_from_dict(unsup_config["dataset"])
        collate_fn = train_util.init_obj_from_dict(unsup_config["collate_fn"])
        if "batch_sampler" in unsup_config:
            batch_sampler = train_util.init_obj_from_dict(
                unsup_config["batch_sampler"], dataset=dataset)
        else:
            batch_sampler = None
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset, collate_fn=collate_fn,
            batch_sampler=batch_sampler, **unsup_config["dataloader_args"])
        dataloaders["dataloader"]["unsup"] = dataloader
        return dataloaders

    
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
        exp_dir = Path(self.config["experiment_path"]) / f"seed_{self.config['seed']}"
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
        self.train_dataloader = dataloaders["dataloader"]["train"]
        self.val_dataloader = dataloaders["dataloader"]["val"]
        self.train_key2refs = dataloaders["key2refs"]["train"]
        self.val_key2refs = dataloaders["key2refs"]["val"]
        self.unsup_dataloader = dataloaders["dataloader"]["unsup"]
        self.logger.info(f"the training dataset has "
            f"{len(self.train_dataloader.dataset)} samples")
        self.tokenizer = self.train_dataloader.collate_fn.tokenizer
        
        self.__dict__.update(self.config["trainer"])

        if not hasattr(self, "epoch_length"):
            self.epoch_length = len(self.train_dataloader)
        self.iterations = self.epoch_length * self.epochs

        #####################################################################
        # Build model
        #####################################################################

        if "teacher" in self.config:
            self.teacher = self._get_teacher(self.logger.info).to(self.device)
        self.aid_to_tchr_seq = {}

        self.model = self._get_model(self.logger.info).to(self.device)
        swa_model = train_util.AveragedModel(self.model)
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
        self.unsup_loss_fn = train_util.init_obj_from_dict(self.config["unsup_loss"])
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
        self.unsup_iter = iter(self.unsup_dataloader)

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
            if wandb.run is not None:
                wandb.log({"val/score": val_score}, step=self.iteration)
            else:
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
            saved_dict = {"model": { k: model_dict[k] for k in self.saving_keys }}
            if self.tokenizer.__class__.__name__ == "DictTokenizer":
                saved_dict["tokenizer"] = self.tokenizer.state_dict()
            torch.save(saved_dict, exp_dir / "swa.pth")

        if wandb.run is not None:
            wandb.finish()

        return exp_dir


    def debug(self, config, **kwargs):
        self.config = train_util.parse_config_or_kwargs(config)
        dataloaders = self._get_dataloaders()
        self.train_dataloader = dataloaders["dataloader"]["train"]
        self.tokenizer = self.train_dataloader.collate_fn.tokenizer
        self.model = self._get_model(print).to(self.device)
        if "teacher" in self.config:
            self.teacher = self._get_teacher(print).to(self.device)
        self.loss_fn = train_util.init_obj_from_dict(self.config["loss"])
        self.unsup_loss_fn = train_util.init_obj_from_dict(self.config["unsup_loss"])
        self.__dict__.update(self.config["trainer"])

        self.ss_ratio = 1.0
        self.iteration = 1
        self.aid_to_tchr_seq = {}
        batch = next(iter(self.train_dataloader))
        output = self._forward(batch, training=True)
        loss = self.loss_fn(output)

        self.unsup_dataloader = dataloaders["dataloader"]["unsup"]
        unsup_batch = next(iter(self.unsup_dataloader))
        output = self._forward_unsup(unsup_batch)
        unsup_loss = self.unsup_loss_fn(output)
        tot_loss = loss + unsup_loss
        tot_loss.backward()
        print("forward and backward done")



if __name__ == "__main__":
    fire.Fire(Runner)
