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
from captioning.pytorch_runners.run import Runner as BaseRunner

class Runner(BaseRunner):

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

        raw_feats = raw_feats.float().to(self.device)
        fc_feats = fc_feats.float().to(self.device)
        attn_feats = attn_feats.float().to(self.device)
        input_dict = {
            "mode": "train" if mode == "train" else "inference",
            "raw_feats": raw_feats,
            "raw_feat_lens": raw_feat_lens,
            "fc_feats": fc_feats,
            "attn_feats": attn_feats,
            "attn_feat_lens": attn_feat_lens
        }

        if mode == "train":
            caps = caps.long().to(self.device)
            input_dict["caps"] = caps
            input_dict["cap_lens"] = cap_lens
            input_dict["ss_ratio"] = kwargs["ss_ratio"]
            output = model(input_dict)
            output["targets"] = caps[:, 1:]
            output["lens"] = torch.as_tensor(cap_lens - 1)
        else:
            input_dict.update(kwargs)
            output = model(input_dict)

        return output

    def inference(self, model, dataloader, vocabulary, zh):
        model.eval()
        key2pred = {}
        with torch.no_grad(), tqdm(total=len(dataloader),
            ncols=100, ascii=True, leave=False) as pbar:
            for batch in dataloader:
                output = self._forward(model, batch, "validation",
                                       sample_method="beam", beam_size=3)
                keys = batch[0]
                seqs = output["seqs"].cpu().numpy()
                for (idx, seq) in enumerate(seqs):
                    candidate = self._convert_idx2sentence(
                        seq, vocabulary.idx2word, zh)
                    key2pred[keys[idx]] = [candidate,]
                pbar.update()
        return key2pred

    def train(self, config, **kwargs):
        """Trains a model on the given configurations.
        :param config: A training configuration. Note that all parameters in the config can also be manually adjusted with --ARG=VALUE
        :param **kwargs: parameters to overwrite yaml config
        """
        from pycocoevalcap.cider.cider import Cider

        conf = train_util.parse_config_or_kwargs(config, **kwargs)
        conf["seed"] = self.seed

        #########################
        # Create checkpoint directory
        #########################
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
        conf["data"]["raw_feat_dim"] = train_dataloader.dataset.raw_feat_dim
        conf["data"]["fc_feat_dim"] = train_dataloader.dataset.fc_feat_dim
        conf["data"]["attn_feat_dim"] = train_dataloader.dataset.attn_feat_dim
        total_iters = conf["iterations"]
        conf["data"]["total_iters"] = total_iters
        logger.info(f"the training dataset has {len(train_dataloader.dataset)} samples")

        #########################
        # Build model
        #########################
        model = self._get_model(conf, logger.info)
        model = model.to(self.device)
        swa_model = torch.optim.swa_utils.AveragedModel(model)
        train_util.pprint_dict(model, logger.info, formatter="pretty")
        num_params = 0
        for param in model.parameters():
            num_params += param.numel()
        logger.info(f"{num_params} parameters in total")

        #########################
        # Build loss function and optimizer
        #########################
        optimizer = getattr(torch.optim, conf["optimizer"])(
            model.parameters(), **conf["optimizer_args"])
        loss_fn = getattr(losses, conf["loss"])(**conf["loss_args"])
        train_util.pprint_dict(optimizer, logger.info, formatter="pretty")
        crtrn_imprvd = train_util.criterion_improver(conf["improvecriterion"])

        #########################
        # Tensorboard record
        #########################
        tb_writer = SummaryWriter(outputdir / "run")

        #########################
        # Create learning rate scheduler
        #########################
        try:
            scheduler = getattr(torch.optim.lr_scheduler, conf["scheduler"])(
                optimizer, **conf["scheduler_args"])
        except AttributeError:
            import captioning.utils.lr_scheduler as lr_scheduler
            if conf["scheduler"] in ["ExponentialDecayScheduler", 
                "CosineWithWarmup"]:
                conf["scheduler_args"]["total_iters"] = total_iters
            if "warmup_iters" not in conf["scheduler_args"]:
                warmup_iters = total_iters // 5
                conf["scheduler_args"]["warmup_iters"] = warmup_iters
            else:
                warmup_iters = conf["scheduler_args"]["warmup_iters"]
            logger.info(f"Warm up iterations: {warmup_iters}")
            scheduler = getattr(lr_scheduler, conf["scheduler"])(
                optimizer, **conf["scheduler_args"])

        if scheduler.__class__.__name__ in ["StepLR", "ReduceLROnPlateau", "ExponentialLR", "MultiStepLR"]:
            epoch_update_lr = True
        else:
            epoch_update_lr = False

        #########################
        # Dump configuration
        #########################
        del conf["vocabulary"]
        train_util.store_yaml(conf, outputdir / "config.yaml")

        #########################
        # Start training
        #########################

        self.ss_ratio = conf["ss_args"]["ss_ratio"]

        train_dataiter = iter(train_dataloader)
        iteration = 0
        pbar = tqdm(total=conf["val_freq"], ascii=True, leave=False, ncols=100)
        loss_history = []
        nsample_history = []

        for iteration in range(1, conf["iterations"] + 1):
            #########################
            # Validation
            #########################
            if iteration % conf["val_freq"] == 0:
                pbar.close()

                if conf["swa"]:
                    if iteration >= conf["swa_start"]:
                        swa_model.update_parameters(model)
                
                key2pred = self.inference(model, val_dataloader, vocabulary, zh)
                scorer = Cider()
                score_output = self._eval_prediction(val_key2refs, key2pred, [scorer])
                score = score_output["CIDEr"]

                #########################
                # Update learning rate
                #########################
                if epoch_update_lr:
                    scheduler.step(score)

                #########################
                # Log results of this epoch
                #########################
                train_loss = np.sum(loss_history) / np.sum(nsample_history)
                lr = optimizer.param_groups[0]["lr"]
                output_str = f"iteration: {iteration}  train_loss: {train_loss:.2g}  val_score: {score:.2g}  {lr:.2g}"
                logger.info(output_str)
                tb_writer.add_scalar(f"score/val", score, iteration)

                #########################
                # Save checkpoint
                #########################
                dump = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": scheduler.state_dict(),
                    "vocabulary": vocabulary.idx2word
                }
                if crtrn_imprvd(score):
                    torch.save(dump, outputdir / "best.pth")
                torch.save(dump, outputdir / "last.pth")

                pbar = tqdm(total=conf["val_freq"], ascii=True, leave=False, ncols=100)
                loss_history = []
                nsample_history = []

            #########################
            # Training
            #########################
            model.train()
            try:
                batch = next(train_dataiter)
            except StopIteration:
                train_dataiter = iter(train_dataloader)
                batch = next(train_dataiter)


            #########################
            # Update scheduled sampling ratio
            #########################
            self._update_ss_ratio(conf)
            tb_writer.add_scalar("scheduled_sampling_prob", self.ss_ratio,
                iteration)

            #########################
            # Update learning rate
            #########################
            if not epoch_update_lr:
                scheduler.step()
                tb_writer.add_scalar("lr", optimizer.param_groups[0]["lr"],
                    iteration)

            #########################
            # Forward and backward
            #########################
            optimizer.zero_grad()
            output = self._forward(model, batch, "train",
                                   ss_ratio=self.ss_ratio)
            loss = loss_fn(output)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                           conf["max_grad_norm"])
            optimizer.step()

            #########################
            # Write the loss summary
            #########################
            cap_lens = batch[-1]
            nsample = sum(cap_lens - 1)
            loss_history.append(loss.item() * nsample)
            nsample_history.append(sum(cap_lens - 1))
            tb_writer.add_scalar("loss/train", loss.item(), iteration)
            pbar.set_postfix(running_loss=loss.item())
            pbar.update()

        pbar.close()

        #########################
        # Stochastic weight averaging
        #########################
        if conf["swa"]:
            torch.save({
                "model": swa_model.module.state_dict(),
                "vocabulary": vocabulary.idx2word
            }, outputdir / "swa.pth")
        return outputdir


if __name__ == "__main__":
    fire.Fire(Runner)
