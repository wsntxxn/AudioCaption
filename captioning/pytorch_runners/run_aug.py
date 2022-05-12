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
from tqdm import tqdm

import captioning.models
import captioning.models.encoder
import captioning.models.decoder
import captioning.losses.loss as losses
import captioning.utils.train_util as train_util
from captioning.utils.build_vocab import Vocabulary
from captioning.pytorch_runners.base import BaseRunner
import captioning.datasets.caption_dataset as ac_dataset

class Runner(BaseRunner):

    def _get_augment_data(self, config):
        augments = train_util.parse_augments(config["augments"])
        if config["distributed"]:
            config["dataloader_args"]["batch_size"] //= self.world_size

        data_config = config["data"]["augmentation"]
        vocabulary = config["vocabulary"]
        raw_audio_to_h5 = train_util.load_dict_from_csv(
            data_config["raw_feat_csv"], ("audio_id", "hdf5_path"))
        fc_audio_to_h5 = train_util.load_dict_from_csv(
            data_config["fc_feat_csv"], ("audio_id", "hdf5_path"))
        attn_audio_to_h5 = train_util.load_dict_from_csv(
            data_config["attn_feat_csv"], ("audio_id", "hdf5_path"))
        caption_info = json.load(open(data_config["caption_file"], "r"))["audios"]

        dataset = ac_dataset.CaptionDataset(
            raw_audio_to_h5,
            fc_audio_to_h5,
            attn_audio_to_h5,
            caption_info,
            vocabulary,
            transform=augments
        )
        # TODO DistributedCaptionSampler
        # sampler = torch.utils.data.DistributedSampler(train_dataset) if config["distributed"] else None
        sampler = ac_dataset.CaptionSampler(dataset, shuffle=True)
        if config["data"]["train"]["size"] > len(dataset):
            batch_size = int(len(dataset) / config["data"]["train"]["size"] * \
                config["dataloader_args"]["batch_size"])
        else:
            batch_size = config["dataloader_args"]["batch_size"]
        dataloader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=ac_dataset.collate_fn([0, 2, 3], 3),
            sampler=sampler,
            batch_size=batch_size,
            num_workers=config["dataloader_args"]["num_workers"]
        )
        key2refs = {}
        for audio_idx in range(len(caption_info)):
            audio_id = caption_info[audio_idx]["audio_id"]
            key2refs[audio_id] = []
            for caption in caption_info[audio_idx]["captions"]:
                key2refs[audio_id].append(
                    caption["token" if config["zh"] else "caption"])
        return {
            "dataloader": dataloader,
            "key2refs": key2refs
        }

    @staticmethod
    def _get_model(config, outputfun=sys.stdout):
        vocabulary = config["vocabulary"]
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

    def _update_ss_ratio(self, config):
        mode = config["ss_args"]["ss_mode"]
        total_iters = config["data"]["total_iters"]
        if mode == "exponential":
            self.ss_ratio *= 0.01 ** (1.0 / total_iters)
        elif mode == "linear":
            self.ss_ratio -= (1.0 - config["ss_args"]["final_ss_ratio"]) / total_iters

    def _update_aug_factor(self, iteration, config):
        # self.aug_discount = iteration / config["data"]["total_iters"] * config["final_aug_discount"]
        self.aug_discount = 0.0

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
        conf["data"]["raw_feat_dim"] = train_dataloader.dataset.raw_feat_dim
        conf["data"]["fc_feat_dim"] = train_dataloader.dataset.fc_feat_dim
        conf["data"]["attn_feat_dim"] = train_dataloader.dataset.attn_feat_dim
        total_iters = len(train_dataloader) * conf["epochs"]
        conf["data"]["total_iters"] = total_iters
        conf["data"]["train"]["size"] = len(train_dataloader.dataset)
        aug_dataloader = self._get_augment_data(conf)["dataloader"]
        logger.info(f"Augmentation data size: {len(aug_dataloader.dataset)}")
        logger.info(f"Augmentation data batch size: {aug_dataloader.batch_size}")
        aug_dataiter = iter(aug_dataloader)

        #########################
        # Build model
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
        swa_model = torch.optim.swa_utils.AveragedModel(model)
        if not conf["distributed"] or not self.local_rank:
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
        if not conf["distributed"] or not self.local_rank:
            train_util.pprint_dict(optimizer, logger.info, formatter="pretty")
            crtrn_imprvd = train_util.criterion_improver(conf["improvecriterion"])

        #########################
        # Tensorboard record
        #########################
        if not conf["distributed"] or not self.local_rank:
            tb_writer = SummaryWriter(outputdir / "run")


        #########################
        # Create learning rate scheduler
        #########################
        try:
            scheduler = getattr(torch.optim.lr_scheduler, conf["scheduler"])(
                optimizer, **conf["scheduler_args"])
        except AttributeError:
            import captioning.utils.lr_scheduler as lr_scheduler
            if conf["scheduler"] == "ExponentialDecayScheduler":
                conf["scheduler_args"]["total_iters"] = total_iters
            if "warmup_iters" not in conf["scheduler_args"]:
                warmup_iters = total_iters // 5
                conf["scheduler_args"]["warmup_iters"] = warmup_iters
            if not conf["distributed"] or not self.local_rank:
                logger.info(f"Warm up iterations: {conf['scheduler_args']['warmup_iters']}")
            scheduler = getattr(lr_scheduler, conf["scheduler"])(
                optimizer, **conf["scheduler_args"])

        if scheduler.__class__.__name__ in ["StepLR", "ReduceLROnPlateau", "ExponentialLR", "MultiStepLR"]:
            epoch_update_lr = True
        else:
            epoch_update_lr = False


        #########################
        # Dump configuration
        #########################
        if not conf["distributed"] or not self.local_rank:
            del conf["vocabulary"]
            train_util.store_yaml(conf, outputdir / "config.yaml")

        #########################
        # Start training
        #########################

        self.ss_ratio = conf["ss_args"]["ss_ratio"]
        iteration = 0
        logger.info("{:^10}\t{:^10}\t{:^10}\t{:^10}\t{:^10}\t{:^10}".format(
            "Epoch", "Real loss", "Aug loss", "Val score", "Learning rate", "Aug discount"))
        
        for epoch in range(1, conf["epochs"] + 1):
            #########################
            # Training of one epoch
            #########################
            model.train()
            real_loss_history = []
            aug_loss_history = []
            with torch.enable_grad(), tqdm(total=len(train_dataloader), 
                                           ncols=100,
                                           desc=f"Epoch {epoch}",
                                           ascii=True) as pbar:
                for batch in train_dataloader:

                    iteration += 1

                    #########################
                    # Update scheduled sampling ratio
                    #########################
                    self._update_ss_ratio(conf)
                    tb_writer.add_scalar("scheduled_sampling_prob", self.ss_ratio, iteration)

                    #########################
                    # Update augmentation discount factor
                    #########################
                    self._update_aug_factor(iteration, conf)
                    tb_writer.add_scalar("augmentation_discount", self.aug_discount, iteration)

                    #########################
                    # Update learning rate
                    #########################
                    if not epoch_update_lr:
                        scheduler.step()
                        tb_writer.add_scalar("lr", optimizer.param_groups[0]["lr"], iteration)

                    #########################
                    # Forward and backward
                    #########################
                    optimizer.zero_grad()
                    # forward real data
                    output = self._forward(model, batch, "train",
                                           ss_ratio=self.ss_ratio)
                    loss_real = loss_fn(output)
                    # forward augmented data
                    try:
                        aug_batch = next(aug_dataiter)
                    except StopIteration:
                        aug_dataiter = iter(aug_dataloader)
                        aug_batch = next(aug_dataiter)
                    output_aug = self._forward(model, aug_batch, "train",
                                               ss_ratio=self.ss_ratio)
                    loss_aug = loss_fn(output_aug)
                    loss = loss_real + loss_aug * self.aug_discount
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                                   conf["max_grad_norm"])
                    optimizer.step()

                    #########################
                    # Write the loss summary
                    #########################
                    cap_lens = batch[-1]
                    real_loss_history.append(loss_real.item())
                    aug_loss_history.append(loss_aug.item())
                    if not conf["distributed"] or not self.local_rank:
                        tb_writer.add_scalar("loss/train", loss.item(), iteration)
                    pbar.set_postfix(running_loss=loss.item())
                    pbar.update()

            #########################
            # Stochastic weight averaging
            #########################
            if conf["swa"]:
                if epoch >= conf["swa_start"]:
                    swa_model.update_parameters(model)

            #########################
            # Validation of one epoch
            #########################
            model.eval()
            key2pred = {}
            with torch.no_grad(), tqdm(total=len(val_dataloader), ncols=100,
                                       ascii=True) as pbar:
                for batch in val_dataloader:
                    output = self._forward(model, batch, "validation",
                                           sample_method="beam", beam_size=3)
                    keys = batch[0]
                    seqs = output["seqs"].cpu().numpy()
                    for (idx, seq) in enumerate(seqs):
                        candidate = self._convert_idx2sentence(
                            seq, vocabulary.idx2word, zh)
                        key2pred[keys[idx]] = [candidate,]
                    pbar.update()
            scorer = Cider(zh=zh)
            score_output = self._eval_prediction(val_key2refs, key2pred, [scorer])
            score = score_output["CIDEr"]

            #########################
            # Update learning rate
            #########################
            if epoch_update_lr:
                scheduler.step(score)

            if not conf["distributed"] or not self.local_rank:
                #########################
                # Log results of this epoch
                #########################
                real_loss = np.mean(real_loss_history)
                aug_loss = np.mean(aug_loss_history)
                lr = optimizer.param_groups[0]["lr"]
                output_str = f"{epoch:^10}\t{real_loss:^10.2g}\t{aug_loss:^10.2g}\t{score:^10.2g}\t{lr:^10.2g}\t{self.aug_discount:^10.2g}"
                logger.info(output_str)
                tb_writer.add_scalar(f"score/val", score, epoch)

                #########################
                # Save checkpoint
                #########################
                dump = {
                    "model": model.state_dict() if not conf["distributed"] else model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": scheduler.state_dict(),
                    "vocabulary": vocabulary.idx2word
                }
                if crtrn_imprvd(score):
                    torch.save(dump, outputdir / "best.pth")
                torch.save(dump, outputdir / "last.pth")


        if not conf["distributed"] or not self.local_rank:
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
