# coding=utf-8
#!/usr/bin/env python3
import os
import sys
import datetime
import random
import uuid

from tqdm import tqdm
import fire
import numpy as np
import pandas as pd
import sklearn.preprocessing as pre
import torch
from ignite.engine.engine import Engine, Events
from ignite.metrics import Accuracy, Loss, RunningAverage, Average
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.contrib.handlers import ProgressBar
from ignite.utils import convert_tensor

sys.path.append(os.getcwd())
import models
import utils.train_util as train_util
from utils.build_vocab import Vocabulary
from runners.base_runner import BaseRunner
from datasets.SJTUDataSet import SJTUInstanceDataset, SJTUDatasetEval, collate_fn

class Runner(BaseRunner):

    @staticmethod
    def _get_dataloaders(config, vocabulary):
        scaler = getattr(
            pre, config["scaler"])(
            **config["scaler_args"])
        inputdim = -1
        caption_df = pd.read_json(config["caption_file"], dtype={"key": str})

        for batch in tqdm(
            torch.utils.data.DataLoader(
                SJTUInstanceDataset(
                    kaldi_stream=config["feature_stream"],
                    caption_df=caption_df,
                    vocabulary=vocabulary,
                ),
                collate_fn=collate_fn([0, 1]),
                **config["dataloader_args"]
            ),
            ascii=True
        ):
            feat = batch[0]
            feat_lens = batch[-2]
            packed_feat = torch.nn.utils.rnn.pack_padded_sequence(
                feat, feat_lens, batch_first=True, enforce_sorted=False).data
            scaler.partial_fit(packed_feat)
            inputdim = feat.shape[-1]
        assert inputdim > 0, "Reading inputstream failed"

        augments = train_util.parse_augments(config["augments"])

        train_keys = np.random.choice(
            caption_df["key"].unique(), 
            int(len(caption_df["key"].unique()) * (config["train_percent"] / 100.)), 
            replace=False
        )
        train_df = caption_df[caption_df["key"].apply(lambda x: x in train_keys)]
        val_df = caption_df[~caption_df.index.isin(train_df.index)]

        train_loader = torch.utils.data.DataLoader(
            SJTUInstanceDataset(
                kaldi_stream=config["feature_stream"],
                caption_df=train_df,
                vocabulary=vocabulary,
                transform=[scaler.transform, augments]
            ),
            shuffle=True,
            collate_fn=collate_fn([0, 1]),
            **config["dataloader_args"]
        )

        if config["zh"]:
            train_key2refs = train_df.groupby("key")["tokens"].apply(list).to_dict()
            val_key2refs = val_df.groupby("key")["tokens"].apply(list).to_dict()
        else:
            train_key2refs = train_df.groupby("key")["caption"].apply(list).to_dict()
            val_key2refs = val_df.groupby("key")["caption"].apply(list).to_dict()
        val_loader = torch.utils.data.DataLoader(
            SJTUInstanceDataset(
                kaldi_stream=config["feature_stream"],
                caption_df=val_df,
                vocabulary=vocabulary,
                transform=scaler.transform
            ),
            shuffle=False,
            collate_fn=collate_fn([0, 1]),
            **config["dataloader_args"]
        )

        return train_loader, val_loader, {
            "scaler": scaler, "inputdim": inputdim, 
            "train_key2refs": train_key2refs, "val_key2refs": val_key2refs
        }

    @staticmethod
    def _get_model(config, vocab_size):
        assert "audio_embed_size" in config["model_args"]
        assert "num_instance" in config["model_args"]
        assert "instance_embed_size" in config["model_args"]
        audio_embed_size = config["model_args"]["audio_embed_size"]
        instance_embed_size = config["model_args"]["instance_embed_size"]
        encodermodel = getattr(
            models.encoder, config["encodermodel"])(
            inputdim=config["inputdim"],
            embed_size=audio_embed_size,
            **config["encodermodel_args"])
        if "pretrained_encoder" in config:
            encoder_state_dict = torch.load(
                config["pretrained_encoder"],
                map_location="cpu")
            encodermodel.load_state_dict(encoder_state_dict, strict=False)

        if config["decodermodel"] == "RNNBahdanauAttnDecoder":
            input_size = audio_embed_size * 2 + instance_embed_size
        else:
            input_size = audio_embed_size + instance_embed_size
        decodermodel = getattr(
            models.decoder, config["decodermodel"])(
            vocab_size=vocab_size,
            input_size=input_size,
            **config["decodermodel_args"])
        model = getattr(models, config["model"])(
            encodermodel, decodermodel, **config["model_args"])
        if "pretrained" in config:
            pretrained_model = torch.load(config["pretrained"], map_location="cpu")["model"]
            model.load_state_dict(pretrained_model.state_dict(), strict=False)
        return model

    def _forward(self, model, batch, mode="train", **kwargs):
        assert mode in ("train", "sample")

        if mode == "sample":
            feats = batch[1]
            feat_lens = batch[-1]
            feats = convert_tensor(feats.float(),
                                   device=self.device,
                                   non_blocking=True)

            if "instance_labels" in kwargs:
                instance_labels = kwargs["instance_labels"]

                instance_labels = convert_tensor(instance_labels.float(),
                                                device=self.device,
                                                non_blocking=True)

                sampled = model(feats, feat_lens, instance_labels, **kwargs)
            else:
                sampled = model(feats, feat_lens, **kwargs)
            return sampled

        # mode is "train"
        assert "tf" in kwargs, "need to know whether to use teacher forcing"

        feats = batch[0]
        caps = batch[1]
        cap_idxs = batch[2]
        feat_lens = batch[-2]
        cap_lens = batch[-1]
        feats = convert_tensor(feats.float(),
                               device=self.device,
                               non_blocking=True)
        caps = convert_tensor(caps.long(),
                              device=self.device,
                              non_blocking=True)
        cap_idxs = convert_tensor(cap_idxs.long(),
                                  device=self.device,
                                  non_blocking=True)
        # pack labels to remove padding from caption labels
        targets = torch.nn.utils.rnn.pack_padded_sequence(
            caps, cap_lens, batch_first=True).data


        if kwargs["tf"]:
            if kwargs["tune_instance"]:
                output = model(feats, feat_lens, caps, cap_lens, 
                               train_mode="tf")
            else:
                output = model(feats, feat_lens, caps, cap_lens, 
                               cap_idxs, train_mode="tf")
        else:
            if kwargs["tune_instance"]:
                output = model(feats, feat_lens, caps, cap_lens, 
                               train_mode="sample")
            else:
                output = model(feats, feat_lens, caps, cap_lens,
                               cap_idxs, train_mode="sample")

        packed_logits = torch.nn.utils.rnn.pack_padded_sequence(
            output["logits"], cap_lens, batch_first=True).data
        packed_logits = convert_tensor(packed_logits, device=self.device, non_blocking=True)
        output["packed_logits"] = packed_logits
        output["targets"] = targets

        return output

    def train(self, config, **kwargs):
        """Trains a model on the given configurations.
        :param config: A training configuration. Note that all parameters in the config can also be manually adjusted with --ARG=VALUE
        :param **kwargs: parameters to overwrite yaml config
        """
        from pycocoevalcap.cider.cider import Cider

        config_parameters = train_util.parse_config_or_kwargs(config, **kwargs)
        config_parameters["seed"] = self.seed
        outputdir = os.path.join(
            config_parameters["outputpath"], config_parameters["model"],
            "{}_{}".format(
                datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%m'),
                uuid.uuid1().hex))

        # Early init because of creating dir
        checkpoint_handler = ModelCheckpoint(
            outputdir,
            "run",
            n_saved=1,
            require_empty=False,
            create_dir=True,
            score_function=lambda engine: -engine.state.metrics["loss"],
            score_name="loss")

        logger = train_util.genlogger(os.path.join(outputdir, "train.log"))
        # print passed config parameters
        logger.info("Storing files in: {}".format(outputdir))
        train_util.pprint_dict(config_parameters, logger.info)

        zh = config_parameters["zh"]
        vocabulary = torch.load(config_parameters["vocab_file"])
        train_loader, val_loader, info = self._get_dataloaders(config_parameters, vocabulary)
        config_parameters["inputdim"] = info["inputdim"]
        val_key2refs = info["val_key2refs"]
        logger.info("<== Estimating Scaler ({}) ==>".format(info["scaler"].__class__.__name__))
        logger.info(
            "Stream: {} Input dimension: {} Vocab Size: {}".format(
                config_parameters["feature_stream"], info["inputdim"], len(vocabulary)))

        model = self._get_model(config_parameters, len(vocabulary))
        if "pretrained_word_embedding" in config_parameters:
            embeddings = np.load(config_parameters["pretrained_word_embedding"])
            model.load_word_embeddings(embeddings, tune=config_parameters["tune_word_embedding"], projection=True)
        model = model.to(self.device)
        train_util.pprint_dict(model, logger.info, formatter="pretty")
        optimizer = getattr(
            torch.optim, config_parameters["optimizer"]
        )(model.parameters(), **config_parameters["optimizer_args"])
        train_util.pprint_dict(optimizer, logger.info, formatter="pretty")


        criterion = torch.nn.CrossEntropyLoss().to(self.device)
        crtrn_imprvd = train_util.criterion_improver(config_parameters['improvecriterion'])
        tf_ratio = config_parameters["teacher_forcing_ratio"]
        tune_instance = config_parameters["tune_instance"]

        def _train_batch(engine, batch):
            model.train()
            tf = True if random.random() < tf_ratio else False
            with torch.enable_grad():
                optimizer.zero_grad()
                output = self._forward(model, batch, tf=tf, tune_instance=tune_instance)
                loss = criterion(output["packed_logits"], output["targets"])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                output["loss"] = loss.item()
                return output

        trainer = Engine(_train_batch)
        RunningAverage(output_transform=lambda x: x["loss"]).attach(trainer, "running_loss")
        pbar = ProgressBar(persist=False, ascii=True)
        pbar.attach(trainer, ["running_loss"])

        key2pred = {}

        def _inference(engine, batch):
            model.eval()
            keys = batch[3]
            with torch.no_grad():
                output = self._forward(model, batch, tf=config_parameters["teacher_forcing_on_validation"], tune_instance=tune_instance)
                seqs = output["seqs"].cpu().numpy()
                for (idx, seq) in enumerate(seqs):
                    if keys[idx] in key2pred:
                        continue
                    candidate = self._convert_idx2sentence(seq, vocabulary, zh)
                    key2pred[keys[idx]] = [candidate,]
                return output

        metrics = {
            "loss": Loss(criterion, output_transform=lambda x: (x["packed_logits"], x["targets"])),
            "accuracy": Accuracy(output_transform=lambda x: (x["packed_logits"], x["targets"])),
        }

        evaluator = Engine(_inference)

        def eval_cv(engine, key2pred, key2refs):
            scorer = Cider(zh=zh)
            score, scores = scorer.compute_score(key2refs, key2pred)
            engine.state.metrics["score"] = score
            key2pred.clear()

        evaluator.add_event_handler(
            Events.EPOCH_COMPLETED, eval_cv, key2pred, val_key2refs)

        for name, metric in metrics.items():
            metric.attach(trainer, name)
            metric.attach(evaluator, name)

        trainer.add_event_handler(
              Events.EPOCH_COMPLETED, train_util.log_results, evaluator, val_loader,
              logger.info, metrics.keys(), ["loss", "accuracy", "score"])

        evaluator.add_event_handler(
            Events.EPOCH_COMPLETED, train_util.save_model_on_improved, crtrn_imprvd,
            "score", {
                "model": model,
                "config": config_parameters,
                "scaler": info["scaler"]
        }, os.path.join(outputdir, "saved.pth"))

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **config_parameters["scheduler_args"])
        evaluator.add_event_handler(
            Events.EPOCH_COMPLETED, train_util.update_reduce_on_plateau,
            scheduler, "score")

        evaluator.add_event_handler(
            Events.EPOCH_COMPLETED, checkpoint_handler, {
                "model": model,
            }
        )

        # early_stop_handler = EarlyStopping(
            # patience=config_parameters["early_stop"],
            # score_function=lambda engine: engine.state.metrics["score"],
            # trainer=trainer)
        # evaluator.add_event_handler(Events.COMPLETED, early_stop_handler)

        trainer.run(train_loader, max_epochs=config_parameters["epochs"])
        return outputdir

    def evaluate(self, 
                 experiment_path: str,
                 kaldi_stream: str,
                 caption_file: str,
                 caption_embedding_path=None,
                 caption_output: str="eval_output.json",
                 score_output: str="scores.txt",
                 **kwargs):
        """kwargs: {'max_length': int, 'method': str, 'beam_size': int}"""

        dump = torch.load(os.path.join(experiment_path, "saved.pth"),
                          map_location="cpu")
        model = dump["model"]
        # Some scaler (sklearn standardscaler)
        scaler = dump["scaler"]
        # Also load previous training config
        config = dump["config"]
        vocabulary = torch.load(config["vocab_file"])
        zh = config["zh"]
        model = model.to(self.device)

        caption_df = pd.read_json(caption_file, dtype={"key": str})
        if zh:
            key2refs = caption_df.groupby("key")["tokens"].apply(list).to_dict()
        else:
            key2refs = caption_df.groupby("key")["caption"].apply(list).to_dict()

        model.eval()

        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.spice.spice import Spice
        from audiocaptioneval.sentbert.sentencebert import SentenceBert
        from utils.diverse_eval import diversity_evaluate

        key2preds_idx = {}
        f = open(os.path.join(experiment_path, score_output), "w")

        if caption_embedding_path is not None:
            key2ref_embeds = np.load(caption_embedding_path, allow_pickle=True)
        else:
            key2ref_embeds = None

        for instance_idx in range(config["model_args"]["num_instance"]):

            key2pred = {}

            def _sample(engine, batch):
                with torch.no_grad():
                    model.eval()
                    keys = batch[0]
                    instance_labels = torch.zeros(len(keys), config["model_args"]["num_instance"])
                    instance_labels[:, instance_idx] = 1
                    output = self._forward(model, batch, mode="sample", 
                                           instance_labels=instance_labels, **kwargs)
                    seqs = output["seqs"].cpu().numpy()

                    for idx, seq in enumerate(seqs):
                        caption = self._convert_idx2sentence(seq, vocabulary, zh)
                        key2pred[keys[idx]] = [caption,]
                        if instance_idx == 0:
                            key2preds_idx[keys[idx]] = [caption,]
                        else:
                            key2preds_idx[keys[idx]].append(caption)

            pbar = ProgressBar(persist=False, ascii=True)
            sampler = Engine(_sample)
            pbar.attach(sampler)
            dataset = SJTUDatasetEval(
                kaldi_stream=kaldi_stream,
                transform=scaler.transform)
            dataloader = torch.utils.data.DataLoader(
                dataset,
                shuffle=False,
                collate_fn=collate_fn((1,)),
                batch_size=32,
                num_workers=0)
            sampler.run(dataloader)

            f.write("--------------------\nidx:{}\n".format(instance_idx + 1))

            scorer = Bleu(n=4, zh=zh)
            score, scores = scorer.compute_score(key2refs, key2pred)
            for n in range(4):
                f.write("Bleu-{}: {:6.3f}\n".format(n + 1, score[n]))

            scorer = Rouge(zh=zh)
            score, scores = scorer.compute_score(key2refs, key2pred)
            f.write("ROUGE: {:6.3f}\n".format(score))

            scorer = Cider(zh=zh)
            score, scores = scorer.compute_score(key2refs, key2pred)
            f.write("CIDEr: {:6.3f}\n".format(score))

            if not zh:
                scorer = Meteor()
                score, scores = scorer.compute_score(key2refs, key2pred)
                f.write("Meteor: {:6.3f}\n".format(score))

                scorer = Spice()
                score, scores = scorer.compute_score(key2refs, key2pred)
                f.write("Spice: {:6.3f}\n".format(score))

            scorer = SentenceBert(zh=zh)
            if key2ref_embeds is not None:
                score, scores = scorer.compute_score(key2ref_embeds, key2pred)
            else:
                score, scores = scorer.compute_score(key2refs, key2pred)
            f.write("SentenceBert: {:6.3f}\n".format(score))

            df = []
            for key, pred in key2pred.items():
                df.append({
                    "tokens": pred[0] if zh else pred[0].split() 
                })
            df = pd.DataFrame(df)
            score = diversity_evaluate(df)
            f.write("Diversity: {:6.3f}\n".format(score))

        f.close()

        pred_df = []
        for key in key2preds_idx.keys():
            item = {
                "filename": key + ".wav"
            }
            for instance_idx in range(config["model_args"]["num_instance"]):
                item["caption_{}".format(instance_idx + 1)] = key2preds_idx[key][instance_idx]
            pred_df.append(item)
        pred_df = pd.DataFrame(pred_df)
        pred_df.to_json(os.path.join(experiment_path, caption_output))

    def evaluate_single(self, 
                        experiment_path: str,
                        kaldi_stream: str,
                        caption_file: str,
                        caption_embedding_path=None,
                        caption_output: str="eval_output.json",
                        score_output: str="scores.txt",
                        **kwargs):
        """kwargs: {'max_length': int, 'method': str, 'beam_size': int}"""

        dump = torch.load(os.path.join(experiment_path, "saved.pth"),
                          map_location="cpu")
        model = dump["model"]
        # Some scaler (sklearn standardscaler)
        scaler = dump["scaler"]
        # Also load previous training config
        config = dump["config"]
        vocabulary = torch.load(config["vocab_file"])
        zh = config["zh"]
        model = model.to(self.device)

        caption_df = pd.read_json(caption_file, dtype={"key": str})
        if zh:
            key2refs = caption_df.groupby("key")["tokens"].apply(list).to_dict()
        else:
            key2refs = caption_df.groupby("key")["caption"].apply(list).to_dict()

        model.eval()

        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.spice.spice import Spice
        from audiocaptioneval.sentbert.sentencebert import SentenceBert
        from utils.diverse_eval import diversity_evaluate

        key2preds_idx = {}
        f = open(os.path.join(experiment_path, score_output), "w")

        if caption_embedding_path is not None:
            key2ref_embeds = np.load(caption_embedding_path, allow_pickle=True)
        else:
            key2ref_embeds = None


        key2pred = {}

        def _sample(engine, batch):
            with torch.no_grad():
                model.eval()
                keys = batch[0]
                output = self._forward(model, batch, mode="sample", **kwargs)
                seqs = output["seqs"].cpu().numpy()

                for idx, seq in enumerate(seqs):
                    caption = self._convert_idx2sentence(seq, vocabulary, zh)
                    key2pred[keys[idx]] = [caption,]

        pbar = ProgressBar(persist=False, ascii=True)
        sampler = Engine(_sample)
        pbar.attach(sampler)
        dataset = SJTUDatasetEval(
            kaldi_stream=kaldi_stream,
            transform=scaler.transform)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            collate_fn=collate_fn((1,)),
            batch_size=32,
            num_workers=0)
        sampler.run(dataloader)

        scorer = Bleu(n=4, zh=zh)
        score, scores = scorer.compute_score(key2refs, key2pred)
        for n in range(4):
            f.write("Bleu-{}: {:6.3f}\n".format(n + 1, score[n]))

        scorer = Rouge(zh=zh)
        score, scores = scorer.compute_score(key2refs, key2pred)
        f.write("ROUGE: {:6.3f}\n".format(score))

        scorer = Cider(zh=zh)
        score, scores = scorer.compute_score(key2refs, key2pred)
        f.write("CIDEr: {:6.3f}\n".format(score))

        if not zh:
            scorer = Meteor()
            score, scores = scorer.compute_score(key2refs, key2pred)
            f.write("Meteor: {:6.3f}\n".format(score))

            scorer = Spice()
            score, scores = scorer.compute_score(key2refs, key2pred)
            f.write("Spice: {:6.3f}\n".format(score))

        scorer = SentenceBert(zh=zh)
        if key2ref_embeds is not None:
            score, scores = scorer.compute_score(key2ref_embeds, key2pred)
        else:
            score, scores = scorer.compute_score(key2refs, key2pred)
        f.write("SentenceBert: {:6.3f}\n".format(score))

        df = []
        for key, pred in key2pred.items():
            df.append({
                "tokens": pred[0] if zh else pred[0].split() 
            })
        df = pd.DataFrame(df)
        score = diversity_evaluate(df)
        f.write("Diversity: {:6.3f}\n".format(score))

        f.close()

        df.to_json(os.path.join(experiment_path, caption_output))

if __name__ == "__main__":
    fire.Fire(Runner)
