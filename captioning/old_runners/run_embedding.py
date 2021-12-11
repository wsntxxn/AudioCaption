# coding=utf-8
#!/usr/bin/env python3
import os
import sys
import datetime
import random
import uuid
import re
from pprint import pformat
from contextlib import contextmanager

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
sys.path.append("/mnt/lustre/sjtu/home/xnx98/utils")
import models
import utils.train_util as train_util
from utils.build_vocab import Vocabulary
from datasets.SJTUDataSet import SJTUSentenceDataset, collate_fn
from runners.base_runner import BaseRunner
import utils.score_util as score_util

class Runner(BaseRunner):

    @staticmethod
    def _get_dataloaders(config, vocabulary):
        caption_df = pd.read_json(config["caption_file"])
        sentence_embedding = np.load(config["sentence_embedding"], allow_pickle=True)
        scaler = getattr(
            pre, config["scaler"])(
            **config["scaler_args"])
        inputdim = -1

        for batch in tqdm(
            torch.utils.data.DataLoader(
                SJTUSentenceDataset(
                    kaldi_stream=config["feature_stream"],
                    caption_df=caption_df,
                    vocabulary=vocabulary,
                    sentence_embedding=sentence_embedding,
                ),
                collate_fn=collate_fn([0, 1]),
                **config["dataloader_args"]
            ),
            ascii=True
        ):
            feat = batch[0]
            feat = feat.reshape(-1, feat.shape[-1])
            scaler.partial_fit(feat)
            inputdim = feat.shape[-1]
        assert inputdim > 0, "Reading inputstream failed"
        augments = train_util.parse_augments(config["augments"])
        cv_keys = np.random.choice(
            caption_df["key"].unique(),
            int(len(caption_df["key"].unique()) * (1 - config["train_percent"] / 100.)),
            replace=False
        )
        cv_df = caption_df[caption_df["key"].apply(lambda x: x in cv_keys)]
        train_df = caption_df[~caption_df.index.isin(cv_df.index)]

        trainloader = torch.utils.data.DataLoader(
            SJTUSentenceDataset(
                kaldi_stream=config["feature_stream"],
                caption_df=train_df,
                vocabulary=vocabulary,
                sentence_embedding=sentence_embedding,
                transform=[augments]
            ),
            shuffle=True,
            collate_fn=collate_fn([0, 1]),
            **config["dataloader_args"]
        )

        if config["zh"]:
            cv_key2refs = cv_df.groupby("key")["tokens"].apply(list).to_dict()
        else:
            cv_key2refs = cv_df.groupby("key")["caption"].apply(list).to_dict()

        cvloader = torch.utils.data.DataLoader(
            SJTUSentenceDataset(
                kaldi_stream=config["feature_stream"],
                caption_df=cv_df,
                vocabulary=vocabulary,
                sentence_embedding=sentence_embedding,
            ),
            shuffle=False,
            collate_fn=collate_fn([0, 1]),
            # drop_last=True,
            **config["dataloader_args"]
        )
        return trainloader, cvloader, {"cv_key2refs": cv_key2refs, "inputdim": inputdim, "scaler": scaler}

    @staticmethod
    def _get_model(config, vocab_size):
        embed_size = config["model_args"]["embed_size"]
        encodermodel = getattr(
            models.encoder, config["encodermodel"])(
            inputdim=config["inputdim"],
            embed_size=embed_size,
            **config["encodermodel_args"])
        decodermodel = getattr(
            models.decoder, config["decodermodel"])(
            vocab_size=vocab_size,
            embed_size=embed_size,
            **config["decodermodel_args"])
        model = getattr(models.WordModel, config["model"])(
            encodermodel, decodermodel, **config["model_args"])

        if "pretrained_decoder" in config:
            pretrained_decoder = torch.load(
                config["pretrained_decoder"],
                map_location="cpu")["model"]
            model.load_state_dict(pretrained_decoder.state_dict(), strict=False)
        if "pretrained_encoder" in config:
            pretrained_encoder = torch.load(
                config["pretrained_encoder"],
                map_location="cpu")["model"]
            model.encoder.load_state_dict(pretrained_encoder.state_dict(), strict=False)
        return model

    def _forward(self, model, batch, tf=None, mode="train", **kwargs):
        assert mode in ("train", "sample")

        if mode == "sample":
            feats = batch[1]
            feat_lens = batch[-1]

            feats = convert_tensor(feats.float(),
                                   device=self.device,
                                   non_blocking=True)
            output = model(feats, feat_lens, mode="sample", **kwargs)
            return output

        # mode is "train"
        assert tf is not None

        feats = batch[0]
        caps = batch[1]
        sent_embeds = batch[2]
        keys= batch[3]
        feat_lens = batch[-2]
        cap_lens = batch[-1]
        feats = convert_tensor(feats.float(),
                               device=self.device,
                               non_blocking=True)
        caps = convert_tensor(caps.long(),
                              device=self.device,
                              non_blocking=True)
        sent_embeds = convert_tensor(sent_embeds.float(),
                                     device=self.device,
                                     non_blocking=True)
        # pack labels to remove padding from caption labels
        targets = torch.nn.utils.rnn.pack_padded_sequence(
            caps, cap_lens, batch_first=True).data

        output = {}

        if tf:
            output = model(feats, feat_lens, caps, cap_lens, mode="forward")
        else:
            output = model(feats, feat_lens, mode="sample", max_length=max(cap_lens))
            probs = torch.nn.utils.rnn.pack_padded_sequence(
                output["probs"], cap_lens, batch_first=True).data
            probs = convert_tensor(probs, device=self.device, non_blocking=True)
            output["probs"] = probs

        batch_size = feats.shape[0]
        if "negative_sampling" in kwargs and kwargs["negative_sampling"] and len(set(keys)) > 1:
            embedding_targets = torch.empty_like(sent_embeds).to(self.device)
            embedding_labels = torch.ones(batch_size).to(self.device)
            embedding_targets[:batch_size // 3, :] = sent_embeds[:batch_size // 3, :]
            for i in range(batch_size // 3, 2 * batch_size // 3):
                key = keys[i]
                neg_idx = np.random.choice(range(batch_size), 1)[0]
                while keys[neg_idx] == key:
                    neg_idx = np.random.choice(range(batch_size), 1)[0]
                embedding_targets[i, :] = sent_embeds[neg_idx, :]
            for i in range(2 * batch_size // 3, batch_size):
                key = keys[i]
                neg_idx = np.random.choice(range(batch_size), 1)[0]
                while keys[neg_idx] == key:
                    neg_idx = np.random.choice(range(batch_size), 1)[0]
                embedding_targets[i, :] = output["embedding_output"][neg_idx, :]
            embedding_labels[batch_size // 3:] = -1
            output["embedding_labels"] = embedding_labels.to(self.device)
            output["embedding_targets"] = embedding_targets
        else:
            output["embedding_labels"] = torch.ones(batch_size).to(self.device)
            output["embedding_targets"] = sent_embeds

        output["word_targets"] = targets

        return output

    def train(self, config, **kwargs):
        """Trains a model on the given configurations.
        :param config: A training configuration. Note that all parameters in the config can also be manually adjusted with --ARG=VALUE
        :param **kwargs: parameters to overwrite yaml config
        """

        from pycocoevalcap.cider.cider import Cider

        config_parameters = train_util.parse_config_or_kwargs(config, **kwargs)
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

        vocabulary = torch.load(config_parameters["vocab_file"])
        trainloader, cvloader, info = self._get_dataloaders(config_parameters, vocabulary)

        cv_key2refs = info["cv_key2refs"]
        config_parameters["inputdim"] = info["inputdim"]
        zh = config_parameters["zh"]

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

        XE_criterion = torch.nn.CrossEntropyLoss().to(self.device)
        embedding_criterion = getattr(torch.nn, config_parameters["embedding_loss"])(**config_parameters["embedding_loss_args"]).to(self.device)
        embedding_loss_ratio = config_parameters["embedding_loss_ratio"]
        crtrn_imprvd = train_util.criterion_improver(config_parameters['improvecriterion'])
        tf_ratio = config_parameters["teacher_forcing_ratio"]

        def _train_batch(engine, batch):
            model.train()
            tf = True if random.random() < tf_ratio else False
            with torch.enable_grad():
                optimizer.zero_grad()
                output = self._forward(model, batch, tf=tf, negative_sampling=True)
                XE_loss = XE_criterion(output["probs"], output["word_targets"])
                if config_parameters["embedding_loss"] == "CosineEmbeddingLoss":
                    embedding_loss = embedding_criterion(
                        output["embedding_output"], 
                        output["embedding_targets"], 
                        output["embedding_labels"]
                    )
                else:
                    embedding_loss = embedding_criterion(
                        output["embedding_output"], output["embedding_targets"]
                    )
                loss = XE_loss + embedding_loss_ratio * embedding_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                output["XE_loss"] = XE_loss.item()
                output["embedding_loss"] = embedding_loss.item()
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
                output = self._forward(model, batch, tf=config_parameters["teacher_forcing_on_validation"], negative_sampling=True)
                XE_loss = XE_criterion(output["probs"], output["word_targets"])
                if config_parameters["embedding_loss"] == "CosineEmbeddingLoss":
                    embedding_loss = embedding_criterion(
                        output["embedding_output"], 
                        output["embedding_targets"], 
                        output["embedding_labels"]
                    )
                else:
                    embedding_loss = embedding_criterion(
                        output["embedding_output"], output["embedding_targets"]
                    )
                loss = XE_loss + embedding_loss_ratio * embedding_loss
                output["XE_loss"] = XE_loss.item()
                output["embedding_loss"] = embedding_loss.item()
                output["loss"] = loss.item()
                seqs = output["seqs"].cpu().numpy()
                for (idx, seq) in enumerate(seqs):
                    if keys[idx] in key2pred:
                        continue
                    candidate = self._convert_idx2sentence(seq, vocabulary, zh)
                    key2pred[keys[idx]] = [candidate,]
                return output

        metrics = {
            "loss": Average(output_transform=lambda x: x["loss"]),
            "XE_loss": Average(output_transform=lambda x: x["XE_loss"]),
            "embedding_loss": Average(output_transform=lambda x: x["embedding_loss"]),
            "accuracy": Accuracy(output_transform=lambda x: (x["probs"], x["word_targets"]))
        }

        evaluator = Engine(_inference)

        for name, metric in metrics.items():
            metric.attach(trainer, name)
            metric.attach(evaluator, name)

        def eval_cv(engine, key2pred, key2refs):
            scorer = Cider(zh=zh)
            score, scores = scorer.compute_score(key2refs, key2pred)
            engine.state.metrics["score"] = score
            key2pred.clear()

        evaluator.add_event_handler(
            Events.EPOCH_COMPLETED, eval_cv, key2pred, cv_key2refs)

        trainer.add_event_handler(
              Events.EPOCH_COMPLETED, train_util.log_results, evaluator, cvloader,
              logger.info, ["XE_loss", "embedding_loss"], ["XE_loss", "embedding_loss", "accuracy", "score"])

        evaluator.add_event_handler(
            Events.EPOCH_COMPLETED, train_util.save_model_on_improved, crtrn_imprvd,
            "loss", {
                "model": model,
                "config": config_parameters,
                "scaler": info["scaler"]
        }, os.path.join(outputdir, "saved.pth"))

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **config_parameters["scheduler_args"])

        # evaluator.add_event_handler(
            # Events.EPOCH_COMPLETED, train_util.update_reduce_on_plateau,
            # scheduler, "score")

        # evaluator.add_event_handler(
            # Events.EPOCH_COMPLETED, checkpoint_handler, {
            # "model": model,
            # }
        # )

        early_stop_handler = EarlyStopping(
            patience=config_parameters["early_stop"],
            score_function=lambda engine: engine.state.metrics["score"],
            trainer=trainer)
        evaluator.add_event_handler(Events.COMPLETED, early_stop_handler)

        trainer.run(trainloader, max_epochs=config_parameters["epochs"])
        return outputdir

    def sample(self,
               experiment_path: str,
               kaldi_stream,
               kaldi_scp,
               max_length=None,
               output: str="output_word.txt"):
        """Generate captions given experiment model"""
        import tableprint as tp
        from datasets.SJTUDataSet import SJTUDatasetEval, collate_fn

        dump = torch.load(os.path.join(experiment_path, "saved.pth"),
                          map_location="cpu")
        model = dump["model"]
        # Some scaler (sklearn standardscaler)
        scaler = dump["scaler"]
        # Also load previous training config
        config = dump["config"]
        vocabulary = torch.load(config["vocab_file"])
        model = model.to(self.device)
        dataset = SJTUDatasetEval(
            kaldi_stream=kaldi_stream,
            kaldi_scp=kaldi_scp,
            transform=scaler.transform)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            collate_fn=collate_fn((1,)),
            batch_size=16,
            num_workers=0)

        if max_length is None:
            max_length = model.max_length
        width_length = max_length * 4
        pbar = ProgressBar(persist=False, ascii=True)
        writer = open(os.path.join(experiment_path, output), "w")
        writer.write(
            tp.header(
                ["InputUtterance", "Output Sentence"], width=[len("InputUtterance"), width_length]))
        writer.write('\n')

        sentences = []
        def _sample(engine, batch):
            # batch: [ids, feats, feat_lens]
            with torch.no_grad():
                model.eval()
                ids = batch[0]
                # sampled = self._forward(model, batch, mode="sample", method="beam", beam_size=3, max_length=max_length)
                sampled = self._forward(model, batch, mode="sample", method="greedy", max_length=max_length)
                seqs = sampled["seqs"].cpu().numpy()
                for idx, seq in enumerate(seqs):
                    caption = []
                    for word_id in seq:
                        word = vocabulary.idx2word[word_id]
                        caption.append(word)
                        if word == "<end>":
                            break
                    sentence = " ".join(caption)
                    writer.write(tp.row([ids[idx], sentence], width=[len("InputUtterance"), width_length]) + "\n")
                    sentences.append(sentence)

        sample_engine = Engine(_sample)
        pbar.attach(sample_engine)
        sample_engine.run(dataloader)
        writer.write(tp.bottom(2, width=[len("InputUtterance"), width_length]) + "\n")
        writer.write("Unique sentence number: {}\n".format(len(set(sentences))))
        writer.close()

    def evaluate(self,
                 experiment_path: str,
                 kaldi_stream: str,
                 dev_embedding_path: str,
                 caption_file: str,
                 eval_embedding_path=None,
                 caption_output: str = "eval_output.json",
                 score_output: str = "scores.txt",
                 **kwargs):

        dump = torch.load(os.path.join(experiment_path, "saved.pth"),
                          map_location="cpu")
        model = dump["model"]
        # Also load previous training config
        config = dump["config"]
        zh = config["zh"]
        vocabulary = torch.load(config["vocab_file"])
        sentence_embedding = np.load(dev_embedding_path, allow_pickle=True)
        model = model.to(self.device)

        caption_df = pd.read_json(caption_file, dtype={"key": str})
        dataset = SJTUSentenceDataset(
            kaldi_stream=kaldi_stream,
            caption_df=caption_df,
            vocabulary=vocabulary,
            sentence_embedding=sentence_embedding)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            collate_fn=collate_fn([0, 1]),
            **config["dataloader_args"]
        )

        if zh:
            key2refs = caption_df.groupby("key")["tokens"].apply(list).to_dict()
        else:
            key2refs = caption_df.groupby("key")["caption"].apply(list).to_dict()

        model.eval()

        key2pred = {}

        def _sample(engine, batch):
            with torch.no_grad():
                model.eval()
                keys = batch[3]
                output = self._forward(
                    model, batch, None, mode="sample", **kwargs)
                seqs = output["seqs"].cpu().numpy()

                for idx, seq in enumerate(seqs):
                    if keys[idx] in key2pred.keys():
                        continue
                    candidate = self._convert_idx2sentence(seq, vocabulary, zh)
                    key2pred[keys[idx]] = [candidate,]

        pbar = ProgressBar(persist=False, ascii=True)
        sampler = Engine(_sample)
        pbar.attach(sampler)
        sampler.run(dataloader)

        pred_df = []
        for key, pred in key2pred.items():
            pred_df.append({
                "filename": key + ".wav",
                "caption": "".join(pred[0]) if zh else pred[0],
                "tokens": pred[0] if zh else pred[0].split() 
            })
        pred_df = pd.DataFrame(pred_df)
        pred_df.to_json(os.path.join(experiment_path, caption_output))

        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.spice.spice import Spice

        f = open(os.path.join(experiment_path, score_output), "w")

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

        from audiocaptioneval.sentbert.sentencebert import SentenceBert
        scorer = SentenceBert(zh=zh)
        if eval_embedding_path is not None:
            key2ref_embeds = np.load(eval_embedding_path, allow_pickle=True)
            score, scores = scorer.compute_score(key2ref_embeds, key2pred)
        else:
            score, scores = scorer.compute_score(key2refs, key2pred)
        f.write("SentenceBert: {:6.3f}\n".format(score))

        from utils.diverse_eval import diversity_evaluate
        score = diversity_evaluate(pred_df)
        f.write("Diversity: {:6.3f}\n".format(score))

        f.close()


if __name__ == "__main__":
    fire.Fire(Runner)
