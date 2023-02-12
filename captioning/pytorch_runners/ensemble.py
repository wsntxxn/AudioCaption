# coding=utf-8
#!/usr/bin/env python3
import sys

from pathlib import Path

import json
import fire
import numpy as np
import torch
from tqdm import tqdm
from typing import List, Union, Dict

import captioning.models
import captioning.models.encoder
import captioning.models.decoder
import captioning.utils.train_util as train_util
from captioning.utils.build_vocab import Vocabulary
from captioning.pytorch_runners.base import BaseRunner
from pycocoevalcap.cider.cider import Cider
import captioning.datasets as dataset_module
import pandas as pd


class EnsembleRunner(BaseRunner):


    def _get_model(self, config, print_fn=sys.stdout.write):
        encoder_cfg = config["model"]["encoder"]
        encoder = train_util.init_obj(
            captioning.models.encoder,
            encoder_cfg)
        if "pretrained" in encoder_cfg:
            pretrained = encoder_cfg["pretrained"]
            train_util.load_pretrained_model(encoder,
                                             pretrained,
                                             print_fn)
        decoder_cfg = config["model"]["decoder"]
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
        model = train_util.init_obj(captioning.models, config["model"],
            encoder=encoder, decoder=decoder)

        if "modelwrapper" in config:
            model = train_util.init_obj(captioning.models,
                                        config["modelwrapper"],
                                        model=model)

        return model


    def resume_checkpoint(self, model, resume_path):
        ckpt = torch.load(resume_path, "cpu")
        train_util.load_pretrained_model(model, ckpt)
        if not hasattr(self, "vocabulary"):
            self.vocabulary = ckpt["vocabulary"]


    def _forward(self, models, batch):
        self.start_idx = models[0].start_idx
        self.end_idx = models[0].end_idx
        self.pad_idx = models[0].pad_idx
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.float().to(self.device)

        input_dict = {
            "mode": "inference",
            "specaug": False
        }
        input_dict.update(batch)

        encoder_outputs = [model.encoder(input_dict) for model in models]
        
        for v in encoder_outputs:
            v["mode"] = "inference"

        if self.eval_config["inference_args"]["method"] == "beam":
            output = self.beam_search(models, encoder_outputs)
        else:
            output = self.stepwise_forward(models, encoder_outputs)
        return output


    def stepwise_forward(self, models, inputs):
        assert len(models) == len(inputs), "size of models and data mismatch"
        n_models = len(models)
        inference_args = self.eval_config["inference_args"]
        first_item = next(iter(inputs[0].values()))
        ############################################
        # prepare output
        ############################################
        max_length = inference_args["max_length"]
        batch_size = first_item.size(0)
        output = {"seq": torch.full((batch_size, max_length), 
                self.end_idx, dtype=torch.long)}
        decoder_outputs = []
        for n in range(n_models):
            inputs[n]["max_length"] = max_length
            decoder_outputs.append(models[n].prepare_output(inputs[n]))
        
        for t in range(max_length):
            decoder_outputs_t = []
            for n in range(n_models):
                inputs[n]["t"] = t
                decoder_input = models[n].prepare_decoder_input(
                    inputs[n], decoder_outputs[n])
                output_n_t = models[n].decoder(decoder_input)
                ############################################
                # sample word
                ############################################
                logit_n_t = output_n_t["logit"]
                if logit_n_t.size(1) == 1:
                    logit_n_t = logit_n_t.squeeze(1)
                    embed_n_t = output_n_t["embed"].squeeze(1)
                elif logit_n_t.size(1) > 1:
                    logit_n_t = logit_n_t[:, -1, :]
                    embed_n_t = output_n_t["embed"][:, -1, :]
                else:
                    raise Exception("no logit output")
                output_n_t["logit"] = logit_n_t
                output_n_t["embed"] = embed_n_t
                decoder_outputs_t.append(output_n_t)
            logprobs_t = torch.stack(
                    [torch.log_softmax(output["logit"].squeeze(1), -1)
                        for output in decoder_outputs_t
                    ]).mean(dim=0)
            sampled = self.sample_next_word_with_logprob(
                logprobs_t,
                method=inference_args["method"],
                temp=inference_args["sample_word_temp"])
            output["seq"][:, t] = sampled["word"]
            ############################################
            # stepwise_process
            ############################################
            for n in range(n_models):
                decoder_outputs_t[n].update(sampled)
                decoder_outputs_t[n]["t"] = t
                models[n].stepwise_process_step(decoder_outputs[n],
                                                decoder_outputs_t[n])
            
        return output


    def beam_search(self, models, inputs):
        assert len(models) == len(inputs), "size of models and data mismatch"
        n_models = len(models)
        inference_args = self.eval_config["inference_args"]
        vocab_size = len(self.vocabulary)
        first_item = next(iter(inputs[0].values()))
        ############################################
        # prepare output
        ############################################
        max_length = inference_args["max_length"]
        batch_size = first_item.size(0)
        beam_size = inference_args["beam_size"]
        if inference_args["n_best"]:
            n_best_size = inference_args["n_best_size"]
            output = {"seq": torch.full((batch_size, n_best_size, max_length),
                                        self.end_idx, dtype=torch.long)}
        else:
            output = {"seq": torch.full((batch_size, max_length),
                                        self.end_idx, dtype=torch.long)}
        for i in range(batch_size):
            ############################################
            # prepare beam search decoder output for i-th sample
            ############################################
            output_i = {
                "topk_logprob": torch.zeros(beam_size).to(self.device),
                "seq": None,
                "prev_words_beam": None,
                "next_word": None,
                "done_beams": [],
            }
            ############################################
            # prepare beam search decoder input
            ############################################
            decoder_outputs = []
            for n in range(n_models):
                inputs[n].update({
                    "sample_idx": i,
                    "beam_size": beam_size,
                    "max_length": max_length
                })
                decoder_outputs.append(models[n].prepare_beamsearch_output(inputs[n]))
            for t in range(max_length):
                decoder_outputs_t = []
                for n in range(n_models):
                    inputs[n]["t"] = t
                    decoder_input = models[n].prepare_beamsearch_decoder_input(
                        inputs[n], decoder_outputs[n])
                    output_n_t = models[n].decoder(decoder_input)
                    logit_n_t = output_n_t["logit"]
                    if logit_n_t.size(1) == 1:
                        logit_n_t = logit_n_t.squeeze(1)
                    elif logit_n_t.size(1) > 1:
                        logit_n_t = logit_n_t[:, -1, :]
                    else:
                        raise Exception("no logit output")
                    output_n_t["logit"] = logit_n_t
                    output_n_t["t"] = t
                    decoder_outputs_t.append(output_n_t)
                logprob_t = torch.stack(
                    [
                        torch.log_softmax(output["logit"], -1)
                            for output in decoder_outputs_t
                    ]).mean(dim=0)
                #######################################
                # merge with previous beam and select the current max prob beam
                #######################################
                logprob_t = torch.log_softmax(logprob_t / inference_args["beam_temp"], dim=1)
                logprob_t = output_i["topk_logprob"].unsqueeze(1) + logprob_t
                if t == 0: # for the first step, all k seq will have the same probs
                    topk_logprob, topk_words = logprob_t[0].topk(
                        beam_size, 0, True, True)
                else: # unroll and find top logprob, and their unrolled indices
                    topk_logprob, topk_words = logprob_t.view(-1).topk(
                        beam_size, 0, True, True)
                topk_words = topk_words.cpu()
                output_i["topk_logprob"] = topk_logprob
                output_i["prev_words_beam"] = torch.div(topk_words, vocab_size,
                                                        rounding_mode='trunc')
                output_i["next_word"] = topk_words % vocab_size  # [beam_size,]
                if t == 0:
                    output_i["seq"] = output_i["next_word"].unsqueeze(1)
                else:
                    output_i["seq"] = torch.cat([
                        output_i["seq"][output_i["prev_words_beam"]],
                        output_i["next_word"].unsqueeze(1)], dim=1)
                # add finished beams to results
                is_end = output_i["next_word"] == self.end_idx
                if t == max_length - 1:
                    is_end.fill_(1)
                for beam_idx in range(beam_size):
                    if is_end[beam_idx]:
                        final_beam = {
                            "seq": output_i["seq"][beam_idx].clone(),
                            "score": output_i["topk_logprob"][beam_idx].item()
                        }
                        final_beam["score"] = final_beam["score"] / (t + 1)
                        output_i["done_beams"].append(final_beam)
                output_i["topk_logprob"][is_end] -= 1000
                #######################################
                # end of merging
                #######################################
                for n in range(n_models):
                    decoder_outputs_t[n].update(output_i)
                    decoder_outputs[n].update(output_i)
                    decoder_outputs_t[n]["t"] = t
                    models[n].beamsearch_process_step(
                        decoder_outputs[n],
                        decoder_outputs_t[n])

            #######################################
            # post process: output_i -> output
            #######################################
            done_beams = sorted(output_i["done_beams"], key=lambda x: -x["score"])
            if inference_args["n_best"]:
                done_beams = done_beams[:inference_args["n_best_size"]]
                for out_idx, done_beam in enumerate(done_beams):
                    seq = done_beam["seq"]
                    output["seq"][i][out_idx, :len(seq)] = seq
            else:
                seq = done_beams[0]["seq"]
                output["seq"][i][:len(seq)] = seq

        return output
    

    def predict(self,
                eval_config: Union[str, Dict],
                return_pred: bool = False,
                dump_output: bool = True,
                **kwargs):

        # note: all models should use the same vocabulary, or a mapping process is needed
       
        if not isinstance(eval_config, dict):
            eval_config = train_util.parse_config_or_kwargs(eval_config, **kwargs)

        self.eval_config = eval_config
        
        models = []
        for exp_path in eval_config["experiment_path"]:
            exp_path = Path(exp_path)
            config = train_util.parse_config_or_kwargs(
                exp_path.parent / "config.yaml")
            model = self._get_model(config)
            self.resume_checkpoint(model, exp_path)
            model = model.to(self.device)
            model.eval()
            models.append(model)
        
        dataset_config = eval_config["data"]["test"]["dataset"]
        dataset = getattr(dataset_module, dataset_config["type"])(
            **dataset_config["args"])
        collate_config = eval_config["data"]["test"]["collate_fn"]
        collate_fn = getattr(dataset_module, collate_config["type"])(
            **collate_config["args"])
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset, collate_fn=collate_fn,
            **eval_config["data"]["test"]["dataloader_args"])
        
        key2pred = {}
        
        with torch.no_grad(), tqdm(total=len(dataloader),
            ncols=100, ascii=True, leave=False) as pbar:
            for batch in dataloader:
                output = self._forward(models, batch)
                keys = batch["audio_id"]
                seqs = output["seq"].cpu().numpy()
                for (idx, seq) in enumerate(seqs):
                    if len(seq.shape) > 1:
                        seq = seq[0]
                    candidate = self._decode_to_sentence(seq)
                    key2pred[keys[idx]] = [candidate,]
                pbar.update()

        pred_data = []
        for key, pred in key2pred.items():
            pred_data.append({
                "filename": key,
                "tokens": pred[0]
            })

        if dump_output:
            output_path = Path(eval_config["output_path"])
            output_file = output_path / eval_config["caption_output"]
            if not output_file.parent.exists():
                output_file.parent.mkdir(parents=True)
            json.dump(
                {"predictions": pred_data}, open(output_file , "w"), indent=4)

        if return_pred:
            return key2pred


    def evaluate(self,
                 eval_config: str,
                 **kwargs):
        eval_config = train_util.parse_config_or_kwargs(eval_config, **kwargs)
        key2pred = self.predict(eval_config,
                                return_pred=True)
        zh = eval_config["zh"]
        caption_file = eval_config["data"]["test"]["caption"]
        captions = json.load(open(caption_file, "r"))["audios"]
        key2refs = {}
        for audio_idx in range(len(captions)):
            audio_id = captions[audio_idx]["audio_id"]
            key2refs[audio_id] = []
            for caption in captions[audio_idx]["captions"]:
                key2refs[audio_id].append(caption["tokens" if zh else "caption"])

        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.spice.spice import Spice
        from fense.fense import Fense

        scorers = [Bleu(n=4), Rouge(), Cider()]
        if not zh:
            scorers.append(Meteor())
            scorers.append(Spice())
            scorers.append(Fense())
        result = self._eval_prediction(key2refs, key2pred, scorers)

        output_filename = Path(eval_config["output_path"]) / eval_config["score_output"]
        if not output_filename.parent.exists():
            output_filename.parent.mkdir(parents=True)
        with open(output_filename, "w") as f:
            spider = 0
            for name, score in result.items():
                if name == "Bleu":
                    for n in range(4):
                        f.write("Bleu-{}: {:6.3f}\n".format(n + 1, score[n]))
                else:
                    f.write("{}: {:6.3f}\n".format(name, score))
                    if name in ["CIDEr", "SPICE"]:
                        spider += score
            if not zh:
                f.write("SPIDEr: {:6.3f}\n".format(spider / 2))


    def dcase_predict(self,
                 eval_config: str,
                 **kwargs):
        eval_config = train_util.parse_config_or_kwargs(eval_config, **kwargs)
        key2pred = self.predict(eval_config,
                                return_pred=True,
                                dump_output=False)

        pred_data = []
        for key, pred in key2pred.items():
            pred_data.append({
                "file_name": key,
                "caption_predicted": pred[0],
            })
        pred_df = pd.DataFrame(pred_data)
        pred_df.to_csv(str(Path(eval_config['output_path']) / eval_config["dcase_output"]   ), index=False)


    def sample_next_word_with_logprob(self, logprob, method, temp):
        """Sample the next word, given probs output by the decoder"""
        if method == "greedy":
            sampled_logprob, word = torch.max(logprob.detach(), 1)
        elif method == "gumbel":
            def sample_gumbel(shape, eps=1e-20):
                U = torch.rand(shape).to(logprob.device)
                return -torch.log(-torch.log(U + eps) + eps)
            def gumbel_softmax_sample(logit, temperature):
                y = logit + sample_gumbel(logit.size())
                return torch.log_softmax(y / temperature, dim=-1)
            _logprob = gumbel_softmax_sample(logprob, temp)
            _, word = torch.max(_logprob.data, 1)
            sampled_logprob = logprob.gather(1, word.unsqueeze(-1))
        else:
            logprob = logprob / temp
            if method.startswith("top"):
                top_num = float(method[3:])
                if 0 < top_num < 1: # top-p sampling
                    probs = torch.softmax(logprob, dim=1)
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=1)
                    _cumsum = sorted_probs.cumsum(1)
                    mask = _cumsum < top_num
                    mask = torch.cat([torch.ones_like(mask[:,:1]), mask[:,:-1]], 1)
                    sorted_probs = sorted_probs * mask.to(sorted_probs)
                    sorted_probs = sorted_probs / sorted_probs.sum(1, keepdim=True)
                    logprob.scatter_(1, sorted_indices, sorted_probs.log())
                else: # top-k sampling
                    k = int(top_num)
                    tmp = torch.empty_like(logprob).fill_(float('-inf'))
                    topk, indices = torch.topk(logprob, k, dim=1)
                    tmp = tmp.scatter(1, indices, topk)
                    logprob = tmp
            word = torch.distributions.Categorical(logits=logprob.detach()).sample()
            sampled_logprob = logprob.gather(1, word.unsqueeze(-1)).squeeze(1)
        word = word.detach().long()
        # sampled_logprob: [N,], word: [N,]
        return {"word": word, "probs": sampled_logprob}


if __name__ == "__main__":
    fire.Fire(EnsembleRunner)
