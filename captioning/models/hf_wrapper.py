from typing import Dict, Callable, Union, List
import random
import math
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchaudio import transforms
from torchlibrosa import SpecAugmentation
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch import utils as efficientnet_utils
from einops import rearrange, reduce
from torch.hub import load_state_dict_from_url
from transformers import PretrainedConfig, PreTrainedModel


def init(m, method="kaiming"):
    if isinstance(m, (nn.Conv2d, nn.Conv1d)):
        if method == "kaiming":
            nn.init.kaiming_uniform_(m.weight)
        elif method == "xavier":
            nn.init.xavier_uniform_(m.weight)
        else:
            raise Exception(f"initialization method {method} not supported")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        if method == "kaiming":
            nn.init.kaiming_uniform_(m.weight)
        elif method == "xavier":
            nn.init.xavier_uniform_(m.weight)
        else:
            raise Exception(f"initialization method {method} not supported")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        if method == "kaiming":
            nn.init.kaiming_uniform_(m.weight)
        elif method == "xavier":
            nn.init.xavier_uniform_(m.weight)
        else:
            raise Exception(f"initialization method {method} not supported")

class _EffiNet(nn.Module):
    """A proxy for efficient net models"""
    def __init__(self,
                 blocks_args=None,
                 global_params=None,
                 ) -> None:
        super().__init__()
        self.eff_net = EfficientNet(blocks_args=blocks_args,
                                    global_params=global_params)
        

    def forward(self, x: torch.Tensor): 
        x = rearrange(x, 'b f t -> b 1 f t')
        x = self.eff_net.extract_features(x)
        return reduce(x, 'b c f t -> b t c', 'mean')


def get_effb2_model(pretrained=True) -> _EffiNet:
    blocks_args, global_params = efficientnet_utils.get_model_params(
        'efficientnet-b2', {'include_top': False})
    model = _EffiNet(blocks_args=blocks_args,
                     global_params=global_params)
    model.eff_net._change_in_channels(1)
    if pretrained:
        state_dict = load_state_dict_from_url(
            'https://github.com/richermans/HEAR2021_EfficientLatent/releases/download/v0.0.1/effb2.pt',
            progress=True)
        del_keys = [key for key in state_dict if key.startswith("front_end")]
        for key in del_keys:
            del state_dict[key]
        model.eff_net.load_state_dict(state_dict)
    return model

def merge_load_state_dict(state_dict,
                          model: torch.nn.Module,
                          output_fn: Callable = sys.stdout.write):
    model_dict = model.state_dict()
    pretrained_dict = {}
    mismatch_keys = []
    for key, value in state_dict.items():
        if key in model_dict and model_dict[key].shape == value.shape:
            pretrained_dict[key] = value
        else:
            mismatch_keys.append(key)
    output_fn(f"Loading pre-trained model, with mismatched keys {mismatch_keys}\n")
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=True)
    return pretrained_dict.keys()


class EfficientNetB2(nn.Module):

    def __init__(self,
                 n_mels: int = 64,
                 win_length: int = 32,
                 hop_length: int = 10,
                 f_min: int = 0,
                 pretrained: bool = False,
                 freeze: bool = False,):
        super().__init__()
        sample_rate = 16000
        self.melspec_extractor = transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=win_length * sample_rate // 1000,
            win_length=win_length * sample_rate // 1000,
            hop_length=hop_length * sample_rate // 1000,
            f_min=f_min,
            n_mels=n_mels,
        )
        self.hop_length = 10 * sample_rate // 1000
        self.db_transform = transforms.AmplitudeToDB(top_db=120)
        self.spec_augmenter = SpecAugmentation(time_drop_width=64,
            time_stripes_num=2, freq_drop_width=8, freq_stripes_num=2)
        self.backbone = get_effb2_model(pretrained=pretrained)
        self.fc_emb_size = self.backbone.eff_net._conv_head.out_channels
        self.downsample_ratio = 32
        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input_dict):
        
        waveform = input_dict["wav"]
        wave_length = input_dict["wav_len"]
        specaug = input_dict["specaug"]
        x = self.melspec_extractor(waveform)
        x = self.db_transform(x)    # (batch_size, mel_bins, time_steps)
        
        x = rearrange(x, 'b f t -> b 1 t f')
        if self.training and specaug:
            x = self.spec_augmenter(x)
        x = rearrange(x, 'b 1 t f -> b f t')
        
        x = self.backbone(x)
        attn_emb = x

        wave_length = torch.as_tensor(wave_length)
        feat_length = torch.div(wave_length, self.hop_length,
            rounding_mode="floor") + 1
        feat_length = torch.div(feat_length, self.downsample_ratio,
            rounding_mode="floor")
        fc_emb = mean_with_lens(attn_emb, feat_length)
        
        output_dict = {
            'fc_emb': fc_emb,
            'attn_emb': attn_emb,
            'attn_emb_len': feat_length
        }
        return output_dict


def generate_length_mask(lens, max_length=None):
    lens = torch.as_tensor(lens)
    N = lens.size(0)
    if max_length is None:
        max_length = max(lens)
        if isinstance(max_length, torch.Tensor):
            max_length = max_length.item()
    idxs = torch.arange(max_length).repeat(N).view(N, max_length)
    idxs = idxs.to(lens.device)
    mask = (idxs < lens.view(-1, 1))
    return mask

def mean_with_lens(features, lens):
    """
    features: [N, T, ...] (assume the second dimension represents length)
    lens: [N,]
    """
    lens = torch.as_tensor(lens)
    if max(lens) != features.size(1):
        max_length = features.size(1)
        mask = generate_length_mask(lens, max_length)
    else:
        mask = generate_length_mask(lens)
    mask = mask.to(features.device) # [N, T]

    while mask.ndim < features.ndim:
        mask = mask.unsqueeze(-1)
    feature_mean = features * mask
    feature_mean = feature_mean.sum(1)
    while lens.ndim < feature_mean.ndim:
        lens = lens.unsqueeze(1)
    feature_mean = feature_mean / lens.to(features.device)
    # feature_mean = features * mask.unsqueeze(-1)
    # feature_mean = feature_mean.sum(1) / lens.unsqueeze(1).to(features.device)
    return feature_mean

def max_with_lens(features, lens):
    """
    features: [N, T, ...] (assume the second dimension represents length)
    lens: [N,]
    """
    lens = torch.as_tensor(lens)
    if max(lens) != features.size(1):
        max_length = features.size(1)
        mask = generate_length_mask(lens, max_length)
    else:
        mask = generate_length_mask(lens)
    mask = mask.to(features.device) # [N, T]

    feature_max = features.clone()
    feature_max[~mask] = float("-inf")
    feature_max, _ = feature_max.max(1)
    return feature_max

def repeat_tensor(x, n):
    return x.unsqueeze(0).repeat(n, *([1] * len(x.shape)))


class CaptionMetaMixin:
    pad_idx = 0
    start_idx = 1
    end_idx = 2
    max_length = 20

    @classmethod
    def set_index(cls, start_idx, end_idx, pad_idx):
        cls.start_idx = start_idx
        cls.end_idx = end_idx
        cls.pad_idx = pad_idx


class CaptionModel(nn.Module, CaptionMetaMixin):
    """
    Encoder-decoder captioning model.
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module, **kwargs):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab_size = decoder.vocab_size
        self.train_forward_keys = ["cap", "cap_len", "ss_ratio"]
        self.inference_forward_keys = ["sample_method", "max_length", "temp"]
        freeze_encoder = kwargs.get("freeze_encoder", False)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.check_decoder_compatibility()

    def check_decoder_compatibility(self):
        compatible_decoders = [x.__class__.__name__ for x in self.compatible_decoders]
        assert isinstance(self.decoder, self.compatible_decoders), \
            f"{self.decoder.__class__.__name__} is incompatible with " \
            f"{self.__class__.__name__}, please use decoder in {compatible_decoders} "

    def forward(self, input_dict: Dict):
        """
        input_dict: {
            (required)
            mode: train/inference,
            [spec, spec_len],
            [fc],
            [attn, attn_len],
            [wav, wav_len],
            [sample_method: greedy],
            [temp: 1.0] (in case of no teacher forcing)

            (optional, mode=train)
            cap,
            cap_len,
            ss_ratio,

            (optional, mode=inference)
            sample_method: greedy/beam,
            max_length,
            temp,
            beam_size (optional, sample_method=beam),
            n_best (optional, sample_method=beam),
        }
        """
        encoder_output_dict = self.encoder(input_dict)
        output = self.forward_decoder(input_dict, encoder_output_dict)
        return output

    def forward_decoder(self, input_dict: Dict, encoder_output_dict: Dict):
        if input_dict["mode"] == "train":
            forward_dict = {
                "mode": "train", "sample_method": "greedy", "temp": 1.0
            }
            for key in self.train_forward_keys:
                forward_dict[key] = input_dict[key]
            forward_dict.update(encoder_output_dict)
            output = self.train_forward(forward_dict)
        elif input_dict["mode"] == "inference":
            forward_dict = {"mode": "inference"}
            default_args = { "sample_method": "greedy", "max_length": self.max_length, "temp": 1.0 }
            for key in self.inference_forward_keys:
                if key in input_dict:
                    forward_dict[key] = input_dict[key]
                else:
                    forward_dict[key] = default_args[key]

            if forward_dict["sample_method"] == "beam":
                forward_dict["beam_size"] = input_dict.get("beam_size", 3)
                forward_dict["n_best"] = input_dict.get("n_best", False)
                forward_dict["n_best_size"] = input_dict.get("n_best_size", forward_dict["beam_size"])
            elif forward_dict["sample_method"] == "dbs":
                forward_dict["beam_size"] = input_dict.get("beam_size", 6)
                forward_dict["group_size"] = input_dict.get("group_size", 3)
                forward_dict["diversity_lambda"] = input_dict.get("diversity_lambda", 0.5)
                forward_dict["group_nbest"] = input_dict.get("group_nbest", True)

            forward_dict.update(encoder_output_dict)
            output = self.inference_forward(forward_dict)
        else:
            raise Exception("mode should be either 'train' or 'inference'")
        output.update(encoder_output_dict)
        return output

    def prepare_output(self, input_dict):
        output = {}
        batch_size = input_dict["fc_emb"].size(0)
        if input_dict["mode"] == "train":
            max_length = input_dict["cap"].size(1) - 1
        elif input_dict["mode"] == "inference":
            max_length = input_dict["max_length"]
        else:
            raise Exception("mode should be either 'train' or 'inference'")
        device = input_dict["fc_emb"].device
        output["seq"] = torch.full((batch_size, max_length), self.end_idx,
                                   dtype=torch.long)
        output["logit"] = torch.empty(batch_size, max_length,
                                      self.vocab_size).to(device)
        output["sampled_logprob"] = torch.zeros(batch_size, max_length)
        output["embed"] = torch.empty(batch_size, max_length,
                                      self.decoder.d_model).to(device)
        return output

    def train_forward(self, input_dict):
        if input_dict["ss_ratio"] != 1: # scheduled sampling training
            input_dict["mode"] = "train"
            return self.stepwise_forward(input_dict)
        output = self.seq_forward(input_dict)
        self.train_process(output, input_dict)
        return output

    def seq_forward(self, input_dict):
        raise NotImplementedError

    def train_process(self, output, input_dict):
        pass

    def inference_forward(self, input_dict):
        if input_dict["sample_method"] == "beam":
            return self.beam_search(input_dict)
        elif input_dict["sample_method"] == "dbs":
            return self.diverse_beam_search(input_dict)
        return self.stepwise_forward(input_dict)

    def stepwise_forward(self, input_dict):
        """Step-by-step decoding"""
        output = self.prepare_output(input_dict)
        max_length = output["seq"].size(1)
        # start sampling
        for t in range(max_length):
            input_dict["t"] = t
            self.decode_step(input_dict, output)
            if input_dict["mode"] == "inference": # decide whether to stop when sampling
                unfinished_t = output["seq"][:, t] != self.end_idx
                if t == 0:
                    unfinished = unfinished_t
                else:
                    unfinished *= unfinished_t
                output["seq"][:, t][~unfinished] = self.end_idx
                if unfinished.sum() == 0:
                    break
        self.stepwise_process(output)
        return output

    def decode_step(self, input_dict, output):
        """Decoding operation of timestep t"""
        decoder_input = self.prepare_decoder_input(input_dict, output)
        # feed to the decoder to get logit
        output_t = self.decoder(decoder_input)
        logit_t = output_t["logit"]
        # assert logit_t.ndim == 3
        if logit_t.size(1) == 1:
            logit_t = logit_t.squeeze(1)
            embed_t = output_t["embed"].squeeze(1)
        elif logit_t.size(1) > 1:
            logit_t = logit_t[:, -1, :]
            embed_t = output_t["embed"][:, -1, :]
        else:
            raise Exception("no logit output")
        # sample the next input word and get the corresponding logit
        sampled = self.sample_next_word(logit_t,
                                        method=input_dict["sample_method"],
                                        temp=input_dict["temp"])

        output_t.update(sampled)
        output_t["t"] = input_dict["t"]
        output_t["logit"] = logit_t
        output_t["embed"] = embed_t
        self.stepwise_process_step(output, output_t)

    def prepare_decoder_input(self, input_dict, output):
        """Prepare the inp ut dict for the decoder"""
        raise NotImplementedError
    
    def stepwise_process_step(self, output, output_t):
        """Postprocessing (save output values) after each timestep t"""
        t = output_t["t"]
        output["logit"][:, t, :] = output_t["logit"]
        output["seq"][:, t] = output_t["word"]
        output["sampled_logprob"][:, t] = output_t["probs"]
        output["embed"][:, t, :] = output_t["embed"]

    def stepwise_process(self, output):
        """Postprocessing after the whole step-by-step autoregressive decoding"""
        pass

    def sample_next_word(self, logit, method, temp):
        """Sample the next word, given probs output by the decoder"""
        logprob = torch.log_softmax(logit, dim=1)
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
                    probs = torch.softmax(logit, dim=1)
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

    def beam_search(self, input_dict):
        output = self.prepare_output(input_dict)
        max_length = input_dict["max_length"]
        beam_size = input_dict["beam_size"]
        if input_dict["n_best"]:
            n_best_size = input_dict["n_best_size"]
            batch_size, max_length = output["seq"].size()
            output["seq"] = torch.full((batch_size, n_best_size, max_length),
                                        self.end_idx, dtype=torch.long)
            
        temp = input_dict["temp"]
        # instance by instance beam seach
        for i in range(output["seq"].size(0)):
            output_i = self.prepare_beamsearch_output(input_dict)
            input_dict["sample_idx"] = i
            for t in range(max_length):
                input_dict["t"] = t
                output_t = self.beamsearch_step(input_dict, output_i)
                #######################################
                # merge with previous beam and select the current max prob beam
                #######################################
                logit_t = output_t["logit"]
                if logit_t.size(1) == 1:
                    logit_t = logit_t.squeeze(1)
                elif logit_t.size(1) > 1:
                    logit_t = logit_t[:, -1, :]
                else:
                    raise Exception("no logit output")
                logprob_t = torch.log_softmax(logit_t, dim=1)
                logprob_t = torch.log_softmax(logprob_t / temp, dim=1)
                logprob_t = output_i["topk_logprob"].unsqueeze(1) + logprob_t
                if t == 0: # for the first step, all k seq will have the same probs
                    topk_logprob, topk_words = logprob_t[0].topk(
                        beam_size, 0, True, True)
                else: # unroll and find top logprob, and their unrolled indices
                    topk_logprob, topk_words = logprob_t.view(-1).topk(
                        beam_size, 0, True, True)
                topk_words = topk_words.cpu()
                output_i["topk_logprob"] = topk_logprob
                # output_i["prev_words_beam"] = topk_words // self.vocab_size  # [beam_size,]
                output_i["prev_words_beam"] = torch.div(topk_words, self.vocab_size,
                                                        rounding_mode='trunc')
                output_i["next_word"] = topk_words % self.vocab_size  # [beam_size,]
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

                self.beamsearch_process_step(output_i, output_t)

                if len(output_i["done_beams"]) == beam_size:
                    break

            self.beamsearch_process(output, output_i, input_dict)
        return output

    def prepare_beamsearch_output(self, input_dict):
        beam_size = input_dict["beam_size"]
        device = input_dict["fc_emb"].device
        output = {
            "topk_logprob": torch.zeros(beam_size).to(device),
            "seq": None,
            "prev_words_beam": None,
            "next_word": None,
            "done_beams": [],
        }
        return output

    def beamsearch_step(self, input_dict, output_i):
        decoder_input = self.prepare_beamsearch_decoder_input(input_dict, output_i)
        output_t = self.decoder(decoder_input)
        output_t["t"] = input_dict["t"]
        return output_t

    def prepare_beamsearch_decoder_input(self, input_dict, output_i):
        raise NotImplementedError
            
    def beamsearch_process_step(self, output_i, output_t):
        pass

    def beamsearch_process(self, output, output_i, input_dict):
        i = input_dict["sample_idx"]
        done_beams = sorted(output_i["done_beams"], key=lambda x: -x["score"])
        if input_dict["n_best"]:
            done_beams = done_beams[:input_dict["n_best_size"]]
            for out_idx, done_beam in enumerate(done_beams):
                seq = done_beam["seq"]
                output["seq"][i][out_idx, :len(seq)] = seq
        else:
            seq = done_beams[0]["seq"]
            output["seq"][i][:len(seq)] = seq
    
    def diverse_beam_search(self, input_dict):
        
        def add_diversity(seq_table, logprob, t, divm, diversity_lambda, bdash):
            local_time = t - divm
            unaug_logprob = logprob.clone()

            if divm > 0:
                change = torch.zeros(logprob.size(-1))
                for prev_choice in range(divm):
                    prev_decisions = seq_table[prev_choice][..., local_time]
                    for prev_labels in range(bdash):
                        change.scatter_add_(0, prev_decisions[prev_labels], change.new_ones(1))

                change = change.to(logprob.device)
                logprob = logprob - repeat_tensor(change, bdash) * diversity_lambda

            return logprob, unaug_logprob

        output = self.prepare_output(input_dict)
        group_size = input_dict["group_size"]
        batch_size = output["seq"].size(0)
        beam_size = input_dict["beam_size"]
        bdash = beam_size // group_size
        input_dict["bdash"] = bdash
        diversity_lambda = input_dict["diversity_lambda"]
        device = input_dict["fc_emb"].device
        max_length = input_dict["max_length"]
        temp = input_dict["temp"]
        group_nbest = input_dict["group_nbest"]
        batch_size, max_length = output["seq"].size()
        if group_nbest:
            output["seq"] = torch.full((batch_size, beam_size, max_length),
                                        self.end_idx, dtype=torch.long)
        else:
            output["seq"] = torch.full((batch_size, group_size, max_length),
                                        self.end_idx, dtype=torch.long)


        for i in range(batch_size):
            input_dict["sample_idx"] = i
            seq_table = [torch.LongTensor(bdash, 0) for _ in range(group_size)] # group_size x [bdash, 0]
            logprob_table = [torch.zeros(bdash).to(device) for _ in range(group_size)]
            done_beams_table = [[] for _ in range(group_size)]

            output_i = {
                "prev_words_beam": [None for _ in range(group_size)],
                "next_word": [None for _ in range(group_size)],
                "state": [None for _ in range(group_size)]
            }

            for t in range(max_length + group_size - 1):
                input_dict["t"] = t
                for divm in range(group_size):
                    input_dict["divm"] = divm
                    if t >= divm and t <= max_length + divm - 1:
                        local_time = t - divm
                        decoder_input = self.prepare_dbs_decoder_input(input_dict, output_i)
                        output_t = self.decoder(decoder_input)
                        output_t["divm"] = divm
                        logit_t = output_t["logit"]
                        if logit_t.size(1) == 1:
                            logit_t = logit_t.squeeze(1)
                        elif logit_t.size(1) > 1:
                            logit_t = logit_t[:, -1, :]
                        else:
                            raise Exception("no logit output")
                        logprob_t = torch.log_softmax(logit_t, dim=1)
                        logprob_t = torch.log_softmax(logprob_t / temp, dim=1)
                        logprob_t, unaug_logprob_t = add_diversity(seq_table, logprob_t, t, divm, diversity_lambda, bdash)
                        logprob_t = logprob_table[divm].unsqueeze(-1) + logprob_t
                        if local_time == 0: # for the first step, all k seq will have the same probs
                            topk_logprob, topk_words = logprob_t[0].topk(
                                bdash, 0, True, True)
                        else: # unroll and find top logprob, and their unrolled indices
                            topk_logprob, topk_words = logprob_t.view(-1).topk(
                                bdash, 0, True, True)
                        topk_words = topk_words.cpu()
                        logprob_table[divm] = topk_logprob
                        output_i["prev_words_beam"][divm] = topk_words // self.vocab_size  # [bdash,]
                        output_i["next_word"][divm] = topk_words % self.vocab_size  # [bdash,]
                        if local_time > 0:
                            seq_table[divm] = seq_table[divm][output_i["prev_words_beam"][divm]]
                        seq_table[divm] = torch.cat([
                            seq_table[divm],
                            output_i["next_word"][divm].unsqueeze(-1)], -1)

                        is_end = seq_table[divm][:, t-divm] == self.end_idx
                        assert seq_table[divm].shape[-1] == t - divm + 1
                        if t == max_length + divm - 1:
                            is_end.fill_(1)
                        for beam_idx in range(bdash):
                            if is_end[beam_idx]:
                                final_beam = {
                                    "seq": seq_table[divm][beam_idx].clone(),
                                    "score": logprob_table[divm][beam_idx].item()
                                }
                                final_beam["score"] = final_beam["score"] / (t - divm + 1)
                                done_beams_table[divm].append(final_beam)
                        logprob_table[divm][is_end] -= 1000
                        self.dbs_process_step(output_i, output_t)
            done_beams_table = [sorted(done_beams_table[divm], key=lambda x: -x["score"])[:bdash] for divm in range(group_size)]
            if group_nbest:
                done_beams = sum(done_beams_table, [])
            else:
                done_beams = [group_beam[0] for group_beam in done_beams_table]
            for _, done_beam in enumerate(done_beams):
                output["seq"][i, _, :len(done_beam["seq"])] = done_beam["seq"]

        return output
            
    def prepare_dbs_decoder_input(self, input_dict, output_i):
        raise NotImplementedError

    def dbs_process_step(self, output_i, output_t):
        pass


class TransformerModel(CaptionModel):

    def __init__(self, encoder: nn.Module, decoder: nn.Module, **kwargs):
        if not hasattr(self, "compatible_decoders"):
            self.compatible_decoders = (
                TransformerDecoder,
            )
        super().__init__(encoder, decoder, **kwargs)

    def seq_forward(self, input_dict):
        cap = input_dict["cap"]
        cap_padding_mask = (cap == self.pad_idx).to(cap.device)
        cap_padding_mask = cap_padding_mask[:, :-1]
        output = self.decoder(
            {
                "word": cap[:, :-1],
                "attn_emb": input_dict["attn_emb"],
                "attn_emb_len": input_dict["attn_emb_len"],
                "cap_padding_mask": cap_padding_mask
            }
        )
        return output

    def prepare_decoder_input(self, input_dict, output):
        decoder_input = {
            "attn_emb": input_dict["attn_emb"],
            "attn_emb_len": input_dict["attn_emb_len"]
        }
        t = input_dict["t"]
        
        ###############
        # determine input word
        ################
        if input_dict["mode"] == "train" and random.random() < input_dict["ss_ratio"]: # training, scheduled sampling
            word = input_dict["cap"][:, :t+1]
        else:
            start_word = torch.tensor([self.start_idx,] * input_dict["attn_emb"].size(0)).unsqueeze(1).long()
            if t == 0:
                word = start_word
            else:
                word = torch.cat((start_word, output["seq"][:, :t]), dim=-1)
        # word: [N, T]
        decoder_input["word"] = word

        cap_padding_mask = (word == self.pad_idx).to(input_dict["attn_emb"].device)
        decoder_input["cap_padding_mask"] = cap_padding_mask
        return decoder_input

    def prepare_beamsearch_decoder_input(self, input_dict, output_i):
        decoder_input = {}
        t = input_dict["t"]
        i = input_dict["sample_idx"]
        beam_size = input_dict["beam_size"]
        ###############
        # prepare attn embeds
        ################
        if t == 0:
            attn_emb = repeat_tensor(input_dict["attn_emb"][i], beam_size)
            attn_emb_len = repeat_tensor(input_dict["attn_emb_len"][i], beam_size)
            output_i["attn_emb"] = attn_emb
            output_i["attn_emb_len"] = attn_emb_len
        decoder_input["attn_emb"] = output_i["attn_emb"]
        decoder_input["attn_emb_len"] = output_i["attn_emb_len"]
        ###############
        # determine input word
        ################
        start_word = torch.tensor([self.start_idx,] * beam_size).unsqueeze(1).long()
        if t == 0:
            word = start_word
        else:
            word = torch.cat((start_word, output_i["seq"]), dim=-1)
        decoder_input["word"] = word
        cap_padding_mask = (word == self.pad_idx).to(input_dict["attn_emb"].device)
        decoder_input["cap_padding_mask"] = cap_padding_mask

        return decoder_input


class BaseDecoder(nn.Module):
    """
    Take word/audio embeddings and output the next word probs
    """
    def __init__(self, emb_dim, vocab_size, fc_emb_dim,
                 attn_emb_dim, dropout=0.2, tie_weights=False):
        super().__init__()
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.fc_emb_dim = fc_emb_dim
        self.attn_emb_dim = attn_emb_dim
        self.tie_weights = tie_weights
        self.word_embedding = nn.Embedding(vocab_size, emb_dim)
        self.in_dropout = nn.Dropout(dropout)

    def forward(self, x):
        raise NotImplementedError

    def load_word_embedding(self, weight, freeze=True):
        embedding = np.load(weight)
        assert embedding.shape[0] == self.vocab_size, "vocabulary size mismatch"
        assert embedding.shape[1] == self.emb_dim, "embed size mismatch"
        
        # embeddings = torch.as_tensor(embeddings).float()
        # self.word_embeddings.weight = nn.Parameter(embeddings)
        # for para in self.word_embeddings.parameters():
            # para.requires_grad = tune
        self.word_embedding = nn.Embedding.from_pretrained(embedding,
            freeze=freeze)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * \
            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # self.register_buffer("pe", pe)
        self.register_parameter("pe", nn.Parameter(pe, requires_grad=False))

    def forward(self, x):
        # x: [T, N, E]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerDecoder(BaseDecoder):

    def __init__(self,
                 emb_dim,
                 vocab_size,
                 fc_emb_dim,
                 attn_emb_dim,
                 dropout,
                 freeze=False,
                 tie_weights=False,
                 **kwargs):
        super().__init__(emb_dim, vocab_size, fc_emb_dim, attn_emb_dim,
                         dropout=dropout, tie_weights=tie_weights)
        self.d_model = emb_dim
        self.nhead = kwargs.get("nhead", self.d_model // 64)
        self.nlayers = kwargs.get("nlayers", 2)
        self.dim_feedforward = kwargs.get("dim_feedforward", self.d_model * 4)

        self.pos_encoder = PositionalEncoding(self.d_model, dropout)
        layer = nn.TransformerDecoderLayer(d_model=self.d_model,
                                           nhead=self.nhead,
                                           dim_feedforward=self.dim_feedforward,
                                           dropout=dropout)
        self.model = nn.TransformerDecoder(layer, self.nlayers)
        self.classifier = nn.Linear(self.d_model, vocab_size, bias=False)
        if tie_weights:
            self.classifier.weight = self.word_embedding.weight
        self.attn_proj = nn.Sequential(
            nn.Linear(self.attn_emb_dim, self.d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(self.d_model)
        )
        self.init_params()

        self.freeze = freeze
        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def load_pretrained(self, pretrained, output_fn):
        checkpoint = torch.load(pretrained, map_location="cpu")

        if "model" in checkpoint:
            checkpoint = checkpoint["model"]
            if next(iter(checkpoint)).startswith("decoder."):
                state_dict = {}
                for k, v in checkpoint.items():
                    state_dict[k[8:]] = v

        loaded_keys = merge_load_state_dict(state_dict, self, output_fn)
        if self.freeze:
            for name, param in self.named_parameters():
                if name in loaded_keys:
                    param.requires_grad = False
                else:
                    param.requires_grad = True


    def generate_square_subsequent_mask(self, max_length):
        mask = (torch.triu(torch.ones(max_length, max_length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, input_dict):
        word = input_dict["word"]
        attn_emb = input_dict["attn_emb"]
        attn_emb_len = input_dict["attn_emb_len"]
        cap_padding_mask = input_dict["cap_padding_mask"]

        p_attn_emb = self.attn_proj(attn_emb)
        p_attn_emb = p_attn_emb.transpose(0, 1) # [T_src, N, emb_dim]
        word = word.to(attn_emb.device)
        embed = self.in_dropout(self.word_embedding(word)) * math.sqrt(self.emb_dim) # [N, T, emb_dim]
        embed = embed.transpose(0, 1) # [T, N, emb_dim]
        embed = self.pos_encoder(embed)

        tgt_mask = self.generate_square_subsequent_mask(embed.size(0)).to(attn_emb.device)
        memory_key_padding_mask = ~generate_length_mask(attn_emb_len, attn_emb.size(1)).to(attn_emb.device)
        output = self.model(embed, p_attn_emb, tgt_mask=tgt_mask,
                            tgt_key_padding_mask=cap_padding_mask,
                            memory_key_padding_mask=memory_key_padding_mask)
        output = output.transpose(0, 1)
        output = {
            "embed": output,
            "logit": self.classifier(output),
        }
        return output
    

class ContraEncoderKdWrapper(nn.Module, CaptionMetaMixin):

    def __init__(self,
                 model: nn.Module,
                 shared_dim: int,
                 tchr_dim: int,
                 ):
        super().__init__()
        self.model = model
        self.tchr_dim = tchr_dim
        if hasattr(model, "encoder"):
            self.stdnt_proj = nn.Linear(model.encoder.fc_emb_size,
                                        shared_dim)
        else:
            self.stdnt_proj = nn.Linear(model.fc_emb_size,
                                        shared_dim)
        self.stdnt_proj.apply(init)
        self.tchr_proj = nn.Linear(tchr_dim, shared_dim)
        self.tchr_proj.apply(init)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, input_dict: Dict):
        unsup = input_dict.get("unsup", False)
        if unsup is False:
            output_dict = self.model(input_dict)
        else:
            output_dict = self.model.encoder(input_dict)
        if "tchr_output" in input_dict:
            stdnt_emb = output_dict["fc_emb"]
            stdnt_emb = self.stdnt_proj(stdnt_emb)
            tchr_emb = input_dict["tchr_output"]["embedding"]
            thcr_emb = self.tchr_proj(tchr_emb)

            stdnt_emb = F.normalize(stdnt_emb, dim=-1)
            thcr_emb = F.normalize(thcr_emb, dim=-1)

            unscaled_logit = stdnt_emb @ thcr_emb.transpose(0, 1)
            logit = self.logit_scale * unscaled_logit
            label = torch.arange(logit.shape[0]).to(logit.device)
            loss1 = F.cross_entropy(logit, label)
            loss2 = F.cross_entropy(logit.transpose(0, 1), label)
            loss = (loss1 + loss2) / 2
            output_dict["enc_kd_loss"] = loss
        return output_dict


class Effb2TrmConfig(PretrainedConfig):

    def __init__(
        self,
        sample_rate: int = 16000,
        tchr_dim: int = 768,
        shared_dim: int = 1024,
        fc_emb_dim: int = 1408,
        attn_emb_dim: int = 1408,
        decoder_n_layers: int = 2,
        decoder_we_tie_weights: bool = True,
        decoder_emb_dim: int = 256,
        decoder_dropout: float = 0.2,
        vocab_size: int = 4981,
        **kwargs
    ):
        self.sample_rate = sample_rate
        self.tchr_dim = tchr_dim
        self.shared_dim = shared_dim
        self.fc_emb_dim = fc_emb_dim
        self.attn_emb_dim = attn_emb_dim
        self.decoder_n_layers = decoder_n_layers
        self.decoder_we_tie_weights = decoder_we_tie_weights
        self.decoder_emb_dim = decoder_emb_dim
        self.decoder_dropout = decoder_dropout
        self.vocab_size = vocab_size
        super().__init__(**kwargs)


class Effb2TrmCaptioningModel(PreTrainedModel):
    config_class = Effb2TrmConfig

    def __init__(self, config):
        super().__init__(config)
        self.encoder = EfficientNetB2(pretrained=True)
        self.decoder = TransformerDecoder(
            emb_dim=config.decoder_emb_dim,
            vocab_size=config.vocab_size,
            fc_emb_dim=config.fc_emb_dim,
            attn_emb_dim=config.attn_emb_dim,
            dropout=config.decoder_dropout,
            nlayers=config.decoder_n_layers,
            tie_weights=config.decoder_we_tie_weights
        )
        model = TransformerModel(self.encoder, self.decoder)
        self.model = ContraEncoderKdWrapper(model, config.shared_dim, config.tchr_dim)
    
    def forward(self,
                audio: torch.Tensor,
                audio_length: Union[List, np.ndarray, torch.Tensor],
                sample_method: str = "beam",
                beam_size: int = 3,
                max_length: int = 20,
                temp: float = 1.0,):
        device = self.device
        input_dict = {
            "wav": audio.to(device),
            "wav_len": audio_length,
            "specaug": False,
            "mode": "inference",
            "sample_method": sample_method,
            "max_length": max_length,
            "temp": temp,
        }
        if sample_method == "beam":
            input_dict["beam_size"] = beam_size
        return self.model(input_dict)["seq"].cpu()

