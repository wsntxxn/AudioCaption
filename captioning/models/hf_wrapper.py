from typing import Dict, Callable, Union, List
import random
import math
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
from torchaudio import transforms
from torchlibrosa import SpecAugmentation
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch import utils as efficientnet_utils
from einops import rearrange, reduce
from torch.hub import load_state_dict_from_url
from transformers import PretrainedConfig, PreTrainedModel


def sort_pack_padded_sequence(input, lengths):
    sorted_lengths, indices = torch.sort(lengths, descending=True)
    tmp = pack_padded_sequence(input[indices], sorted_lengths.cpu(), batch_first=True)
    inv_ix = indices.clone()
    inv_ix[indices] = torch.arange(0,len(indices)).type_as(inv_ix)
    return tmp, inv_ix

def pad_unsort_packed_sequence(input, inv_ix):
    tmp, _ = pad_packed_sequence(input, batch_first=True)
    tmp = tmp[inv_ix]
    return tmp

def pack_wrapper(module, attn_feats, attn_feat_lens):
    packed, inv_ix = sort_pack_padded_sequence(attn_feats, attn_feat_lens)
    if isinstance(module, torch.nn.RNNBase):
        return pad_unsort_packed_sequence(module(packed)[0], inv_ix)
    else:
        return pad_unsort_packed_sequence(PackedSequence(module(packed[0]), packed[1]), inv_ix)

def embedding_pooling(x, lens, pooling="mean"):
    if pooling == "max":
        fc_embs = max_with_lens(x, lens)
    elif pooling == "mean":
        fc_embs = mean_with_lens(x, lens)
    elif pooling == "mean+max":
        x_mean = mean_with_lens(x, lens)
        x_max = max_with_lens(x, lens)
        fc_embs = x_mean + x_max
    elif pooling == "last":
        indices = (lens - 1).reshape(-1, 1, 1).repeat(1, 1, x.size(-1))
        # indices: [N, 1, hidden]
        fc_embs = torch.gather(x, 1, indices).squeeze(1)
    else:
        raise Exception(f"pooling method {pooling} not support")
    return fc_embs

def interpolate(x, ratio):
    """Interpolate data in time domain. This is used to compensate the 
    resolution reduction in downsampling of a CNN.
    
    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate

    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled

def pad_framewise_output(framewise_output, frames_num):
    """Pad framewise_output to the same length as input frames. The pad value 
    is the same as the value of the last frame.

    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad

    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    pad = framewise_output[:, -1 :, :].repeat(1, frames_num - framewise_output.shape[1], 1)
    """tensor for padding"""

    output = torch.cat((framewise_output, pad), dim=1)
    """(batch_size, frames_num, classes_num)"""

    return output

def find_contiguous_regions(activity_array):
    """Find contiguous regions from bool valued numpy.array.
    Copy of https://dcase-repo.github.io/dcase_util/_modules/dcase_util/data/decisions.html#DecisionEncoder

    Reason is:
    1. This does not belong to a class necessarily
    2. Import DecisionEncoder requires sndfile over some other imports..which causes some problems on clusters

    """

    # Find the changes in the activity_array
    change_indices = np.logical_xor(activity_array[1:],
                                    activity_array[:-1]).nonzero()[0]

    # Shift change_index with one, focus on frame after the change.
    change_indices += 1

    if activity_array[0]:
        # If the first element of activity_array is True add 0 at the beginning
        change_indices = np.r_[0, change_indices]

    if activity_array[-1]:
        # If the last element of activity_array is True, add the length of the array
        change_indices = np.r_[change_indices, activity_array.size]

    # Reshape the result into two columns
    return change_indices.reshape((-1, 2))

def double_threshold(x, high_thres, low_thres, n_connect=1):
    """double_threshold
    Helper function to calculate double threshold for n-dim arrays

    :param x: input array
    :param high_thres: high threshold value
    :param low_thres: Low threshold value
    :param n_connect: Distance of <= n clusters will be merged
    """
    assert x.ndim <= 3, "Whoops something went wrong with the input ({}), check if its <= 3 dims".format(
        x.shape)
    if x.ndim == 3:
        apply_dim = 1
    elif x.ndim < 3:
        apply_dim = 0
    # x is assumed to be 3d: (batch, time, dim)
    # Assumed to be 2d : (time, dim)
    # Assumed to be 1d : (time)
    # time axis is therefore at 1 for 3d and 0 for 2d (
    return np.apply_along_axis(lambda x: _double_threshold(
        x, high_thres, low_thres, n_connect=n_connect),
                               axis=apply_dim,
                               arr=x)

def _double_threshold(x, high_thres, low_thres, n_connect=1, return_arr=True):
    """_double_threshold
    Computes a double threshold over the input array

    :param x: input array, needs to be 1d
    :param high_thres: High threshold over the array
    :param low_thres: Low threshold over the array
    :param n_connect: Postprocessing, maximal distance between clusters to connect
    :param return_arr: By default this function returns the filtered indiced, but if return_arr = True it returns an array of tsame size as x filled with ones and zeros.
    """
    assert x.ndim == 1, "Input needs to be 1d"
    high_locations = np.where(x > high_thres)[0]
    locations = x > low_thres
    encoded_pairs = find_contiguous_regions(locations)

    filtered_list = list(
        filter(
            lambda pair:
            ((pair[0] <= high_locations) & (high_locations <= pair[1])).any(),
            encoded_pairs))

    filtered_list = connect_(filtered_list, n_connect)
    if return_arr:
        zero_one_arr = np.zeros_like(x, dtype=int)
        for sl in filtered_list:
            zero_one_arr[sl[0]:sl[1]] = 1
        return zero_one_arr
    return filtered_list

def connect_(pairs, n=1):
    """connect_
    Connects two adjacent clusters if their distance is <= n

    :param pairs: Clusters of iterateables e.g., [(1,5),(7,10)]
    :param n: distance between two clusters 
    """
    if len(pairs) == 0:
        return []
    start_, end_ = pairs[0]
    new_pairs = []
    for i, (next_item, cur_item) in enumerate(zip(pairs[1:], pairs[0:])):
        end_ = next_item[1]
        if next_item[0] - cur_item[1] <= n:
            pass
        else:
            new_pairs.append((start_, cur_item[1]))
            start_ = next_item[0]
    new_pairs.append((start_, end_))
    return new_pairs

def segments_to_temporal_tag(segments, thre=0.5):
    after_flag, while_flag = 0, 0
    for j in range(len(segments)):
        for k in range(len(segments)):
            if segments[j][0] == segments[k][0]:
                continue
            min_duration = min(segments[j][2] - segments[j][1], segments[k][2] - segments[k][1])
            overlap = segments[j][2] - segments[k][1]
            if overlap < thre * min_duration:
                after_flag = 2
            if segments[j][1] < segments[k][1] and overlap > thre * min_duration:
                while_flag = 1
    return after_flag + while_flag

def decode_with_timestamps(labels, time_resolution):
    batch_results = []
    for lab in labels:
        segments = []
        for i, label_column in enumerate(lab.T):
            change_indices = find_contiguous_regions(label_column)
            # append [onset, offset] in the result list
            for row in change_indices:
                segments.append((i, row[0] * time_resolution, row[1] * time_resolution))
        temporal_tag = segments_to_temporal_tag(segments)
        batch_results.append(temporal_tag)
    return batch_results

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
        self.tchr_proj = nn.Linear(tchr_dim, shared_dim)
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
        encoder = EfficientNetB2(pretrained=True)
        decoder = TransformerDecoder(
            emb_dim=config.decoder_emb_dim,
            vocab_size=config.vocab_size,
            fc_emb_dim=config.fc_emb_dim,
            attn_emb_dim=config.attn_emb_dim,
            dropout=config.decoder_dropout,
            nlayers=config.decoder_n_layers,
            tie_weights=config.decoder_we_tie_weights
        )
        model = TransformerModel(encoder, decoder)
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


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        
        return x


class Cnn14Encoder(nn.Module):

    def __init__(self, sample_rate=32000):
        super().__init__()
        sr_to_fmax = {
            32000: 14000,
            16000: 8000
        }
        # Logmel spectrogram extractor
        self.melspec_extractor = transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=32 * sample_rate // 1000,
            win_length=32 * sample_rate // 1000,
            hop_length=10 * sample_rate // 1000,
            f_min=50,
            f_max=sr_to_fmax[sample_rate],
            n_mels=64,
            norm="slaney",
            mel_scale="slaney"
        )
        self.hop_length = 10 * sample_rate // 1000
        self.db_transform = transforms.AmplitudeToDB()

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.downsample_ratio = 32

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_emb_size = 2048
 
    def forward(self, input_dict):
        lms = input_dict["lms"]
        wave_length = input_dict["wav_len"]

        x = lms    # (batch_size, mel_bins, time_steps)
        x = x.transpose(1, 2)
        x = x.unsqueeze(1)      # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        attn_emb = x.transpose(1, 2)
        
        wave_length = torch.as_tensor(wave_length)
        feat_length = torch.div(wave_length, self.hop_length,
            rounding_mode="floor") + 1
        feat_length = torch.div(feat_length, self.downsample_ratio,
            rounding_mode="floor")
        x_max = max_with_lens(attn_emb, feat_length)
        x_mean = mean_with_lens(attn_emb, feat_length)
        x = x_max + x_mean
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        fc_emb = F.dropout(x, p=0.5, training=self.training)
        
        output_dict = {
            'fc_emb': fc_emb,
            'attn_emb': attn_emb,
            'attn_emb_len': feat_length
        }

        return output_dict


class RnnEncoder(nn.Module):

    def __init__(self,
                 attn_feat_dim,
                 pooling="mean",
                 **kwargs):
        super().__init__()
        self.pooling = pooling
        self.hidden_size = kwargs.get('hidden_size', 512)
        self.bidirectional = kwargs.get('bidirectional', False)
        self.num_layers = kwargs.get('num_layers', 1)
        self.dropout = kwargs.get('dropout', 0.2)
        self.rnn_type = kwargs.get('rnn_type', "GRU")
        self.in_bn = kwargs.get('in_bn', False)
        self.embed_dim = self.hidden_size * (self.bidirectional + 1)
        self.network = getattr(nn, self.rnn_type)(
            attn_feat_dim,
            self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            dropout=self.dropout,
            batch_first=True)
        if self.in_bn:
            self.bn = nn.BatchNorm1d(self.embed_dim)

    def forward(self, input_dict):
        x = input_dict["attn"]
        lens = input_dict["attn_len"]
        lens = torch.as_tensor(lens)
        # x: [N, T, E]
        if self.in_bn:
            x = pack_wrapper(self.bn, x, lens)
        out = pack_wrapper(self.network, x, lens)
        # out: [N, T, hidden]
        attn_emb = out
        fc_emb = embedding_pooling(out, lens, self.pooling)
        return {
            "attn_emb": attn_emb,
            "fc_emb": fc_emb,
            "attn_emb_len": lens
        }


class Cnn14RnnEncoder(nn.Module):

    def __init__(self,
                 sample_rate,
                 rnn_bidirectional,
                 rnn_hidden_size,
                 rnn_dropout,
                 rnn_num_layers):
        super().__init__()
        self.cnn = Cnn14Encoder(sample_rate=sample_rate)
        self.rnn = RnnEncoder(
            2048,
            bidirectional=rnn_bidirectional,
            hidden_size=rnn_hidden_size,
            dropout=rnn_dropout,
            num_layers=rnn_num_layers,
        )

    def forward(self, input_dict):
        output_dict = self.cnn(input_dict)
        output_dict["attn"] = output_dict["attn_emb"]
        output_dict["attn_len"] = output_dict["attn_emb_len"]
        del output_dict["attn_emb"], output_dict["attn_emb_len"]
        output_dict = self.rnn(output_dict)
        return output_dict


class Seq2SeqAttention(nn.Module):

    def __init__(self, hs_enc, hs_dec, attn_size):
        """
        Args:
            hs_enc: encoder hidden size
            hs_dec: decoder hidden size
            attn_size: attention vector size
        """
        super(Seq2SeqAttention, self).__init__()
        self.h2attn = nn.Linear(hs_enc + hs_dec, attn_size)
        self.v = nn.Parameter(torch.randn(attn_size))

    def forward(self, h_dec, h_enc, src_lens):
        """
        Args:
            h_dec: decoder hidden (query), [N, hs_dec]
            h_enc: encoder memory (key/value), [N, src_max_len, hs_enc]
            src_lens: source (encoder memory) lengths, [N, ]
        """
        N = h_enc.size(0)
        src_max_len = h_enc.size(1)
        h_dec = h_dec.unsqueeze(1).repeat(1, src_max_len, 1) # [N, src_max_len, hs_dec]

        attn_input = torch.cat((h_dec, h_enc), dim=-1)
        attn_out = torch.tanh(self.h2attn(attn_input)) # [N, src_max_len, attn_size]

        v = self.v.repeat(N, 1).unsqueeze(1) # [N, 1, attn_size]
        score = torch.bmm(v, attn_out.transpose(1, 2)).squeeze(1) # [N, src_max_len]

        idxs = torch.arange(src_max_len).repeat(N).view(N, src_max_len)
        mask = (idxs < src_lens.view(-1, 1)).to(h_dec.device)

        score = score.masked_fill(mask == 0, -1e10)
        weights = torch.softmax(score, dim=-1) # [N, src_max_len]
        ctx = torch.bmm(weights.unsqueeze(1), h_enc).squeeze(1) # [N, hs_enc]

        return ctx, weights


class RnnDecoder(BaseDecoder):

    def __init__(self, emb_dim, vocab_size, fc_emb_dim, attn_emb_dim,
                 dropout, d_model, **kwargs):
        super().__init__(emb_dim, vocab_size, fc_emb_dim, attn_emb_dim,
                         dropout,)
        self.d_model = d_model
        self.num_layers = kwargs.get('num_layers', 1)
        self.bidirectional = kwargs.get('bidirectional', False)
        self.rnn_type = kwargs.get('rnn_type', "GRU")
        self.classifier = nn.Linear(
            self.d_model * (self.bidirectional + 1), vocab_size)

    def forward(self, x):
        raise NotImplementedError

    def init_hidden(self, bs, device):
        num_dire = self.bidirectional + 1
        n_layer = self.num_layers
        hid_dim = self.d_model
        if self.rnn_type == "LSTM":
            return (torch.zeros(num_dire * n_layer, bs, hid_dim).to(device),
                    torch.zeros(num_dire * n_layer, bs, hid_dim).to(device))
        else:
            return torch.zeros(num_dire * n_layer, bs, hid_dim).to(device)
    

class BahAttnCatFcDecoder(RnnDecoder):

    def __init__(self, emb_dim, vocab_size, fc_emb_dim, attn_emb_dim,
                 dropout, d_model, **kwargs):
        """
        concatenate fc, attn, word to feed to the rnn
        """
        super().__init__(emb_dim, vocab_size, fc_emb_dim, attn_emb_dim,
                         dropout, d_model, **kwargs)
        attn_size = kwargs.get("attn_size", self.d_model)
        self.model = getattr(nn, self.rnn_type)(
            input_size=self.emb_dim * 3,
            hidden_size=self.d_model,
            batch_first=True,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional)
        self.attn = Seq2SeqAttention(self.attn_emb_dim,
                                     self.d_model * (self.bidirectional + 1) * \
                                             self.num_layers,
                                     attn_size)
        self.fc_proj = nn.Linear(self.fc_emb_dim, self.emb_dim)
        self.ctx_proj = nn.Linear(self.attn_emb_dim, self.emb_dim)

    def forward(self, input_dict):
        word = input_dict["word"]
        state = input_dict.get("state", None) # [n_layer * n_dire, bs, d_model]
        fc_emb = input_dict["fc_emb"]
        attn_emb = input_dict["attn_emb"]
        attn_emb_len = input_dict["attn_emb_len"]

        word = word.to(fc_emb.device)
        embed = self.in_dropout(self.word_embedding(word))

        # embed: [N, 1, embed_size]
        if state is None:
            state = self.init_hidden(word.size(0), fc_emb.device)
        if self.rnn_type == "LSTM":
            query = state[0].transpose(0, 1).flatten(1)
        else:
            query = state.transpose(0, 1).flatten(1)
        c, attn_weight = self.attn(query, attn_emb, attn_emb_len)

        p_fc_emb = self.fc_proj(fc_emb)
        p_ctx = self.ctx_proj(c)
        rnn_input = torch.cat((embed, p_ctx.unsqueeze(1), p_fc_emb.unsqueeze(1)),
                              dim=-1)

        out, state = self.model(rnn_input, state)

        output = {
            "state": state,
            "embed": out,
            "logit": self.classifier(out),
            "attn_weight": attn_weight
        }
        return output


class TemporalBahAttnDecoder(BahAttnCatFcDecoder):

    def __init__(self, emb_dim, vocab_size, fc_emb_dim, attn_emb_dim,
                 dropout, d_model, **kwargs):
        """
        concatenate fc, attn, word to feed to the rnn
        """
        super().__init__(emb_dim, vocab_size, fc_emb_dim, attn_emb_dim,
                         dropout, d_model, **kwargs)
        self.temporal_embedding = nn.Embedding(4, emb_dim)

    def forward(self, input_dict):
        word = input_dict["word"]
        state = input_dict.get("state", None) # [n_layer * n_dire, bs, d_model]
        fc_embs = input_dict["fc_emb"]
        attn_embs = input_dict["attn_emb"]
        attn_emb_lens = input_dict["attn_emb_len"]
        temporal_tag = input_dict["temporal_tag"]

        if input_dict["t"] == 0:
            embed = self.in_dropout(
                self.temporal_embedding(temporal_tag)).unsqueeze(1)
        elif word.size(-1) == self.fc_emb_dim: # fc_embs
            embed = word.unsqueeze(1)
        elif word.size(-1) == 1: # word
            word = word.to(fc_embs.device)
            embed = self.in_dropout(self.word_embedding(word))
        else:
            raise Exception(f"problem with word input size {word.size()}")

        # embed: [N, 1, embed_size]
        if state is None:
            state = self.init_hidden(word.size(0), fc_embs.device)
        if self.rnn_type == "LSTM":
            query = state[0].transpose(0, 1).flatten(1)
        else:
            query = state.transpose(0, 1).flatten(1)
        c, attn_weight = self.attn(query, attn_embs, attn_emb_lens)

        p_ctx = self.ctx_proj(c)
        p_fc_embs = self.fc_proj(fc_embs)
        p_ctx = self.ctx_proj(c)
        rnn_input = torch.cat((embed, p_ctx.unsqueeze(1), p_fc_embs.unsqueeze(1)), dim=-1)

        out, state = self.model(rnn_input, state)

        output = {
            "state": state,
            "embed": out,
            "logit": self.classifier(out),
            "attn_weight": attn_weight
        }
        return output


class Seq2SeqAttnModel(CaptionModel):

    def __init__(self, encoder, decoder, **kwargs):
        if not hasattr(self, "compatible_decoders"):
            self.compatible_decoders = (
                BahAttnCatFcDecoder,
            )
        super().__init__(encoder, decoder, **kwargs)


    def seq_forward(self, input_dict):
        # Bahdanau attention only supports step-by-step implementation, so we implement forward in 
        # step-by-step manner whether in training or evaluation
        return self.stepwise_forward(input_dict)

    def prepare_output(self, input_dict):
        output = super().prepare_output(input_dict)
        attn_weight = torch.empty(output["seq"].size(0),
            input_dict["attn_emb"].size(1), output["seq"].size(1))
        output["attn_weight"] = attn_weight
        return output

    def prepare_decoder_input(self, input_dict, output):
        decoder_input = {
            "fc_emb": input_dict["fc_emb"],
            "attn_emb": input_dict["attn_emb"],
            "attn_emb_len": input_dict["attn_emb_len"]
        }
        t = input_dict["t"]
        ###############
        # determine input word
        ################
        if input_dict["mode"] == "train" and random.random() < input_dict["ss_ratio"]: # training, scheduled sampling
            word = input_dict["cap"][:, t]
        else:
            if t == 0:
                word = torch.tensor([self.start_idx,] * input_dict["fc_emb"].size(0)).long()
            else:
                word = output["seq"][:, t-1]
        # word: [N,]
        decoder_input["word"] = word.unsqueeze(1)

        ################
        # prepare rnn state
        ################
        if t > 0:
            decoder_input["state"] = output["state"]
        return decoder_input

    def stepwise_process_step(self, output, output_t):
        super().stepwise_process_step(output, output_t)
        output["state"] = output_t["state"]
        t = output_t["t"]
        output["attn_weight"][:, :, t] = output_t["attn_weight"]

    def prepare_beamsearch_output(self, input_dict):
        output = super().prepare_beamsearch_output(input_dict)
        beam_size = input_dict["beam_size"]
        max_length = input_dict["max_length"]
        output["attn_weight"] = torch.empty(beam_size,
            max(input_dict["attn_emb_len"]), max_length)
        return output

    def prepare_beamsearch_decoder_input(self, input_dict, output_i):
        decoder_input = {}
        t = input_dict["t"]
        i = input_dict["sample_idx"]
        beam_size = input_dict["beam_size"]
        ###############
        # prepare fc embeds
        ################
        if t == 0:
            fc_emb = repeat_tensor(input_dict["fc_emb"][i], beam_size)
            output_i["fc_emb"] = fc_emb
        decoder_input["fc_emb"] = output_i["fc_emb"]

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
        if t == 0:
            word = torch.tensor([self.start_idx,] * beam_size).long()
        else:
            word = output_i["next_word"]
        decoder_input["word"] = word.unsqueeze(1)

        ################
        # prepare rnn state
        ################
        if t > 0:
            if self.decoder.rnn_type == "LSTM":
                decoder_input["state"] = (output_i["state"][0][:, output_i["prev_words_beam"], :].contiguous(),
                                          output_i["state"][1][:, output_i["prev_words_beam"], :].contiguous())
            else:
                decoder_input["state"] = output_i["state"][:, output_i["prev_words_beam"], :].contiguous()

        return decoder_input

    def beamsearch_process_step(self, output_i, output_t):
        t = output_t["t"]
        output_i["state"] = output_t["state"]
        output_i["attn_weight"][..., t] = output_t["attn_weight"]
        output_i["attn_weight"] = output_i["attn_weight"][output_i["prev_words_beam"], ...]

    def beamsearch_process(self, output, output_i, input_dict):
        super().beamsearch_process(output, output_i, input_dict)
        i = input_dict["sample_idx"]
        output["attn_weight"][i] = output_i["attn_weight"][0]

    def prepare_dbs_decoder_input(self, input_dict, output_i):
        decoder_input = {}
        t = input_dict["t"]
        i = input_dict["sample_idx"]
        bdash = input_dict["bdash"]
        divm = input_dict["divm"]

        local_time = t - divm
        ###############
        # prepare fc embeds
        ################
        # repeat only at the first timestep to save consumption
        if t == 0:
            fc_emb = repeat_tensor(input_dict["fc_emb"][i], bdash).unsqueeze(1)
            output_i["fc_emb"] = fc_emb
        decoder_input["fc_emb"] = output_i["fc_emb"]

        ###############
        # prepare attn embeds
        ################
        if t == 0:
            attn_emb = repeat_tensor(input_dict["attn_emb"][i], bdash)
            attn_emb_len = repeat_tensor(input_dict["attn_emb_len"][i], bdash)
            output_i["attn_emb"] = attn_emb
            output_i["attn_emb_len"] = attn_emb_len
        decoder_input["attn_emb"] = output_i["attn_emb"]
        decoder_input["attn_emb_len"] = output_i["attn_emb_len"]

        ###############
        # determine input word
        ################
        if local_time == 0:
            word = torch.tensor([self.start_idx,] * bdash).long()
        else:
            word = output_i["next_word"][divm]
        decoder_input["word"] = word.unsqueeze(1)

        ################
        # prepare rnn state
        ################
        if local_time > 0:
            if self.decoder.rnn_type == "LSTM":
                decoder_input["state"] = (
                    output_i["state"][0][divm][
                        :, output_i["prev_words_beam"][divm], :].contiguous(),
                    output_i["state"][1][divm][
                        :, output_i["prev_words_beam"][divm], :].contiguous()
                )
            else:
                decoder_input["state"] = output_i["state"][divm][
                    :, output_i["prev_words_beam"][divm], :].contiguous()

        return decoder_input

    def dbs_process_step(self, output_i, output_t):
        divm = output_t["divm"]
        output_i["state"][divm] = output_t["state"]
        # TODO attention weight


class TemporalSeq2SeqAttnModel(Seq2SeqAttnModel):

    def __init__(self, encoder, decoder, **kwargs):
        if not hasattr(self, "compatible_decoders"):
            self.compatible_decoders = (
                TemporalBahAttnDecoder,
            )
        super().__init__(encoder, decoder, **kwargs)
        self.train_forward_keys = ["cap", "cap_len", "ss_ratio", "temporal_tag"] 
        self.inference_forward_keys = ["sample_method", "max_length", "temp", "temporal_tag"]
        
    
    def prepare_decoder_input(self, input_dict, output):
        decoder_input = super().prepare_decoder_input(input_dict, output)
        decoder_input["temporal_tag"] = input_dict["temporal_tag"]
        decoder_input["t"] = input_dict["t"]
       
        return decoder_input


    def prepare_beamsearch_decoder_input(self, input_dict, output_i):
        decoder_input = super().prepare_beamsearch_decoder_input(input_dict, output_i)
        t = input_dict["t"]
        i = input_dict["sample_idx"]
        beam_size = input_dict["beam_size"]
        ###############
        # prepare temporal_tag
        ################
        if t == 0:
            temporal_tag = repeat_tensor(input_dict["temporal_tag"][i], beam_size)
            output_i["temporal_tag"] = temporal_tag
        decoder_input["temporal_tag"] = output_i["temporal_tag"]
        decoder_input["t"] = input_dict["t"]

        return decoder_input

    def prepare_dbs_decoder_input(self, input_dict, output_i):
        decoder_input = super.prepare_dbs_decoder_input(input_dict, output_i)
        t = input_dict["t"]
        i = input_dict["sample_idx"]
        bdash = input_dict["bdash"]

        ###############
        # prepare temporal tag
        ################
        # repeat only at the first timestep to save consumption
        if t == 0:
            temporal_tag = repeat_tensor(input_dict["temporal_tag"][i], bdash)
            output_i["temporal_tag"] = temporal_tag
        decoder_input["temporal_tag"] = output_i["temporal_tag"]
        decoder_input["t"] = input_dict["t"]

        return decoder_input


class Cnn8rnnSedModel(nn.Module):
    def __init__(self, classes_num):
        
        super().__init__()

        self.time_resolution = 0.01
        self.interpolate_ratio = 4     # Downsampled ratio

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(512, 512, bias=True)
        self.rnn = nn.GRU(512, 256, bidirectional=True, batch_first=True)
        self.fc_audioset = nn.Linear(512, classes_num, bias=True)
    
    def forward(self, lms):
        output = self.forward_prob(lms)
        framewise_output = output["framewise_output"].cpu().numpy()
        thresholded_predictions = double_threshold(
            framewise_output, 0.75, 0.25)
        decoded_tags = decode_with_timestamps(
            thresholded_predictions, self.time_resolution
        )
        return decoded_tags
 
    def forward_prob(self, lms):
        """
        lms: (batch_size, mel_bins, time_steps)"""

        x = lms
        x = x.transpose(1, 2)
        x = x.unsqueeze(1)

        frames_num = x.shape[2]
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg+max')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg+max')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(1, 2), pool_type='avg+max')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(1, 2), pool_type='avg+max')
        x = F.dropout(x, p=0.2, training=self.training) # (batch_size, 256, time_steps / 4, mel_bins / 16)
        x = torch.mean(x, dim=3)
        
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        x, _  = self.rnn(x)
        segmentwise_output = torch.sigmoid(self.fc_audioset(x)).clamp(1e-7, 1.)
        
        framewise_output = interpolate(segmentwise_output,
                                       self.interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, frames_num)

        output_dict = {
            "segmentwise_output": segmentwise_output,
            'framewise_output': framewise_output,
        }

        return output_dict


class Cnn14RnnTempAttnGruConfig(PretrainedConfig):

    def __init__(
        self,
        sample_rate: int = 32000,
        encoder_rnn_bidirectional: bool = True,
        encoder_rnn_hidden_size: int = 256,
        encoder_rnn_dropout: float = 0.5,
        encoder_rnn_num_layers: int = 3,
        decoder_emb_dim: int = 512,
        vocab_size: int = 4981,
        fc_emb_dim: int = 512,
        attn_emb_dim: int = 512,
        decoder_rnn_type: str = "GRU",
        decoder_num_layers: int = 1,
        decoder_d_model: int = 512,
        decoder_dropout: float = 0.5,
        **kwargs
    ):
        self.sample_rate = sample_rate
        self.encoder_rnn_bidirectional = encoder_rnn_bidirectional
        self.encoder_rnn_hidden_size = encoder_rnn_hidden_size
        self.encoder_rnn_dropout = encoder_rnn_dropout
        self.encoder_rnn_num_layers = encoder_rnn_num_layers
        self.decoder_emb_dim = decoder_emb_dim
        self.vocab_size = vocab_size
        self.fc_emb_dim = fc_emb_dim
        self.attn_emb_dim = attn_emb_dim
        self.decoder_rnn_type = decoder_rnn_type
        self.decoder_num_layers = decoder_num_layers
        self.decoder_d_model = decoder_d_model
        self.decoder_dropout = decoder_dropout
        super().__init__(**kwargs)


class Cnn14RnnTempAttnGruModel(PreTrainedModel):
    config_class = Cnn14RnnTempAttnGruConfig

    def __init__(self, config):
        super().__init__(config)
        sample_rate = config.sample_rate
        sr_to_fmax = {
            32000: 14000,
            16000: 8000
        }
        self.melspec_extractor = transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=32 * sample_rate // 1000,
            win_length=32 * sample_rate // 1000,
            hop_length=10 * sample_rate // 1000,
            f_min=50,
            f_max=sr_to_fmax[sample_rate],
            n_mels=64,
            norm="slaney",
            mel_scale="slaney"
        )
        self.db_transform = transforms.AmplitudeToDB()

        encoder = Cnn14RnnEncoder(
            sample_rate=config.sample_rate,
            rnn_bidirectional=config.encoder_rnn_bidirectional,
            rnn_hidden_size=config.encoder_rnn_hidden_size,
            rnn_dropout=config.encoder_rnn_dropout,
            rnn_num_layers=config.encoder_rnn_num_layers
        )
        decoder = TemporalBahAttnDecoder(
            emb_dim=config.decoder_emb_dim,
            vocab_size=config.vocab_size,
            fc_emb_dim=config.fc_emb_dim,
            attn_emb_dim=config.attn_emb_dim,
            rnn_type=config.decoder_rnn_type,
            num_layers=config.decoder_num_layers,
            d_model=config.decoder_d_model,
            dropout=config.decoder_dropout,
        )
        cap_model = TemporalSeq2SeqAttnModel(encoder, decoder)
        sed_model = Cnn8rnnSedModel(classes_num=447)
        self.cap_model = cap_model
        self.sed_model = sed_model

    def forward(self,
                audio: torch.Tensor,
                audio_length: Union[List, np.ndarray, torch.Tensor],
                temporal_tag: Union[List, np.ndarray, torch.Tensor] = None,
                sample_method: str = "beam",
                beam_size: int = 3,
                max_length: int = 20,
                temp: float = 1.0,):
        device = self.device
        mel_spec = self.melspec_extractor(audio.to(device))
        log_mel_spec = self.db_transform(mel_spec)

        sed_tag = self.sed_model(log_mel_spec)
        sed_tag = torch.as_tensor(sed_tag).to(device)
        if temporal_tag is not None:
            temporal_tag = torch.as_tensor(temporal_tag).to(device)
            temporal_tag = torch.stack([temporal_tag, sed_tag], dim=0)
            temporal_tag = torch.min(temporal_tag, dim=0).values
        else:
            temporal_tag = sed_tag

        input_dict = {
            "lms": log_mel_spec,
            "wav_len": audio_length,
            "temporal_tag": temporal_tag,
            "mode": "inference",
            "sample_method": sample_method,
            "max_length": max_length,
            "temp": temp,
        }
        if sample_method == "beam":
            input_dict["beam_size"] = beam_size
        return self.cap_model(input_dict)["seq"].cpu()