# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from captioning.models.utils import mean_with_lens

class CaptionModel(nn.Module):
    """
    Encoder-decoder captioning model.
    """

    pad_idx = 0
    start_idx = 1
    end_idx = 2
    max_length = 20

    def __init__(self, encoder: nn.Module, decoder: nn.Module, **kwargs):
        super(CaptionModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab_size = decoder.vocab_size
        freeze_encoder = kwargs.get("freeze_encoder", False)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.check_decoder_compatibility()

    def check_decoder_compatibility(self):
        assert isinstance(self.decoder, self.compatible_decoders), \
            f"{self.decoder.__class__.__name__} is incompatible with {self.__class__.__name__}, please use decoder in {self.compatible_decoders} "

    @classmethod
    def set_index(cls, start_idx, end_idx):
        cls.start_idx = start_idx
        cls.end_idx = end_idx

    def forward(self, input_dict):
        """
        input_dict: {
            (required)
            mode: train/inference,
            raw_feats,
            raw_feat_lens,
            fc_feats,
            attn_feats,
            attn_feat_lens,
            [sample_method: greedy],
            [temp: 1.0] (in case of no teacher forcing)

            (optional, mode=train)
            caps,
            cap_lens,
            ss_ratio,

            (optional, mode=inference)
            sample_method: greedy/beam,
            max_length,
            temp,
            beam_size (optional, sample_method=beam),
        }
        """
        encoder_input_keys = ["raw_feats", "raw_feat_lens", "fc_feats", "attn_feats", "attn_feat_lens"]
        encoder_input = { key: input_dict[key] for key in encoder_input_keys }
        encoder_output_dict = self.encoder(encoder_input)
        if input_dict["mode"] == "train":
            forward_dict = { "mode": "train", "sample_method": "greedy", "temp": 1.0 }
            forward_keys = ["caps", "cap_lens", "ss_ratio"]
            for key in forward_keys:
                forward_dict[key] = input_dict[key]
            forward_dict.update(encoder_output_dict)
            output = self.train_forward(forward_dict)
        elif input_dict["mode"] == "inference":
            forward_dict = {"mode": "inference"}
            default_args = { "sample_method": "greedy", "max_length": self.max_length, "temp": 1.0 }
            forward_keys = ["sample_method", "max_length", "temp"]
            for key in forward_keys:
                if key in input_dict:
                    forward_dict[key] = input_dict[key]
                else:
                    forward_dict[key] = default_args[key]
            if forward_dict["sample_method"] == "beam":
                if "beam_size" in input_dict:
                    forward_dict["beam_size"] = input_dict["beam_size"]
                else:
                    forward_dict["beam_size"] = 3
            forward_dict.update(encoder_output_dict)
            output = self.inference_forward(forward_dict)
        else:
            raise Exception("mode should be either 'train' or 'inference'")

        return output

    def prepare_output(self, input_dict):
        output = {}
        batch_size = input_dict["fc_embs"].size(0)
        if input_dict["mode"] == "train":
            max_length = input_dict["caps"].size(1) - 1
        elif input_dict["mode"] == "inference":
            max_length = input_dict["max_length"]
        else:
            raise Exception("mode should be either 'train' or 'inference'")
        device = input_dict["fc_embs"].device
        output["seqs"] = torch.empty(batch_size, max_length, dtype=torch.long).fill_(self.end_idx)
        output["logits"] = torch.empty(batch_size, max_length, self.vocab_size).to(device)
        output["sampled_logprobs"] = torch.zeros(batch_size, max_length)
        output["embeds"] = torch.empty(batch_size, max_length, self.decoder.d_model).to(device)
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
        return self.stepwise_forward(input_dict) 

    def stepwise_forward(self, input_dict):
        """Step-by-step decoding"""
        output = self.prepare_output(input_dict)
        max_length = output["seqs"].size(1)
        # start sampling
        for t in range(max_length):
            input_dict["t"] = t
            self.decode_step(input_dict, output)
            if input_dict["mode"] == "inference": # decide whether to stop when sampling
                unfinished_t = output["seqs"][:, t] != self.end_idx
                if t == 0:
                    unfinished = unfinished_t
                else:
                    unfinished *= unfinished_t
                output["seqs"][:, t][~unfinished] = self.end_idx
                if unfinished.sum() == 0:
                    break
        self.stepwise_process(output)
        return output

    def decode_step(self, input_dict, output):
        """Decoding operation of timestep t"""
        decoder_input = self.prepare_decoder_input(input_dict, output)
        # feed to the decoder to get logits
        output_t = self.decoder(decoder_input)
        logits_t = output_t["logits"]
        # assert logits_t.ndim == 3
        if logits_t.size(1) == 1:
            logits_t = logits_t.squeeze(1)
            embeds_t = output_t["embeds"].squeeze(1)
        elif logits_t.size(1) > 1:
            logits_t = logits_t[:, -1, :]
            embeds_t = output_t["embeds"][:, -1, :]
        else:
            raise Exception("no logits output")
        # sample the next input word and get the corresponding logits
        sampled = self.sample_next_word(logits_t,
                                        method=input_dict["sample_method"],
                                        temp=input_dict["temp"])

        output_t.update(sampled)
        output_t["t"] = input_dict["t"]
        output_t["logits"] = logits_t
        output_t["embeds"] = embeds_t
        self.stepwise_process_step(output, output_t)

    def prepare_decoder_input(self, input_dict, output):
        """Prepare the inp ut dict for the decoder"""
        raise NotImplementedError
    
    def stepwise_process_step(self, output, output_t):
        """Postprocessing (save output values) after each timestep t"""
        t = output_t["t"]
        output["logits"][:, t, :] = output_t["logits"]
        output["seqs"][:, t] = output_t["word"]
        output["sampled_logprobs"][:, t] = output_t["probs"]
        output["embeds"][:, t, :] = output_t["embeds"]

    def stepwise_process(self, output):
        """Postprocessing after the whole step-by-step autoregressive decoding"""
        pass

    def sample_next_word(self, logits, method, temp):
        """Sample the next word, given probs output by the decoder"""
        logprobs = torch.log_softmax(logits, dim=1)
        if method == "greedy":
            sampled_logprobs, word = torch.max(logprobs.detach(), 1)
        elif method == "sample":
            prob_prev = torch.exp(logprobs / temp)
            word = torch.multinomial(prob_prev, 1)
            # w_t: [N, 1]
            sampled_logprobs = logprobs.gather(1, word).squeeze(1)
            word = word.view(-1)
        else:
            raise Exception(f"sample method {method} not supported")
        word = word.detach().long()

        # sampled_logprobs: [N,], w_t: [N,]
        return {"word": word, "probs": sampled_logprobs}

    def beam_search(self, input_dict):
        output = self.prepare_output(input_dict)
        max_length = input_dict["max_length"]
        beam_size = input_dict["beam_size"]
        temp = input_dict["temp"]
        # instance by instance beam seach
        for i in range(output["seqs"].size(0)):
            output_i = self.prepare_beamsearch_output(input_dict)
            # decoder_input = {}
            input_dict["sample_idx"] = i
            for t in range(max_length):
                input_dict["t"] = t
                output_t = self.beamsearch_step(input_dict, output_i)
                #######################################
                # merge with previous beam and select the current max prob beam
                #######################################
                logits_t = output_t["logits"]
                if logits_t.size(1) == 1:
                    logits_t = logits_t.squeeze(1)
                elif logits_t.size(1) > 1:
                    logits_t = logits_t[:, -1, :]
                else:
                    raise Exception("no logits output")
                logprobs_t = torch.log_softmax(logits_t, dim=1)
                logprobs_t = torch.log_softmax(logprobs_t / temp, dim=1)
                logprobs_t = output_i["topk_logprobs"].unsqueeze(1).expand_as(logprobs_t) + logprobs_t
                if t == 0: # for the first step, all k seqs will have the same probs
                    topk_logprobs, topk_words = logprobs_t[0].topk(beam_size, 0, True, True)
                else: # unroll and find top logprobs, and their unrolled indices
                    topk_logprobs, topk_words = logprobs_t.view(-1).topk(beam_size, 0, True, True)
                topk_words = topk_words.cpu()
                output_i["topk_logprobs"] = topk_logprobs
                output_i["prev_words_beam"] = topk_words // self.vocab_size  # [beam_size,]
                output_i["next_word"] = topk_words % self.vocab_size  # [beam_size,]
                if t == 0:
                    output_i["seqs"] = output_i["next_word"].unsqueeze(1)
                else:
                    output_i["seqs"] = torch.cat((output_i["seqs"][output_i["prev_words_beam"]],
                                                  output_i["next_word"].unsqueeze(1)), dim=1)

                self.beamsearch_process_step(output_i, output_t)
            self.beamsearch_process(output, output_i, input_dict)
        return output

    def prepare_beamsearch_output(self, input_dict):
        beam_size = input_dict["beam_size"]
        device = input_dict["fc_embs"].device
        output = {
            "topk_logprobs": torch.zeros(beam_size).to(device)
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
        output["seqs"][i] = output_i["seqs"][0]


class CaptionSequenceModel(nn.Module):

    def __init__(self, model, seq_output_size):
        super().__init__()
        self.model = model
        if model.decoder.d_model != seq_output_size:
            self.output_transform = nn.Linear(model.decoder.d_model, seq_output_size)
        else:
            self.output_transform = lambda x: x

    def forward(self, input_dict):
        output = self.model(input_dict)

        if input_dict["mode"] == "train":
            lens = input_dict["cap_lens"] - 1
            # seq_outputs: [N, d_model]
        elif input_dict["mode"] == "inference":
            if "sample_method" in input_dict and input_dict["sample_method"] == "beam":
                return output
            seqs = output["seqs"]
            lens = torch.where(seqs == self.model.end_idx, torch.zeros_like(seqs), torch.ones_like(seqs)).sum(dim=1)
        else:
            raise Exception("mode should be either 'train' or 'inference'")
        seq_outputs = mean_with_lens(output["embeds"], lens)
        seq_outputs = self.output_transform(seq_outputs)
        output["seq_outputs"] = seq_outputs
        return output

