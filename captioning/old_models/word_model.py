# -*- coding: utf-8 -*-

import random

import numpy as np
import torch
import torch.nn as nn

import sys
import os
sys.path.append(os.getcwd())
import models.utils as utils

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

        if hasattr(encoder, "use_hidden") and encoder.use_hidden:
            assert encoder.network.hidden_size == decoder.model.hidden_size, \
                "hidden size not compatible while use hidden!"
            assert encoder.network.num_layers == decoder.model.num_layers, \
                """number of layers not compatible while use hidden!
                please either set use_hidden as False or use the same number of layers"""

        if "freeze_encoder" in kwargs and kwargs["freeze_encoder"]:
            for param in self.encoder.parameters():
                param.requires_grad = False

    @classmethod
    def set_index(cls, start_idx, end_idx):
        cls.start_idx = start_idx
        cls.end_idx = end_idx

    def forward(self, *input, **kwargs):
        """
        an encoder first encodes audio feature into an embedding sequence, obtaining 
        `encoded`: {
            fc_embs: [
            attn_embs: [N, attn_max_len, attn_emb_dim],
            state: rnn style hidden states, [num_dire * num_layers, N, hs_enc],
            attn_emb_lens: [N,]
        }
        """
        if len(input) == 4:
            feats, feat_lens, caps, cap_lens = input
            encoded = self.encoder(feats, feat_lens)
            output = self.train_forward(encoded, caps, cap_lens, **kwargs)
        elif len(input) == 2:
            feats, feat_lens = input
            encoded = self.encoder(feats, feat_lens)
            output = self.inference_forward(encoded, **kwargs)
        else:
            raise Exception("Number of input should be either 4 (feats, feat_lens, caps, cap_lens) or 2 (feats, feat_lens)")

        return output

    def prepare_output(self, encoded, output, max_length):
        N = encoded["fc_embs"].size(0)
        output["seqs"] = torch.empty(N, max_length, dtype=torch.long).fill_(self.end_idx)
        output["logits"] = torch.empty(N, max_length, self.vocab_size).to(encoded["fc_embs"].device)
        output["sampled_logprobs"] = torch.zeros(output["seqs"].size(0), max_length)

    def train_forward(self, encoded, caps, cap_lens, **kwargs):
        if kwargs["ss_ratio"] != 1: # scheduled sampling training
            return self.stepwise_forward(encoded, caps, cap_lens, **kwargs)
        cap_max_len = caps.size(1)
        output = {}
        self.prepare_output(encoded, output, cap_max_len - 1)
        fc_embs = encoded["fc_embs"].unsqueeze(1).repeat(1, cap_max_len - 1, 1) # [N, cap_max_len-1, src_emb_dim]
        # init_state = encoded["audio_embeds_pooled"].unsqueeze(0)
        decoder_output = self.decoder(word=caps[:, :-1], state=encoded["state"], fc_embs=fc_embs)
        # decoder_output = self.decoder(word=caps[:, :-1], state=init_state, enc_mem=enc_mem)
        self.train_process(output, decoder_output, cap_lens)
        return output

    def train_process(self, output, decoder_output, cap_lens):
        output.update(decoder_output)

    def inference_forward(self, encoded, **kwargs):
        # optional sampling keyword arguments
        method = kwargs.get("method", "greedy")
        max_length = kwargs.get("max_length", self.max_length)
        if method == "beam":
            beam_size = kwargs.get("beam_size", 5)
            return self.beam_search(encoded, max_length, beam_size)
        return self.stepwise_forward(encoded, None, None, **kwargs) 

    def stepwise_forward(self, encoded, caps, cap_lens, **kwargs):
        """Step-by-step decoding, when `caps` is provided, it means teacher forcing training"""
        if cap_lens is not None: # scheduled sampling training
            max_length = max(cap_lens) - 1
        else: # inference
            max_length = kwargs.get("max_length", self.max_length)
        decoder_input = {}
        output = {}
        self.prepare_output(encoded, output, max_length)
        # start sampling
        for t in range(max_length):
            self.decode_step(decoder_input, encoded, caps, output, t, **kwargs)
            if caps is None: # decide whether to stop when sampling
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

    def decode_step(self, decoder_input, encoded, caps, output, t, **kwargs):
        """Decoding operation of timestep t"""
        self.prepare_decoder_input(decoder_input, encoded, caps, output, t, **kwargs)
        # feed to the decoder to get states and logits
        output_t = self.decoder(**decoder_input)
        logits_t = output_t["logits"].squeeze(1)
        # sample the next input word and get the corresponding logits
        sampled = self.sample_next_word(logits_t, **kwargs)
        self.stepwise_process_step(output, output_t, t, sampled)
    
    def prepare_decoder_input(self, decoder_input, encoded, caps, output, t, **kwargs):
        """Prepare the input dict `decoder_input` for the decoder and timestep t"""
        if t == 0:
            decoder_input["state"] = encoded["state"]
            # decoder_input["state"] = encoded["audio_embeds_pooled"].unsqueeze(0)
            # decoder_input["enc_mem"] = encoded["audio_embeds_pooled"].unsqueeze(1)
            decoder_input["fc_embs"] = encoded["fc_embs"].unsqueeze(1)
            w_t = torch.tensor([self.start_idx,] * output["seqs"].size(0)).long()
        else:
            w_t = output["seqs"][:, t - 1]
            if caps is not None and random.random() < kwargs["ss_ratio"]: # training, scheduled sampling
                w_t = caps[:, t]
            decoder_input["state"] = output["state"]
        # w_t: [N,]
        decoder_input["word"] = w_t.unsqueeze(1)
    
    def stepwise_process_step(self, output, output_t, t, sampled):
        """Postprocessing (save output values) after each timestep t"""
        output["logits"][:, t, :] = output_t["logits"].squeeze(1)
        output["seqs"][:, t] = sampled["w_t"]
        output["sampled_logprobs"][:, t] = sampled["probs"]
        output["state"] = output_t["states"]

    def stepwise_process(self, output):
        """Postprocessing after the whole step-by-step autoregressive decoding"""
        pass

    def sample_next_word(self, logits, **kwargs):
        """Sample the next word, given probs output by the decoder"""
        method = kwargs.get("method", "greedy")
        temp = kwargs.get("temp", 1)
        logprobs = torch.log_softmax(logits, dim=1)
        if method == "greedy":
            sampled_logprobs, w_t = torch.max(logprobs.detach(), 1)
        else:
            prob_prev = torch.exp(logprobs / temp)
            w_t = torch.multinomial(prob_prev, 1)
            # w_t: [N, 1]
            sampled_logprobs = logprobs.gather(1, w_t).squeeze(1)
            w_t = w_t.view(-1)
        w_t = w_t.detach().long()

        # sampled_logprobs: [N,], w_t: [N,]
        return {"w_t": w_t, "probs": sampled_logprobs}

    def beam_search(self, encoded, max_length, beam_size):
        output = {}
        self.prepare_output(encoded, output, max_length)
        # instance by instance beam seach
        for i in range(encoded["fc_embs"].size(0)):
            output_i = {}
            self.prepare_beamsearch_output(output_i, beam_size, encoded, max_length)
            decoder_input = {}
            for t in range(max_length):
                output_t = self.beamsearch_step(decoder_input, encoded, output_i, i, t, beam_size)
                logits_t = output_t["logits"].squeeze(1)
                logprobs_t = torch.log_softmax(logits_t, dim=1)
                logprobs_t = output_i["top_k_logprobs"].unsqueeze(1).expand_as(logprobs_t) + logprobs_t
                if t == 0: # for the first step, all k seqs will have the same probs
                    top_k_logprobs, top_k_words = logprobs_t[0].topk(beam_size, 0, True, True)
                else: # unroll and find top logprobs, and their unrolled indices
                    top_k_logprobs, top_k_words = logprobs_t.view(-1).topk(beam_size, 0, True, True)
                output_i["top_k_logprobs"] = top_k_logprobs
                output_i["prev_word_inds"] = top_k_words // self.vocab_size  # [beam_size,]
                output_i["next_word_inds"] = top_k_words % self.vocab_size  # [beam_size,]
                if t == 0:
                    output_i["seqs"] = output_i["next_word_inds"].unsqueeze(1)
                else:
                    output_i["seqs"] = torch.cat([output_i["seqs"][output_i["prev_word_inds"]], 
                                                  output_i["next_word_inds"].unsqueeze(1)], dim=1)
                self.beamsearch_process_step(output_i, output_t)
            self.beamsearch_process(output, output_i, i)
        return output

    def prepare_beamsearch_output(self, output, beam_size, encoded, max_length):
        output["top_k_logprobs"] = torch.zeros(beam_size).to(encoded["fc_embs"].device)

    def beamsearch_step(self, decoder_input, encoded, output, i, t, beam_size):
        self.prepare_beamsearch_decoder_input(decoder_input, encoded, output, i, t, beam_size)
        output_t = self.decoder(**decoder_input)
        # decoder_input["state"] = output_t["states"]
        return output_t

    def prepare_beamsearch_decoder_input(self, decoder_input, encoded, output, i, t, beam_size):
        if t == 0:
            # enc_mem = encoded["fc_embs"][i].reshape(1, -1).repeat(beam_size, 1)
            # enc_mem = enc_mem.unsqueeze(1) # [beam_size, 1, enc_mem_size]
            # decoder_input["enc_mem"] = enc_mem
            decoder_input["fc_embs"] = utils.repeat_tensor(encoded["fc_embs"][i], beam_size)

            # decoder_input["state"] = utils.repeat_tensor(encoded["fc_embs"][i], beam_size)
            state = encoded["state"]
            if state is not None: # state: [num_layers, N, enc_hid_size]
                state = state[:, i, :].unsqueeze(1).repeat(1, beam_size, 1)
                state = state.contiguous() # [num_layers, beam_size, enc_hid_size]

            decoder_input["state"] = state

            w_t = torch.tensor([self.start_idx,] * beam_size).long()
        else:
            w_t = output["next_word_inds"]
            decoder_input["state"] = output["state"][:, output["prev_word_inds"], :].contiguous()
        decoder_input["word"] = w_t.unsqueeze(1)
            
    def beamsearch_process_step(self, output, output_t):
        output["state"] = output_t["states"]

    def beamsearch_process(self, output, output_i, i):
        output["seqs"][i] = output_i["seqs"][0]


class CaptionSentenceModel(CaptionModel):

    def __init__(self, encoder, decoder, seq_output_size, **kwargs):
        super(CaptionSentenceModel, self).__init__(encoder, decoder, **kwargs)
        self.output_transform = nn.Sequential()
        if decoder.model.hidden_size != seq_output_size:
            self.output_transform = nn.Linear(decoder.model.hidden_size, seq_output_size)

    def prepare_output(self, encoded, output, max_length):
        super(CaptionSentenceModel, self).prepare_output(encoded, output, max_length)
        output["hiddens"] = torch.zeros(output["seqs"].size(0), max_length, self.decoder.model.hidden_size).to(encoded["audio_embeds"].device)

    def train_process(self, output, decoder_output, cap_lens):
        super(CaptionSentenceModel, self).train_process(output, decoder_output, cap_lens)
        # obtain sentence outputs
        seq_outputs = utils.mean_with_lens(output["output"], cap_lens - 1)
        # seq_outputs: [N, dec_hid_size]
        seq_outputs = self.output_transform(seq_outputs)
        output["seq_outputs"] = seq_outputs

    def stepwise_process_step(self, output, output_t, t, sampled):
        super(CaptionSentenceModel, self).stepwise_process_step(output, output_t, t, sampled)
        output["hiddens"][:, t, :] = output_t["output"].squeeze(1)

    def stepwise_process(self, output):
        seqs = output["seqs"]
        lens = torch.where(seqs == self.end_idx, torch.zeros_like(seqs), torch.ones_like(seqs)).sum(dim=1)
        seq_outputs = utils.mean_with_lens(output["hiddens"], lens)
        seq_outputs = self.output_transform(seq_outputs)
        output["seq_outputs"] = seq_outputs

