# -*- coding: utf-8 -*-

import random

import numpy as np
import torch
import torch.nn as nn

import utils.score_util as score_util
from utils.train_util import mean_with_lens

class CaptionModel(nn.Module):
    """
    Encoder-decoder captioning model.
    """

    start_idx = 1
    end_idx = 2
    max_length = 20

    def __init__(self, encoder, decoder, **kwargs):
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

    @classmethod
    def set_index(cls, start_idx, end_idx):
        cls.start_idx = start_idx
        cls.end_idx = end_idx

    def forward(self, *input, **kwargs):
        """
        an encoder first encodes audio feature into an embedding sequence, obtaining `encoded`: {
            audio_embeds: [N, enc_mem_max_len, enc_mem_size]
            state: rnn style hidden states, [num_dire * num_layers, N, hs_enc]
            audio_embeds_lens: [N,] 
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
        N = encoded["audio_embeds"].size(0)
        seqs = torch.empty(N, max_length, dtype=torch.long).fill_(self.end_idx)
        logits = torch.empty(N, max_length, self.vocab_size).to(encoded["audio_embeds"].device)
        # sampled_logprobs = torch.zeros(N, max_length)
        output["seqs"] = seqs
        output["logits"] = logits
        # output["sampled_logprobs"] = sampled_logprobs

    def train_forward(self, encoded, caps, cap_lens, **kwargs):
        if kwargs["ss_ratio"] != 1: # scheduled sampling training
            return self.stepwise_forward(encoded, caps, cap_lens, **kwargs)
        N, cap_max_len = caps.size(0), caps.size(1)
        output = {}
        self.prepare_output(encoded, output, cap_max_len - 1)
        enc_mem = mean_with_lens(encoded["audio_embeds"], 
                                 encoded["audio_embeds_lens"]) # [N, src_emb_dim]
        enc_mem = enc_mem.unsqueeze(1).repeat(1, cap_max_len - 1, 1) # [N, cap_max_len-1, src_emb_dim]
        decoder_output = self.decoder(word=caps[:, :-1], state=encoded["state"], enc_mem=enc_mem)
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
        return self.stepwise_forward(encoded, None, None) 

    def stepwise_forward(self, encoded, caps, cap_lens, **kwargs):
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
        self.prepare_decoder_input(decoder_input, encoded, caps, output, t, **kwargs)
        # feed to the decoder to get states and logits
        output_t = self.decoder(**decoder_input)
        decoder_input["state"] = output_t["states"]
        logits_t = output_t["logits"].squeeze(1)
        # sample the next input word and get the corresponding logits
        sampled = self.sample_next_word(logits_t, **kwargs)
        self.stepwise_process_step(output, output_t, t, sampled)
    
    def prepare_decoder_input(self, decoder_input, encoded, caps, output, t, **kwargs):
        # prepare input word
        if t == 0:
            decoder_input["state"] = encoded["state"]
            decoder_input["enc_mem"] = mean_with_lens(
                encoded["audio_embeds"], encoded["audio_embeds_lens"]).unsqueeze(1) # [N, 1, src_emb_dim]
            w_t = torch.tensor([self.start_idx,] * output["seqs"].size(0)).long()
        else:
            w_t = output["seqs"][:, t - 1]
            if caps is not None and random.random() < kwargs["ss_ratio"]: # training, scheduled sampling
                w_t = caps[:, t]
        # w_t: [N,]
        decoder_input["word"] = w_t.unsqueeze(1)
    
    def stepwise_process_step(self, output, output_t, t, sampled):
        output["logits"][:, t, :] = output_t["logits"].squeeze(1)
        output["seqs"][:, t] = sampled["w_t"]
        # output["sampled_logprobs"][:, t] = sampled["probs"]

    def stepwise_process(self, output):
        pass

    def sample_next_word(self, logits, **kwargs):
        """Sample the next word, given probs output by the decoder
        """
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
        for i in range(encoded["audio_embeds"].size(0)):
            output_i = {}
            self.prepare_beamsearch_output(output_i, beam_size, encoded, max_length)
            decoder_input = {}
            for t in range(max_length):
                output_t = self.beamsearch_decode_step(decoder_input, encoded, output_i, i, t, beam_size)
                logits_t = output_t["logits"].squeeze(1)
                logprobs_t = torch.log_softmax(logits_t, dim=1)
                logprobs_t = output_i["top_k_logprobs"].unsqueeze(1).expand_as(logprobs_t) + logprobs_t
                if t == 0: # for the first step, all k seqs will have the same probs
                    top_k_logprobs, top_k_words = logprobs_t[0].topk(beam_size, 0, True, True)
                else: # unroll and find top logprobs, and their unrolled indices
                    top_k_logprobs, top_k_words = logprobs_t.view(-1).topk(beam_size, 0, True, True)
                output_i["top_k_logprobs"] = top_k_logprobs
                output_i["prev_word_inds"] = top_k_words / self.vocab_size  # [beam_size,]
                output_i["next_word_inds"] = top_k_words % self.vocab_size  # [beam_size,]
                if t == 0:
                    output_i["seqs"] = output_i["next_word_inds"].unsqueeze(1)
                else:
                    output_i["seqs"] = torch.cat([output_i["seqs"][output_i["prev_word_inds"]], 
                                                  output_i["next_word_inds"].unsqueeze(1)], dim=1)
                self.beamsearch_process_step(output_i)
            self.beamsearch_process(output, output_i, i)
        return output

    def prepare_beamsearch_output(self, output, beam_size, encoded, max_length):
        output["top_k_logprobs"] = torch.zeros(beam_size).to(encoded["audio_embeds"].device)

    def beamsearch_decode_step(self, decoder_input, encoded, output, i, t, beam_size):
        self.prepare_beamsearch_decoder_input(decoder_input, encoded, output, i, t, beam_size)
        output_t = self.decoder(**decoder_input)
        decoder_input["state"] = output_t["states"]
        return output_t

    def prepare_beamsearch_decoder_input(self, decoder_input, encoded, output, i, t, beam_size):
        if t == 0:
            enc_mem = torch.mean(encoded["audio_embeds"][i, :encoded["audio_embeds_lens"][i], :], dim=0)
            enc_mem = enc_mem.reshape(1, -1).repeat(beam_size, 1)
            enc_mem = enc_mem.unsqueeze(1) # [beam_size, 1, enc_mem_size]
            decoder_input["enc_mem"] = enc_mem

            state = encoded["state"]
            if state is not None: # state: [num_layers, N, enc_hid_size]
                state = state[:, i, :].unsqueeze(1).repeat(1, beam_size, 1)
                state = state.contiguous() # [num_layers, beam_size, enc_hid_size]
            decoder_input["state"] = state
            w_t = torch.tensor([self.start_idx,] * beam_size).long()
        else:
            w_t = output["next_word_inds"]
            decoder_input["state"] = decoder_input["state"][:, output["prev_word_inds"], :].contiguous()
        decoder_input["word"] = w_t.unsqueeze(1)
            
    def beamsearch_process_step(self, output):
        pass

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
        seq_outputs = mean_with_lens(output["output"], cap_lens - 1)
        # seq_outputs: [N, dec_hid_size]
        seq_outputs = self.output_transform(seq_outputs)
        output["seq_outputs"] = seq_outputs

    def stepwise_process_step(self, output, output_t, t, sampled):
        super(CaptionSentenceModel, self).stepwise_process_step(output, output_t, t, sampled)
        output["hiddens"][:, t, :] = output_t["output"].squeeze(1)

    def stepwise_process(self, output):
        seqs = output["seqs"]
        lens = torch.where(seqs == self.end_idx, torch.zeros_like(seqs), torch.ones_like(seqs)).sum(dim=1)
        seq_outputs = mean_with_lens(output["hiddens"], lens)
        seq_outputs = self.output_transform(seq_outputs)
        output["seq_outputs"] = seq_outputs

# class CaptionInstanceModel(CaptionModel):

    # def __init__(self, encoder, decoder, num_instance, instance_embed_size, **kwargs):
        # super(CaptionInstanceModel, self).__init__(encoder, decoder, **kwargs)
        # self.word_embeddings = nn.Embedding(self.vocab_size, self.decoder.model.input_size)
        # nn.init.kaiming_uniform_(self.word_embeddings.weight)
        # self.instance_embedding = nn.Embedding(num_instance, instance_embed_size)
        # nn.init.kaiming_uniform_(self.instance_embedding.weight)

    # def forward(self, *input, **kwargs):
        # if len(input) != 5 and len(input) != 3:
            # raise Exception("number of input should be either 5 (feats, feat_lens, caps, cap_lens, cap_idxs) or 3 (feats, feat_lens, instance_labels)!")

        # if len(input) == 5:
            # train_mode = kwargs.get("train_mode", "tf")
            # assert train_mode in ("tf", "sample"), "unknown training mode"
            # kwargs["train_mode"] = train_mode
            # # "tf": teacher forcing training, "sample": no teacher forcing training
            # feats, feat_lens, caps, cap_lens, cap_idxs = input
            # instance_embeds = self.instance_embedding(cap_idxs)
        # else:
            # feats, feat_lens, instance_labels = input
            # instance_embeds = torch.matmul(instance_labels, self.instance_embedding.weight)
            # caps = None
            # cap_lens = None

        # encoded = self.encoder(feats, feat_lens)
        # # encoded: {
        # #     audio_embeds: [N, emb_dim]
        # #     audio_embeds_time: [N, src_max_len, emb_dim]
        # #     state: rnn style hidden states, [num_dire * num_layers, N, hs_enc]
        # #     audio_embeds_lens: [N, ]
        # encoded["audio_embeds"] = torch.cat((encoded["audio_embeds"], instance_embeds), dim=-1)
        # output = self.sample(encoded, caps, cap_lens, **kwargs)
        # return output

