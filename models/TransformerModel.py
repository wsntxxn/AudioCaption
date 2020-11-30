# -*- coding: utf-8 -*-
import random
import torch

from models.WordModel import CaptionModel

class TransformerModel(CaptionModel):

    def __init__(self, encoder, decoder, **kwargs):
        super(TransformerModel, self).__init__(encoder, decoder, **kwargs)

    def train_forward(self, encoded, caps, cap_lens, **kwargs):
        if kwargs["ss_ratio"] != 1: # scheduled sampling training
            return self.stepwise_forward(encoded, caps, cap_lens, **kwargs)
        cap_max_len = caps.size(1)
        output = {}
        self.prepare_output(encoded, output, cap_max_len)
        caps_padding_mask = (caps == self.pad_idx)
        caps_padding_mask = torch.cat([torch.tensor([[False]] * caps.size(0), device=caps.device), caps_padding_mask[:, :-1]], dim=1)
        decoder_output = self.decoder(words=[encoded["audio_embeds_pooled"].unsqueeze(1), caps[:, :-1]], 
                                      enc_mem=encoded["audio_embeds"],
                                      enc_mem_lens=encoded["audio_embeds_lens"],
                                      caps_padding_mask=caps_padding_mask)
        self.train_process(output, decoder_output, cap_lens)
        return output

    def decode_step(self, decoder_input, encoded, caps, output, t, **kwargs):
        """Decoding operation of timestep t"""
        self.prepare_decoder_input(decoder_input, encoded, caps, output, t, **kwargs)
        # feed to the decoder to get states and logits
        output_t = self.decoder(**decoder_input)
        logits_t = output_t["logits"][:, -1, :]
        output_t["logits"] = logits_t.unsqueeze(1)
        # sample the next input word and get the corresponding logits
        sampled = self.sample_next_word(logits_t, **kwargs)
        self.stepwise_process_step(output, output_t, t, sampled)

    def prepare_decoder_input(self, decoder_input, encoded, caps, output, t, **kwargs):
        """Prepare the input dict `decoder_input` for the decoder and timestep t"""
        super(TransformerModel, self).prepare_decoder_input(decoder_input, encoded, caps, output, t, **kwargs)
        if t == 0:
            decoder_input["enc_mem"] = encoded["audio_embeds"]
            decoder_input["enc_mem_lens"] = encoded["audio_embeds_lens"]
            words = [encoded["audio_embeds_pooled"].unsqueeze(1),]
        else:
            words = [encoded["audio_embeds_pooled"].unsqueeze(1), output["seqs"][:, :t]]
            if caps is not None and random.random() < kwargs["ss_ratio"]: # training, scheduled sampling
                words = [encoded["audio_embeds_pooled"].unsqueeze(1), caps[:, :t]]
        decoder_input["words"] = words
        caps_padding_mask = torch.tensor([[False]] * output["seqs"].size(0), device=encoded["audio_embeds"].device)
        if len(words) > 1:
            caps_padding_mask = torch.cat([caps_padding_mask, (words[1] == self.pad_idx).to(encoded["audio_embeds"].device)], dim=1)
        decoder_input["caps_padding_mask"] = caps_padding_mask

    def beamsearch_step(self, decoder_input, encoded, output, i, t, beam_size):
        self.prepare_beamsearch_decoder_input(decoder_input, encoded, output, i, t, beam_size)
        output_t = self.decoder(**decoder_input)
        output_t["logits"] = output_t["logits"][:, -1, :].unsqueeze(1)
        return output_t

    def prepare_beamsearch_decoder_input(self, decoder_input, encoded, output, i, t, beam_size):
        w_0 = encoded["audio_embeds_pooled"][i].reshape(1, -1).repeat(beam_size, 1).unsqueeze(1)
        if t == 0:
            enc_mem_lens = encoded["audio_embeds_lens"][i]
            decoder_input["enc_mem_lens"] = enc_mem_lens.repeat(beam_size)
            enc_mem = encoded["audio_embeds"][i, :enc_mem_lens]
            decoder_input["enc_mem"] = enc_mem.unsqueeze(0).repeat(beam_size, 1, 1)

            # w_t = torch.tensor([self.start_idx,] * beam_size).long()
            words = [w_0,]
        else:
            words = [w_0, output["seqs"]]
        decoder_input["words"] = words
        caps_padding_mask = torch.tensor([[False]] * beam_size, device=encoded["audio_embeds"].device)
        if len(words) > 1:
            caps_padding_mask = torch.cat([caps_padding_mask, (words[1] == self.pad_idx)], dim=1)
        decoder_input["caps_padding_mask"] = caps_padding_mask
