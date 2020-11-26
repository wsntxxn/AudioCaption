import torch
import torch.nn as nn

from models.WordModel import CaptionModel

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
        nn.init.kaiming_uniform_(self.h2attn.weight)

    def forward(self, h_dec, h_enc, src_lens):
        """
        Args:
            h_dec: decoder hidden state, [N, hs_dec]
            h_enc: encoder hiddens/outputs, [N, src_max_len, hs_enc]
            src_lens: source (encoder input) lengths, [N, ]
        """
        N = h_enc.size(0)
        src_max_len = h_enc.size(1)
        h_dec = h_dec.unsqueeze(1).repeat(1, src_max_len, 1) # [N, src_max_len, hs_dec]

        attn_input = torch.cat((h_dec, h_enc), dim=-1)
        attn_out = torch.tanh(self.h2attn(attn_input)) # [N, src_max_len, attn_size]

        v = self.v.repeat(N, 1).unsqueeze(1) # [N, 1, attn_size]
        # score = torch.bmm(v, attn_out.permute(0, 2, 1)).squeeze(1) # [N, src_max_len]
        score = (v@attn_out.permute(0, 2, 1)).squeeze(1) # [N, src_max_len]

        idxs = torch.arange(src_max_len).repeat(N).view(N, src_max_len)
        mask = (idxs < src_lens.view(-1, 1)).to(h_dec.device)

        score = score.masked_fill(mask == 0, -1e10)
        weights = torch.softmax(score, dim=-1) # [N, src_max_len]
        # ctx = torch.bmm(weights.unsqueeze(1), h_enc).squeeze(1) # [N, hs_enc]
        ctx = (weights.unsqueeze(1)@h_enc).squeeze(1) # [N, hs_enc]

        return ctx, weights


class Seq2SeqAttnModel(CaptionModel):

    def __init__(self, encoder, decoder, **kwargs):
        super(Seq2SeqAttnModel, self).__init__(encoder, decoder, **kwargs)

    def train_forward(self, encoded, caps, cap_lens, **kwargs):
        # Bahdanau attention only supports step-by-step implementation, so we implement forward in 
        # step-by-step manner whether in training or evaluation
        return self.stepwise_forward(encoded, caps, cap_lens, **kwargs)

    def prepare_output(self, encoded, output, max_length):
        super(Seq2SeqAttnModel, self).prepare_output(encoded, output, max_length)
        attn_weights = torch.empty(output["seqs"].size(0), max(encoded["audio_embeds_lens"]), max_length)
        output["attn_weights"] = attn_weights

    def prepare_decoder_input(self, decoder_input, encoded, caps, output, t, **kwargs):
        super(Seq2SeqAttnModel, self).prepare_decoder_input(decoder_input, encoded, caps, output, t, **kwargs)
        if t == 0:
            decoder_input["enc_mem"] = encoded["audio_embeds"]
            decoder_input["enc_mem_lens"] = encoded["audio_embeds_lens"]
            if encoded["state"] is None:
                state = self.decoder.init_hidden(output["seqs"].size(0))
                state = state.to(encoded["audio_embeds"].device)
                decoder_input["state"] = state
        # decoder_input: { "word": ..., "state": ..., "enc_mem": ..., "enc_mem_lens": ... }

    def stepwise_process_step(self, output, output_t, t, sampled):
        super(Seq2SeqAttnModel, self).stepwise_process_step(output, output_t, t, sampled)
        output["attn_weights"][:, :, t] = output_t["weights"]

    def prepare_beamsearch_output(self, output, beam_size, encoded, max_length):
        super(Seq2SeqAttnModel, self).prepare_beamsearch_output(output, beam_size, encoded, max_length)
        output["attn_weights"] = torch.empty(beam_size, max(encoded["audio_embeds_lens"]), max_length)

    def prepare_beamsearch_decoder_input(self, decoder_input, encoded, output, i, t, beam_size):
        super(Seq2SeqAttnModel, self).prepare_beamsearch_decoder_input(decoder_input, encoded, output, i, t, beam_size)
        if t == 0:
            enc_mem = encoded["audio_embeds"][i]
            decoder_input["enc_mem"] = enc_mem.unsqueeze(0).repeat(beam_size, 1, 1)
            enc_mem_lens = encoded["audio_embeds_lens"][i]
            decoder_input["enc_mem_lens"] = enc_mem_lens.repeat(beam_size)
            if decoder_input["state"] is None:
                decoder_input["state"] = self.decoder.init_hidden(beam_size)
                decoder_input["state"] = decoder_input["state"].to(decoder_input["enc_mem"].device)

    def beamsearch_decode_step(self, decoder_input, encoded, output, i, t, beam_size):
        output_t = super(Seq2SeqAttnModel, self).beamsearch_decode_step(decoder_input, encoded, output, i, t, beam_size)
        output["attn_weights"][:, :, t] = output_t["weights"]
        return output_t

    def beamsearch_process_step(self, output):
        output["attn_weights"] = output["attn_weights"][output["prev_word_inds"], :, :]

    def beamsearch_process(self, output, output_i, i):
        output["seqs"][i] = output_i["seqs"][0]
        output["attn_weights"][i] = output_i["attn_weights"][0]


# class Seq2SeqAttnInstanceModel(Seq2SeqAttnModel):

    # def __init__(self, encoder, decoder, num_instance, instance_embed_size, **kwargs):
        # super(Seq2SeqAttnInstanceModel, self).__init__(encoder, decoder, **kwargs)
        # self.instance_embedding = nn.Embedding(num_instance, instance_embed_size)
        # nn.init.kaiming_uniform_(self.instance_embedding.weight)
        # self.audio_embed2instance_label = nn.Sequential(
            # nn.Linear(self.embed_size, 16),
            # nn.ReLU(),
            # nn.Linear(16, num_instance)
        # )

    # def forward(self, *input, **kwargs):
        # if len(input) < 2 or len(input) > 5:
            # raise Exception("Input should be [feats, feat_lens, (caps), (cap_lens), (instance_labels)]!")

        # if len(input) == 5:
            # train_mode = kwargs.get("train_mode", "tf")
            # assert train_mode in ("tf", "sample"), "unknown training mode"
            # kwargs["train_mode"] = train_mode
            # # "tf": teacher forcing training, "sample": no teacher forcing training
            # feats, feat_lens, caps, cap_lens, cap_idxs = input
            # instance_embeds = self.instance_embedding(cap_idxs)
        # elif len(input) == 4:
            # feats, feat_lens, caps, cap_lens = input
        # elif len(input) == 3:
            # feats, feat_lens, instance_labels = input
            # instance_embeds = torch.matmul(instance_labels, self.instance_embedding.weight)
            # caps = None
            # cap_lens = None
        # elif len(input) == 2:
            # feats, feat_lens = input
            # caps = None
            # cap_lens = None

        # encoded = self.encoder(feats, feat_lens)
        # # encoded: {
        # #     audio_embeds: [N, emb_dim]
        # #     audio_embeds_time: [N, src_max_len, emb_dim]
        # #     state: rnn style hidden states, [num_dire * num_layers, N, hs_enc]
        # #     audio_embeds_lens: [N, ] }
        # if len(input) == 4 or len(input) == 2:
            # instance_labels = torch.softmax(self.audio_embed2instance_label(encoded["audio_embeds"]), dim=-1)
        # instance_embeds = torch.matmul(instance_labels, self.instance_embedding.weight)
        # output = self.sample(encoded, instance_embeds, caps, cap_lens, **kwargs)
        # return output

    # def sample(self, encoded, instance_embeds, caps, cap_lens, **kwargs):
        # # optional keyword arguments
        # method = kwargs.get("method", "greedy")
        # temp = kwargs.get("temperature", 1.0)
        # max_length = kwargs.get("max_length", self.max_length)

        # if cap_lens is not None:
            # max_length = max(cap_lens)

        # assert method in ("greedy", "sample", "beam"), "unknown sampling method"

        # if method == "beam":
            # beam_size = kwargs.get("beam_size", 5)
            # return self.sample_beam(
                # encoded, instance_embeds, max_length=max_length, beam_size=beam_size)

        # h_enc = encoded["audio_embeds_time"]
        # src_lens = encoded["audio_embeds_lens"]
        # h_t = encoded["state"]

        # N = h_enc.size(0)
        # seqs = torch.empty(N, max_length, dtype=torch.long).fill_(self.end_idx)
        # logits = torch.empty(N, max_length, self.vocab_size).to(h_enc.device)
        # sampled_logprobs = torch.zeros(N, max_length)
        # attn_weights = torch.empty(N, max(src_lens), max_length)

        # # start sampling
        # for t in range(max_length):
            # # prepare input word/audio embedding
            # if t == 0:
                # e_t = encoded["audio_embeds"]
                # if h_t is None:
                    # h_t = self.decoder.init_hidden(N)
                    # h_t = h_t.to(h_enc.device)
            # else:
                # e_t = self.word_embeddings(w_t)
                # e_t = self.dropoutlayer(e_t)
            # # e_t: [N, emb_dim]

            # e_t = torch.cat((e_t, instance_embeds), dim=-1)
            # output_t = self.decoder(e_t, h_t, attn=self.attn, h_enc=h_enc, src_lens=src_lens)
            # attn_weights[:, :, t] = output_t["weights"]
            # h_t = output_t["states"]

            # # outputs["logits"]: [N, vocab_size]
            # logits_t = output_t["logits"]
            # logits[:, t, :] = logits_t

            # if caps is None or kwargs["train_mode"] == "sample":
                # # sample the next input word and get the corresponding logits
                # sampled = self.sample_next_word(logits_t, method, temp)
                # w_t = sampled["w_t"]
                # sampled_logprobs[:, t] = sampled["probs"]

                # seqs[:, t] = w_t
                
                # if caps is None: # decide whether to stop when sampling
                    # if t == 0:
                        # unfinished = w_t != self.end_idx
                    # else:
                        # unfinished = unfinished * (w_t != self.end_idx)
                    # seqs[:, t][~unfinished] = self.end_idx
                    # if unfinished.sum() == 0:
                        # break
            # else:
                # w_t = caps[:, t]

        # output = {
            # "seqs": seqs, 
            # "logits": logits, 
            # "sampled_logprobs": sampled_logprobs, 
            # "attn_weights": attn_weights
        # }
        # return output

    # def sample_beam(self, encoded, instance_embeds, **kwargs):
        # N = encoded["audio_embeds"].size(0)
        # seqs = torch.zeros(N, kwargs["max_length"], dtype=torch.long)

        # k = kwargs["beam_size"]
        # max_length = kwargs["max_length"]
        # state = encoded["state"]

        # attn_weights = torch.empty(N, max(encoded["audio_embeds_lens"]), max_length)

        # # beam search sentence by sentence
        # for i in range(N):
            # top_k_logprobs = torch.zeros(k).to(encoded["audio_embeds"].device)
            # h_t = state if state is None else state[:, i, :]
            # if h_t is not None:
                # h_t = h_t.unsqueeze(1).expand(-1, k, h_t.size(-1))
            # h_enc = encoded["audio_embeds_time"][i]
            # h_enc = h_enc.unsqueeze(0).expand(k, -1, h_enc.size(-1))
            # src_lens = encoded["audio_embeds_lens"][i]
            # src_lens = src_lens.expand(k)
            # instance_embeds_i = instance_embeds[i].unsqueeze(0).expand(k, -1)
            
            # attn_weight_i = torch.empty(k, max(encoded["audio_embeds_lens"]), max_length)

            # for t in range(max_length):
                # if t == 0:
                    # e_t = encoded["audio_embeds"][i].reshape(1, -1).expand(k, -1)
                    # if h_t is None:
                        # h_t = self.decoder.init_hidden(k)
                        # h_t = h_t.to(h_enc.device)
                # else:
                    # e_t = self.word_embeddings(next_word_inds)
                    # e_t = self.dropoutlayer(e_t)
                
                # e_t = torch.cat((e_t, instance_embeds_i), dim=-1)
                # # e_t: [k, emb_dim]

                # output_t = self.decoder(e_t, h_t, attn=self.attn, h_enc=h_enc, src_lens=src_lens)
                # attn_weight_i[:, :, t] = output_t["weights"]
                # h_t = output_t["states"]

                # logprobs_t = torch.log_softmax(output_t["logits"], dim=-1)
                # # logprobs_t: [k, vocab_size]

                # # calculate the joint probability up to the timestep t
                # logprobs_t = top_k_logprobs.unsqueeze(1).expand_as(logprobs_t) + logprobs_t
                
                # if t == 0:
                    # # for the first step, all k seqs will have the same probs
                    # top_k_logprobs, top_k_words = logprobs_t[0].topk(k, 0, True, True)
                # else:
                    # # unroll and find top logprobs, and their unrolled indices
                    # top_k_logprobs, top_k_words = logprobs_t.view(-1).topk(k, 0, True, True)

                # # convert unrolled indices to actual indices of scores
                # prev_word_inds = top_k_words / self.vocab_size  # [k,]
                # next_word_inds = top_k_words % self.vocab_size  # [k,]

                # # add new words to sequences
                # if t == 0:
                    # seqs_i = next_word_inds.unsqueeze(1)
                    # # seqs: [k, 1]
                # else:
                    # seqs_i = torch.cat([seqs_i[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # [k, t + 1]

                # h_t = h_t[:, prev_word_inds, :].contiguous()
                # attn_weight_i = attn_weight_i[prev_word_inds, :, :]

            # seqs[i, :] = seqs_i[0]
            # attn_weights[i, :, :] = attn_weight_i[0]

        # return {"seqs": seqs, "weights": attn_weights}
