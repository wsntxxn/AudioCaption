import random
import torch

from captioning.models.base_model import CaptionModel
from captioning.models.style_model import StyleCaptionModel
from captioning.models.utils import repeat_tensor
import captioning.models.decoder

class Seq2SeqAttnModel(CaptionModel):

    def __init__(self, encoder, decoder, **kwargs):
        if not hasattr(self, "compatible_decoders"):
            self.compatible_decoders = (
                captioning.models.decoder.BahAttnDecoder,
                captioning.models.decoder.BahAttnDecoder2,
                captioning.models.decoder.BahAttnDecoder3,
            )
        super().__init__(encoder, decoder, **kwargs)


    def seq_forward(self, input_dict):
        # Bahdanau attention only supports step-by-step implementation, so we implement forward in 
        # step-by-step manner whether in training or evaluation
        return self.stepwise_forward(input_dict)

    def prepare_output(self, input_dict):
        output = super().prepare_output(input_dict)
        attn_weights = torch.empty(output["seqs"].size(0),
                                   input_dict["attn_embs"].size(1),
                                   output["seqs"].size(1))
        output["attn_weights"] = attn_weights
        return output

    def prepare_decoder_input(self, input_dict, output):
        decoder_input = {
            "fc_embs": input_dict["fc_embs"],
            "attn_embs": input_dict["attn_embs"],
            "attn_emb_lens": input_dict["attn_emb_lens"]
        }
        t = input_dict["t"]
        ###############
        # determine input word
        ################
        if input_dict["mode"] == "train" and random.random() < input_dict["ss_ratio"]: # training, scheduled sampling
            word = input_dict["caps"][:, t]
        else:
            if t == 0:
                word = torch.tensor([self.start_idx,] * input_dict["fc_embs"].size(0)).long()
            else:
                word = output["seqs"][:, t-1]
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
        output["attn_weights"][:, :, t] = output_t["weights"]

    def prepare_beamsearch_output(self, input_dict):
        output = super().prepare_beamsearch_output(input_dict)
        beam_size = input_dict["beam_size"]
        max_length = input_dict["max_length"]
        output["attn_weights"] = torch.empty(beam_size,
                                             max(input_dict["attn_emb_lens"]),
                                             max_length)
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
            fc_embs = repeat_tensor(input_dict["fc_embs"][i], beam_size)
            output_i["fc_embs"] = fc_embs
        decoder_input["fc_embs"] = output_i["fc_embs"]

        ###############
        # prepare attn embeds
        ################
        if t == 0:
            attn_embs = repeat_tensor(input_dict["attn_embs"][i], beam_size)
            attn_emb_lens = repeat_tensor(input_dict["attn_emb_lens"][i], beam_size)
            output_i["attn_embs"] = attn_embs
            output_i["attn_emb_lens"] = attn_emb_lens
        decoder_input["attn_embs"] = output_i["attn_embs"]
        decoder_input["attn_emb_lens"] = output_i["attn_emb_lens"]

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
        output_i["attn_weights"][..., t] = output_t["weights"]
        output_i["attn_weights"] = output_i["attn_weights"][output_i["prev_words_beam"], ...]

    def beamsearch_process(self, output, output_i, input_dict):
        super().beamsearch_process(output, output_i, input_dict)
        i = input_dict["sample_idx"]
        output["attn_weights"][i] = output_i["attn_weights"][0]

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
            fc_embs = repeat_tensor(input_dict["fc_embs"][i], bdash).unsqueeze(1)
            output_i["fc_embs"] = fc_embs
        decoder_input["fc_embs"] = output_i["fc_embs"]

        ###############
        # prepare attn embeds
        ################
        if t == 0:
            attn_embs = repeat_tensor(input_dict["attn_embs"][i], bdash)
            attn_emb_lens = repeat_tensor(input_dict["attn_emb_lens"][i], bdash)
            output_i["attn_embs"] = attn_embs
            output_i["attn_emb_lens"] = attn_emb_lens
        decoder_input["attn_embs"] = output_i["attn_embs"]
        decoder_input["attn_emb_lens"] = output_i["attn_emb_lens"]

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
        # TODO attention weights


class ConditionalSeq2SeqAttnModel(Seq2SeqAttnModel):

    def __init__(self, encoder, decoder, **kwargs):
        if not hasattr(self, "compatible_decoders"):
            self.compatible_decoders = (
                captioning.models.decoder.ConditionalBahAttnDecoder,
                captioning.models.decoder.SpecificityBahAttnDecoder
            )
        super().__init__(encoder, decoder, **kwargs)
        self.train_forward_keys.append("conditions")
        self.inference_forward_keys.append("conditions")

    def prepare_decoder_input(self, input_dict, output):
        decoder_input = super().prepare_decoder_input(input_dict, output)
        decoder_input["conditions"] = input_dict["conditions"]
        return decoder_input

    def prepare_beamsearch_decoder_input(self, input_dict, output_i):
        decoder_input = super().prepare_beamsearch_decoder_input(input_dict, output_i)
        t = input_dict["t"]
        i = input_dict["sample_idx"]
        beam_size = input_dict["beam_size"]

        ###############
        # prepare conditions
        ################

        if t == 0:
            conditions = repeat_tensor(input_dict["conditions"][i], beam_size)
            output_i["conditions"] = conditions
        decoder_input["conditions"] = output_i["conditions"]

        return decoder_input


class StyleSeq2SeqAttnModel(StyleCaptionModel):

    def __init__(self, encoder, decoder, ref_encoder, **kwargs):
        if not hasattr(self, "compatible_decoders"):
            self.compatible_decoders = (
                captioning.models.decoder.StyleBahAttnDecoder
            )
        super().__init__(encoder, decoder, ref_encoder, **kwargs)


    def seq_forward(self, input_dict):
        # Bahdanau attention only supports step-by-step implementation, so we implement forward in 
        # step-by-step manner whether in training or evaluation
        return self.stepwise_forward(input_dict)

    def prepare_output(self, input_dict):
        output = super().prepare_output(input_dict)
        attn_weights = torch.empty(output["seqs"].size(0),
                                   input_dict["attn_embs"].size(1),
                                   output["seqs"].size(1))
        output["attn_weights"] = attn_weights
        return output

    def prepare_decoder_input(self, input_dict, output):
        decoder_input = {
            "fc_embs": input_dict["fc_embs"],
            "attn_embs": input_dict["attn_embs"],
            "attn_emb_lens": input_dict["attn_emb_lens"],
            "styles": input_dict["styles"]
        }
        t = input_dict["t"]
        ###############
        # determine input word
        ################
        if input_dict["mode"] == "train" and random.random() < input_dict["ss_ratio"]: # training, scheduled sampling
            word = input_dict["caps"][:, t]
        else:
            if t == 0:
                word = torch.tensor([self.start_idx,] * input_dict["fc_embs"].size(0)).long()
            else:
                word = output["seqs"][:, t-1]
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
        output["attn_weights"][:, :, t] = output_t["weights"]

    def prepare_beamsearch_output(self, input_dict):
        output = super().prepare_beamsearch_output(input_dict)
        beam_size = input_dict["beam_size"]
        max_length = input_dict["max_length"]
        output["attn_weights"] = torch.empty(beam_size,
                                             max(input_dict["attn_emb_lens"]),
                                             max_length)
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
            fc_embs = repeat_tensor(input_dict["fc_embs"][i], beam_size)
            output_i["fc_embs"] = fc_embs
        decoder_input["fc_embs"] = output_i["fc_embs"]

        ###############
        # prepare attn embeds
        ################
        if t == 0:
            attn_embs = repeat_tensor(input_dict["attn_embs"][i], beam_size)
            attn_emb_lens = repeat_tensor(input_dict["attn_emb_lens"][i], beam_size)
            output_i["attn_embs"] = attn_embs
            output_i["attn_emb_lens"] = attn_emb_lens
        decoder_input["attn_embs"] = output_i["attn_embs"]
        decoder_input["attn_emb_lens"] = output_i["attn_emb_lens"]

        ###############
        # prepare styles
        ################
        if t == 0:
            styles = repeat_tensor(input_dict["styles"][i], beam_size)
            output_i["styles"] = styles
        decoder_input["styles"] = output_i["styles"]

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
        output_i["attn_weights"][..., t] = output_t["weights"]
        output_i["attn_weights"] = output_i["attn_weights"][output_i["prev_words_beam"], ...]

    def beamsearch_process(self, output, output_i, input_dict):
        super().beamsearch_process(output, output_i, input_dict)
        i = input_dict["sample_idx"]
        output["attn_weights"][i] = output_i["attn_weights"][0]


class StructSeq2SeqAttnModel(Seq2SeqAttnModel):

    def __init__(self, encoder, decoder, **kwargs):
        if not hasattr(self, "compatible_decoders"):
            self.compatible_decoders = (
                captioning.models.decoder.StructBahAttnDecoder
            )
        super().__init__(encoder, decoder, **kwargs)
        self.train_forward_keys.append("structures")
        self.inference_forward_keys.append("structures")

    def prepare_decoder_input(self, input_dict, output):
        decoder_input = super().prepare_decoder_input(input_dict, output)
        decoder_input["structures"] = input_dict["structures"]
        return decoder_input

    def prepare_beamsearch_decoder_input(self, input_dict, output_i):
        decoder_input = super().prepare_beamsearch_decoder_input(input_dict, output_i)
        t = input_dict["t"]
        i = input_dict["sample_idx"]
        beam_size = input_dict["beam_size"]

        ###############
        # prepare conditions
        ################

        if t == 0:
            conditions = repeat_tensor(input_dict["structures"][i], beam_size)
            output_i["structures"] = conditions
        decoder_input["structures"] = output_i["structures"]

        return decoder_input
