import random
import torch

from captioning.models.base import CaptionModel
from captioning.models.style_model import StyleCaptionModel
from captioning.utils.model_util  import repeat_tensor
import captioning.models.rnn_decoder


class Seq2SeqAttnModel(CaptionModel):

    def __init__(self, encoder, decoder, **kwargs):
        if not hasattr(self, "compatible_decoders"):
            self.compatible_decoders = (
                captioning.models.rnn_decoder.BahAttnDecoder,
                captioning.models.rnn_decoder.BahAttnCatFcDecoder,
                captioning.models.rnn_decoder.BahAttnAddFcDecoder,
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


class ConditionalSeq2SeqAttnModel(Seq2SeqAttnModel):

    def __init__(self, encoder, decoder, **kwargs):
        if not hasattr(self, "compatible_decoders"):
            self.compatible_decoders = (
                captioning.models.rnn_decoder.ConditionalBahAttnDecoder,
                captioning.models.rnn_decoder.SpecificityBahAttnDecoder
            )
        super().__init__(encoder, decoder, **kwargs)
        self.train_forward_keys.append("condition")
        self.inference_forward_keys.append("condition")

    def prepare_decoder_input(self, input_dict, output):
        decoder_input = super().prepare_decoder_input(input_dict, output)
        decoder_input["condition"] = input_dict["condition"]
        return decoder_input

    def prepare_beamsearch_decoder_input(self, input_dict, output_i):
        decoder_input = super().prepare_beamsearch_decoder_input(input_dict, output_i)
        t = input_dict["t"]
        i = input_dict["sample_idx"]
        beam_size = input_dict["beam_size"]

        ###############
        # prepare condition
        ################

        if t == 0:
            condition = repeat_tensor(input_dict["condition"][i], beam_size)
            output_i["condition"] = condition
        decoder_input["condition"] = output_i["condition"]

        return decoder_input


class StyleSeq2SeqAttnModel(StyleCaptionModel, Seq2SeqAttnModel):
    # MRO: (StyleSeq2SeqAttnModel, StyleCaptionModel, Seq2SeqAttnModel,
    #       CaptionModel, nn.Module, object)

    def __init__(self, encoder, decoder, ref_encoder, **kwargs):
        if not hasattr(self, "compatible_decoders"):
            self.compatible_decoders = (
                captioning.models.decoder.StyleBahAttnDecoder,
            )
        super().__init__(encoder, decoder, ref_encoder, **kwargs)

    def prepare_decoder_input(self, input_dict, output):
        decoder_input = super().prepare_decoder_input(input_dict, output)
        decoder_input["style"] = input_dict["style"]
        return decoder_input

    def prepare_beamsearch_decoder_input(self, input_dict, output_i):
        decoder_input = super().prepare_beamsearch_decoder_input(
            input_dict, output_i)
        t = input_dict["t"]
        i = input_dict["sample_idx"]
        beam_size = input_dict["beam_size"]
        ###############
        # prepare style
        ################
        if t == 0:
            style = repeat_tensor(input_dict["style"][i], beam_size)
            output_i["style"] = style
        decoder_input["style"] = output_i["style"]
        return decoder_input


class StructSeq2SeqAttnModel(Seq2SeqAttnModel):

    def __init__(self, encoder, decoder, **kwargs):
        if not hasattr(self, "compatible_decoders"):
            self.compatible_decoders = (
                captioning.models.decoder.StructBahAttnDecoder,
            )
        super().__init__(encoder, decoder, **kwargs)
        self.train_forward_keys.append("structure")
        self.inference_forward_keys.append("structure")

    def prepare_decoder_input(self, input_dict, output):
        decoder_input = super().prepare_decoder_input(input_dict, output)
        decoder_input["structure"] = input_dict["structure"]
        return decoder_input

    def prepare_beamsearch_decoder_input(self, input_dict, output_i):
        decoder_input = super().prepare_beamsearch_decoder_input(input_dict, output_i)
        t = input_dict["t"]
        i = input_dict["sample_idx"]
        beam_size = input_dict["beam_size"]

        ###############
        # prepare condition
        ################

        if t == 0:
            structure = repeat_tensor(input_dict["structure"][i], beam_size)
            output_i["structure"] = structure
        decoder_input["structure"] = output_i["structure"]

        return decoder_input
