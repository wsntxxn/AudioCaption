import random
import torch
import torch.nn as nn

from captioning.models.base_model import CaptionModel
from captioning.models.utils import repeat_tensor
import captioning.models.decoder

class FcModel(CaptionModel):

    def __init__(self, encoder: nn.Module, decoder: nn.Module, **kwargs):
        if not hasattr(self, "compatible_decoders"):
            self.compatible_decoders = (
                captioning.models.decoder.RnnFcDecoder,
            )
        super().__init__(encoder, decoder, **kwargs)

    def seq_forward(self, input_dict):
        caps = input_dict["caps"]
        cap_max_len = caps.size(1)
        fc_embs = input_dict["fc_embs"].unsqueeze(1).repeat(1, cap_max_len - 1, 1) # [N, cap_max_len-1, src_emb_dim]
        output = self.decoder(
            {
                "word": caps[:, :-1],
                "fc_embs": fc_embs
            }
        )
        return output

    def prepare_decoder_input(self, input_dict, output):
        decoder_input = { "fc_embs": input_dict["fc_embs"].unsqueeze(1) }
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

    def prepare_beamsearch_decoder_input(self, input_dict, output_i):
        decoder_input = {}
        t = input_dict["t"]
        i = input_dict["sample_idx"]
        beam_size = input_dict["beam_size"]
        ###############
        # prepare fc embeds
        ################
        # repeat only at the first timestep to save consumption
        if t == 0:
            fc_embs = repeat_tensor(input_dict["fc_embs"][i], beam_size).unsqueeze(1)
            output_i["fc_embs"] = fc_embs
        decoder_input["fc_embs"] = output_i["fc_embs"]

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
            decoder_input["state"] = output_i["state"][:, output_i["prev_words_beam"], :].contiguous()

        return decoder_input

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

    def beamsearch_process_step(self, output_i, output_t):
        output_i["state"] = output_t["state"]

    def dbs_process_step(self, output_i, output_t):
        divm = output_t["divm"]
        output_i["state"][divm] = output_t["state"]


class FcStartModel(FcModel):

    def __init__(self, encoder: nn.Module, decoder: nn.Module, **kwargs):
        if not hasattr(self, "compatible_decoders"):
            self.compatible_decoders = (
                captioning.models.decoder.RnnFcStartDecoder
            )
        super().__init__(encoder, decoder, **kwargs)

    def prepare_output(self, input_dict):
        output = {}
        batch_size = input_dict["fc_embs"].size(0)
        if input_dict["mode"] == "train":
            max_length = input_dict["caps"].size(1)
        elif input_dict["mode"] == "inference":
            max_length = input_dict["max_length"]
        else:
            raise Exception("mode should be either 'train' or 'inference'")
        device = input_dict["fc_embs"].device
        output["seqs"] = torch.full((batch_size, max_length), self.end_idx, dtype=torch.long)
        output["logits"] = torch.empty(batch_size, max_length, self.vocab_size).to(device)
        output["sampled_logprobs"] = torch.zeros(batch_size, max_length)
        output["embeds"] = torch.empty(batch_size, max_length, self.decoder.d_model).to(device)
        return output

    def seq_forward(self, input_dict):
        caps = input_dict["caps"]
        fc_embs = input_dict["fc_embs"]
        output = self.decoder(
            {
                "word": caps[:, :-1],
                "fc_embs": fc_embs
            }
        )
        return output

    def prepare_decoder_input(self, input_dict, output):
        decoder_input = {}
        t = input_dict["t"]

        if input_dict["mode"] == "train" and random.random() < input_dict["ss_ratio"]: # training, scheduled sampling
            if t == 0:
                word = None
            else:
                word = input_dict["caps"][:, t-1].unsqueeze(1)
        else:
            if t == 0:
                word = None
            else:
                word = output["seqs"][:, t-1].unsqueeze(1)
        decoder_input["word"] = word

        if t == 0:
            fc_embs = input_dict["fc_embs"]
        else:
            fc_embs = None
        decoder_input["fc_embs"] = fc_embs

        if t > 0:
            decoder_input["state"] = output["state"]
        return decoder_input

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
        else:
            fc_embs = None
        decoder_input["fc_embs"] = fc_embs

        ###############
        # determine input word
        ################
        if t == 0:
            word = None
        else:
            word = output_i["next_word"].unsqueeze(1)
        decoder_input["word"] = word

        ################
        # prepare rnn state
        ################
        if t > 0:
            decoder_input["state"] = output_i["state"][:, output_i["prev_words_beam"], :].contiguous()

        return decoder_input

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
        if local_time == 0:
            fc_embs = repeat_tensor(input_dict["fc_embs"][i], bdash)
        else:
            fc_embs = None
        decoder_input["fc_embs"] = fc_embs

        ###############
        # determine input word
        ################
        if local_time == 0:
            word = None
        else:
            word = output_i["next_word"][divm].unsqueeze(1)
        decoder_input["word"] = word

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
