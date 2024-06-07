import random
import torch
import torch.nn as nn

from captioning.models.base import CaptionModel
from captioning.utils.model_util import repeat_tensor
import captioning.models.rnn_decoder


class FcModel(CaptionModel):

    def __init__(self, encoder: nn.Module, decoder: nn.Module, **kwargs):
        if not hasattr(self, "compatible_decoders"):
            self.compatible_decoders = (
                captioning.models.rnn_decoder.RnnFcDecoder
            )
        super().__init__(encoder, decoder, **kwargs)

    def seq_forward(self, input_dict):
        cap = input_dict["cap"]
        cap_max_len = cap.size(1)
        fc_emb = input_dict["fc_emb"].unsqueeze(1).repeat(
            1, cap_max_len - 1, 1) # [N, cap_max_len-1, src_emb_dim]
        output = self.decoder(
            {
                "word": cap[:, :-1],
                "fc_emb": fc_emb
            }
        )
        return output

    def prepare_decoder_input(self, input_dict, output):
        decoder_input = { "fc_emb": input_dict["fc_emb"].unsqueeze(1) }
        t = input_dict["t"]

        ###############
        # determine input word
        ################
        if input_dict["mode"] == "train" and \
                random.random() < input_dict["ss_ratio"]:
            # training, scheduled sampling
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
            fc_embs = repeat_tensor(input_dict["fc_emb"][i], beam_size).unsqueeze(1)
            output_i["fc_emb"] = fc_embs
        decoder_input["fc_emb"] = output_i["fc_emb"]

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
            fc_embs = repeat_tensor(input_dict["fc_emb"][i], bdash).unsqueeze(1)
            output_i["fc_emb"] = fc_embs
        decoder_input["fc_emb"] = output_i["fc_emb"]

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
