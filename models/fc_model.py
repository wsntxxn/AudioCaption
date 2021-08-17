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

    def beamsearch_process_step(self, output_i, output_t):
        output_i["state"] = output_t["state"]
