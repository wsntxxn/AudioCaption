import math

import torch
import torch.nn.functional as F
from torchaudio import transforms
import torch.nn as nn
from einops import rearrange

# from captioning.models.transformer_encoder import Htsat
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch import utils as efficientnet_utils
from einops import rearrange, reduce


class _EffiNet(nn.Module):
    """A proxy for efficient net models"""
    def __init__(self,
                 blocks_args=None,
                 global_params=None,
                 ) -> None:
        super().__init__()
        self.eff_net = EfficientNet(blocks_args=blocks_args,
                                    global_params=global_params)
        

    def forward(self, x: torch.Tensor): 
        x = rearrange(x, 'b f t -> b 1 f t')
        x = self.eff_net.extract_features(x)
        return reduce(x, 'b c f t -> b t c', 'mean')


def get_model():
    blocks_args, global_params = efficientnet_utils.get_model_params(
        'efficientnet-b2', {'include_top': False})
    model = _EffiNet(blocks_args=blocks_args,
                     global_params=global_params)
    model.eff_net._change_in_channels(1)
    return model


class EfficientNetB2(nn.Module):

    def __init__(self,
                 n_mels: int = 64,
                 win_length: int = 32,
                 hop_length: int = 10,
                 f_min: int = 0,
                 pretrained: bool = False,
                 freeze: bool = False,):
        super().__init__()
        sample_rate = 16000
        self.melspec_extractor = transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=win_length * sample_rate // 1000,
            win_length=win_length * sample_rate // 1000,
            hop_length=hop_length * sample_rate // 1000,
            f_min=f_min,
            n_mels=n_mels,
        )
        self.hop_length = 10 * sample_rate // 1000
        self.db_transform = transforms.AmplitudeToDB(top_db=120)
        self.backbone = get_model()
        self.fc_emb_size = self.backbone.eff_net._conv_head.out_channels
        self.downsample_ratio = 32
        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    # def forward(self, input_dict):
    #
    #     waveform = input_dict["wav"]
    #     wave_length = input_dict["wav_len"]
    #     specaug = input_dict["specaug"]
    #     x = self.melspec_extractor(waveform)
    #     x = self.db_transform(x)    # (batch_size, mel_bins, time_steps)
    #
    #     x = rearrange(x, 'b f t -> b 1 t f')
    #     if self.training and specaug:
    #         x = self.spec_augmenter(x)
    #     x = rearrange(x, 'b 1 t f -> b f t')
    #
    #     x = self.backbone(x)
    #     attn_emb = x
    #
    #     wave_length = torch.as_tensor(wave_length)
    #     feat_length = torch.div(wave_length, self.hop_length,
    #         rounding_mode="floor") + 1
    #     feat_length = torch.div(feat_length, self.downsample_ratio,
    #         rounding_mode="floor")
    #     fc_emb = mean_with_lens(attn_emb, feat_length)
    #
    #     output_dict = {
    #         'fc_emb': fc_emb,
    #         'attn_emb': attn_emb,
    #         'attn_emb_len': feat_length
    #     }
    #     return output_dict

    def forward(self, waveform):
        
        x = self.melspec_extractor(waveform)
        x = self.db_transform(x)    # (batch_size, mel_bins, time_steps)
        x = self.backbone(x)
        attn_emb = x

        wave_length = torch.as_tensor([waveform.shape[1]] * waveform.shape[0])
        feat_length = torch.div(wave_length, self.hop_length,
            rounding_mode="floor") + 1
        feat_length = torch.div(feat_length, self.downsample_ratio,
            rounding_mode="floor")

        return attn_emb, feat_length


def generate_length_mask(lens, max_length=None):
    lens = torch.as_tensor(lens)
    N = lens.size(0)
    if max_length is None:
        max_length = max(lens)
        if isinstance(max_length, torch.Tensor):
            max_length = max_length.item()
    idxs = torch.arange(max_length).repeat(N).view(N, max_length)
    idxs = idxs.to(lens.device)
    mask = (idxs < lens.view(-1, 1))
    return mask


# class HtsatFlopsModel(Htsat):
#
#     def forward(self, waveform):  # out_feat_keys: List[str] = None):
#         # wave_length = input_dict["wav_len"]
#         x = self.melspec_extractor(waveform)
#         x = self.db_transform(x)    # (batch_size, mel_bins, time_steps)
#
#         x = x.transpose(1, 2)
#         x = x.unsqueeze(1)      # (batch_size, 1, time_steps, mel_bins)
#
#         x = x.transpose(1, 3)
#         x = self.bn0(x)
#         x = x.transpose(1, 3)
#
#         x = self.reshape_wav2img(x)
#         # (time_steps, mel_bins) -> (n * mel_bins, window_size), time_steps = n * window_size
#         output_dict = self.forward_features(x)
#         wave_length = torch.as_tensor([waveform.shape[1]] * waveform.shape[0])
#         # wave_length = torch.as_tensor(wave_length)
#         feat_length = torch.div(wave_length, self.hop_length,
#             rounding_mode="floor") + 1
#         emb_length = feat_length // (2 ** (len(self.depths) - 1)) // self.patch_stride[1]
#         output_dict["attn_emb_len"] = emb_length
#         return output_dict


# class EfficientNetB2FlopsModel(EfficientNetB2):
#
#     def forward(self, waveform):
#
#         # waveform = input_dict["wav"]
#         # wave_length = input_dict["wav_len"]
#         print(waveform.shape)
#         x = self.melspec_extractor(waveform)
#         x = self.db_transform(x)    # (batch_size, mel_bins, time_steps)
#         x = self.backbone(x)
#         attn_emb = x
#
#         # wave_length = torch.as_tensor(wave_length)
#         wave_length = torch.as_tensor([waveform.shape[1]] * waveform.shape[0])
#         feat_length = torch.div(wave_length, self.hop_length,
#             rounding_mode="floor") + 1
#         feat_length = torch.div(feat_length, self.downsample_ratio,
#             rounding_mode="floor")
#
#         # fc_emb = mean_with_lens(attn_emb, feat_length)
#         # fc_emb = attn_emb.mean(dim=1)
#
#         # output_dict = {
#         #     'fc_emb': fc_emb,
#         #     'attn_emb': attn_emb,
#         #     'attn_emb_len': feat_length
#         # }
#         # return output_dict
#
#         return attn_emb, feat_length


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * \
            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # self.register_buffer("pe", pe)
        self.register_parameter("pe", nn.Parameter(pe, requires_grad=False))

    def forward(self, x):
        # x: [T, N, E]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerDecoder(nn.Module):

    def __init__(self,
                 emb_dim,
                 vocab_size,
                 attn_emb_dim,
                 dropout):
        super().__init__()
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.attn_emb_dim = attn_emb_dim
        self.word_embedding = nn.Embedding(vocab_size, emb_dim)
        self.in_dropout = nn.Dropout(dropout)
        self.d_model = emb_dim
        self.nhead = self.d_model // 64
        self.nlayers = 2
        self.dim_feedforward = self.d_model * 4

        self.pos_encoder = PositionalEncoding(self.d_model, dropout)
        layer = nn.TransformerDecoderLayer(d_model=self.d_model,
                                           nhead=self.nhead,
                                           dim_feedforward=self.dim_feedforward,
                                           dropout=dropout)
        self.model = nn.TransformerDecoder(layer, self.nlayers)
        self.classifier = nn.Linear(self.d_model, vocab_size, bias=False)
        self.classifier.weight = self.word_embedding.weight
        self.attn_proj = nn.Sequential(
            nn.Linear(self.attn_emb_dim, self.d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(self.d_model)
        )
        self.init_params()

    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, max_length):
        mask = (torch.triu(torch.ones(max_length, max_length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, word, attn_emb, attn_emb_len, cap_padding_mask):
        p_attn_emb = self.attn_proj(attn_emb)
        p_attn_emb = p_attn_emb.transpose(0, 1) # [T_src, N, emb_dim]
        word = word.to(attn_emb.device)
        embed = self.in_dropout(self.word_embedding(word)) * math.sqrt(self.emb_dim) # [N, T, emb_dim]
        embed = embed.transpose(0, 1) # [T, N, emb_dim]
        embed = self.pos_encoder(embed)

        tgt_mask = self.generate_square_subsequent_mask(embed.size(0)).to(attn_emb.device)
        memory_key_padding_mask = ~generate_length_mask(attn_emb_len, attn_emb.size(1)).to(attn_emb.device)
        output = self.model(embed, p_attn_emb, tgt_mask=tgt_mask,
                            tgt_key_padding_mask=cap_padding_mask,
                            memory_key_padding_mask=memory_key_padding_mask)
        output = output.transpose(0, 1)
        output = self.classifier(output)
        return output


class EffTrm(nn.Module):

    def __init__(self, emb_dim, vocab_size, dropout):
        super().__init__()
        self.encoder = EfficientNetB2()
        self.decoder = TransformerDecoder(emb_dim=emb_dim,
                                          vocab_size=vocab_size,
                                          attn_emb_dim=1408,
                                          dropout=dropout)
        self.pad_idx = 0
        self.start_idx = 1

    def forward(self, waveform: torch.Tensor, cap_len: int = 20):
        attn_emb, attn_emb_len = self.encoder(waveform)
        batch_size = attn_emb.size(0)
        device = attn_emb.device
        seq = torch.tensor([self.start_idx,] * batch_size).unsqueeze(1).long()
        for t in range(cap_len):
            cap_padding_mask = (seq == self.pad_idx).to(device)
            logit_t = self.decoder(seq, attn_emb, attn_emb_len, cap_padding_mask)
            logit_t = logit_t[:, -1, :]
            word_t = logit_t.argmax(dim=-1)
            seq = torch.cat([seq, word_t.unsqueeze(1)], dim=1)

        return seq


if __name__ == "__main__":
    import argparse
    import torchaudio
    import captioning.datasets.text_tokenizer as text_tokenizer

    parser = argparse.ArgumentParser()

    parser.add_argument("--flops_counter", type=str,
                        choices=["thop", "ptflops", "calflops", "torch_flops"],
                        default="calflops")

    args = parser.parse_args()

    model = EffTrm(emb_dim=256,
                   vocab_size=4969,
                   dropout=0.2)
    ckpt = torch.load("./experiments/audiocaps/dict_tokenize/TransformerModel/effb2_2trm/kd_seq_enc_cntr_unsup_as/seed_4/swa.pth", "cpu")
    pretrained_dict = {}
    del ckpt["model"]["stdnt_proj.weight"]
    del ckpt["model"]["stdnt_proj.bias"]
    del ckpt["model"]["tchr_proj.weight"]
    del ckpt["model"]["tchr_proj.bias"]
    del ckpt["model"]["logit_scale"]
    for k, v in ckpt["model"].items():
        pretrained_dict[k[6:]] = v
    pretrained_dict["decoder.classifier.weight"] = pretrained_dict["decoder.word_embedding.weight"]
    pretrained_dict["decoder.pos_encoder.pe"] = model.decoder.pos_encoder.pe
    # model.load_state_dict(pretrained_dict, strict=True)
    model.eval()

    waveform, _ = torchaudio.load("/mnt/fast/nobackup/scratch4weeks/xx00336/workspace/audio_caption_xnx/data/audiocaps/audio_link/test/Y7fmOlUlwoNg.wav")
    waveform = torchaudio.functional.resample(waveform, orig_freq=32000, new_freq=16000)

    torch.onnx.export(model, torch.randn(1, 64, 1001), "EffB2_Trm.onnx")
    import pdb; pdb.set_trace()
    
    if args.flops_counter == "thop":
        from thop import profile
        macs, params = profile(model, inputs=(waveform,))
        print("MACs:%s   Params:%s \n" %(macs, params))
    elif args.flops_counter == "ptflops":
        from ptflops import get_model_complexity_info
        ###### ptflops
        macs, params = get_model_complexity_info(model, (1, 160000,), as_strings=True,
                                                 input_constructor=lambda x: {
                                                    "waveform": torch.randn(*x),
                                                    "cap_len": 20
                                                 },
                                                 verbose=True)

        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    elif args.flops_counter == "calflops":
        from calflops import calculate_flops
        ###### calflops
        flops, macs, params = calculate_flops(model=model,
                                              args=[waveform],
                                              output_precision=4)
        print("FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
    elif args.flops_counter == "torch_flops":
        from torch_flops import TorchFLOPsByFX
        with torch.no_grad():
            model(waveform)
        with torch.no_grad():
            flops_counter = TorchFLOPsByFX(model)
            flops_counter.propagate(waveform)
        # Print the flops of each node in the graph. Note that if there are unsupported operations, the "flops" of these ops will be marked as 'not recognized'.
        print('*' * 120)
        result_table = flops_counter.print_result_table()
        # # Print the total FLOPs
        total_flops = flops_counter.print_total_flops(show=True)
        total_time = flops_counter.print_total_time()
        max_memory = flops_counter.print_max_memory()

    # with torch.no_grad():
    #     output = model.generate_sampling(waveform, 20)
    #
    # tokenizer = text_tokenizer.DictTokenizer("data/audiocaps/train/vocab.pkl")
    # print(tokenizer.decode(output.numpy()))
