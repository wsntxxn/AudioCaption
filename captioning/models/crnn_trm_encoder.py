import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio import transforms

from captioning.models import embedding_pooling, BaseEncoder
from captioning.utils.model_util import init


class Block2D(nn.Module):

    def __init__(self, cin, cout, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(cin),
            nn.Conv2d(cin,
                      cout,
                      kernel_size=kernel_size,
                      padding=padding,
                      bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.1))

    def forward(self, x):
        return self.block(x)


class Cdur5Encoder(nn.Module):

    def __init__(self, sample_rate=16000, win_length=40, hop_length=20,
                 n_mels=64, pooling="mean"):
        super().__init__()
        sr_to_fmax = {
            32000: 14000,
            16000: 8000
        }
        # Logmel spectrogram extractor
        self.melspec_extractor = transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=win_length * sample_rate // 1000,
            win_length=win_length * sample_rate // 1000,
            hop_length=hop_length * sample_rate // 1000,
            f_min=50,
            f_max=sr_to_fmax[sample_rate],
            n_mels=n_mels,
            norm="slaney",
            mel_scale="slaney"
        )
        self.hop_length = hop_length * sample_rate // 1000
        self.db_transform = transforms.AmplitudeToDB()
        self.pooling = pooling
        self.features = nn.Sequential(
            Block2D(1, 32),
            nn.LPPool2d(4, (2, 4)),
            Block2D(32, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (2, 4)),
            Block2D(128, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (1, 4)),
            nn.Dropout(0.3),
        )
        self.downsample_ratio = 4
        with torch.no_grad():
            rnn_input_dim = self.features(
                torch.randn(1, 1, 100, n_mels)).shape
            rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]

        self.gru = nn.GRU(rnn_input_dim,
                          128,
                          bidirectional=True,
                          batch_first=True)
        self.apply(init)

    def forward(self, input_dict):
        waveform = input_dict["wav"]
        wave_length = input_dict["wav_len"]
        wave_length = torch.as_tensor(wave_length)
        x = self.melspec_extractor(waveform)
        x = self.db_transform(x)    # (batch_size, mel_bins, time_steps)
        x = x.transpose(1, 2)
        x = x.unsqueeze(1)      # (batch_size, 1, time_steps, mel_bins)

        x = self.features(x)
        x = x.transpose(1, 2).contiguous().flatten(-2)
        x, _ = self.gru(x)
        feat_length = torch.div(wave_length, self.hop_length,
            rounding_mode="floor") + 1
        feat_length = torch.div(feat_length, self.downsample_ratio,
            rounding_mode="floor")
        attn_emb = x
        fc_emb = embedding_pooling(x, feat_length, self.pooling)
        return {
            "attn_emb": attn_emb,
            "fc_emb": fc_emb,
            "attn_emb_len": feat_length
        }


class MMPool(nn.Module):

    def __init__(self, dims):
        super().__init__()
        self.avgpool = nn.AvgPool2d(dims)
        self.maxpool = nn.MaxPool2d(dims)

    def forward(self, x):
        return self.avgpool(x) + self.maxpool(x)


def conv_conv_block(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel,
                  out_channel,
                  kernel_size=3,
                  bias=False,
                  padding=1),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(True),
        nn.Conv2d(out_channel,
                  out_channel,
                  kernel_size=3,
                  bias=False,
                  padding=1),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(True)
    )


class Cdur8Encoder(BaseEncoder):
    
    def __init__(self, spec_dim, fc_feat_dim, attn_feat_dim, pooling="mean"):
        super().__init__(spec_dim, fc_feat_dim, attn_feat_dim)
        self.pooling = pooling
        self.features = nn.Sequential(
            conv_conv_block(1, 64),
            MMPool((2, 2)),
            nn.Dropout(0.2, True),
            conv_conv_block(64, 128),
            MMPool((2, 2)),
            nn.Dropout(0.2, True),
            conv_conv_block(128, 256),
            MMPool((1, 2)),
            nn.Dropout(0.2, True),
            conv_conv_block(256, 512),
            MMPool((1, 2)),
            nn.Dropout(0.2, True),
            nn.AdaptiveAvgPool2d((None, 1)),
        )
        self.init_bn = nn.BatchNorm2d(spec_dim)
        self.embedding = nn.Linear(512, 512)
        self.gru = nn.GRU(512, 256, bidirectional=True, batch_first=True)
        self.apply(init)

    def forward(self, input_dict):
        x = input_dict["spec"]
        lens = input_dict["spec_len"]
        lens = torch.as_tensor(copy.deepcopy(lens))
        x = x.unsqueeze(1)  # B x 1 x T x D
        x = x.transpose(1, 3)
        x = self.init_bn(x)
        x = x.transpose(1, 3)
        x = self.features(x)
        x = x.transpose(1, 2).contiguous().flatten(-2)
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.embedding(x))
        x, _ = self.gru(x)
        attn_emb = x
        lens //= 4
        fc_emb = embedding_pooling(x, lens, self.pooling)
        return {
            "attn_emb": attn_emb,
            "fc_emb": fc_emb,
            "attn_emb_len": lens
        }


class CrnnEncoder(nn.Module):

    def __init__(self,
                 cnn,
                 rnn,
                 freeze_cnn=False,
                 freeze_cnn_bn=False,
                 **kwargs):
        super().__init__()
        self.cnn = cnn
        self.rnn = rnn
        if freeze_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False
            self.freeze_cnn_bn = freeze_cnn_bn

    def train(self, mode):
        super().train(mode=mode)
        if self.freeze_cnn_bn:
            def bn_eval(module):
                class_name = module.__class__.__name__
                if class_name.find("BatchNorm") != -1:
                    module.eval()
            self.cnn.apply(bn_eval)
        return self

    def forward(self, input_dict):
        output_dict = self.cnn(input_dict)
        output_dict["attn"] = output_dict["attn_emb"]
        output_dict["attn_len"] = output_dict["attn_emb_len"]
        del output_dict["attn_emb"], output_dict["attn_emb_len"]
        output_dict = self.rnn(output_dict)
        return output_dict


class Cnn14TransformerEncoder(nn.Module):

    def __init__(self,
                 cnn,
                 transformer,
                 freeze_cnn=False,
                 freeze_cnn_bn=False,
                 **kwargs):
        super().__init__()
        self.cnn = cnn
        self.trm = transformer
        if freeze_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False
            self.freeze_cnn_bn = freeze_cnn_bn

    def train(self, mode):
        super().train(mode=mode)
        if self.freeze_cnn_bn:
            def bn_eval(module):
                class_name = module.__class__.__name__
                if class_name.find("BatchNorm") != -1:
                    module.eval()
            self.cnn.apply(bn_eval)
        return self

    def forward(self, input_dict):
        output_dict = self.cnn(input_dict)
        output_dict["attn"] = output_dict["attn_emb"]
        output_dict["attn_len"] = output_dict["attn_emb_len"]
        del output_dict["attn_emb"], output_dict["attn_emb_len"]
        output_dict = self.trm(output_dict)
        return output_dict
