# -*- coding: utf-8 -*-

from functools import partial
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio import transforms
import torchvision
from torchlibrosa.augmentation import SpecAugmentation

from captioning.models.utils import mean_with_lens, max_with_lens
from captioning.utils.train_util import merge_load_state_dict


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        
        return x


class ConvBlock5x5(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock5x5, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(5, 5), stride=(1, 1),
                              padding=(2, 2), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_bn(self.bn1)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        
        return x


class Cnn6Encoder(nn.Module):

    def __init__(self, sample_rate=32000, freeze=False):
        super().__init__()

        sr_to_fmax = {
            32000: 14000,
            16000: 8000
        }
        # Logmel spectrogram extractor
        self.melspec_extractor = transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=32 * sample_rate // 1000,
            win_length=32 * sample_rate // 1000,
            hop_length=10 * sample_rate // 1000,
            f_min=50,
            f_max=sr_to_fmax[sample_rate],
            n_mels=64,
            norm="slaney",
            mel_scale="slaney"
        )
        self.hop_length = 10 * sample_rate // 1000
        self.db_transform = transforms.AmplitudeToDB()
        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64,
            time_stripes_num=2, freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock5x5(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock5x5(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock5x5(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock5x5(in_channels=256, out_channels=512)

        self.downsample_ratio = 16

        self.fc1 = nn.Linear(512, 512, bias=True)
        self.fc_emb_size = 512
        self.init_weight()
        self.freeze = freeze

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)

    def load_pretrained(self, pretrained):
        checkpoint = torch.load(pretrained, map_location="cpu")

        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            raise Exception("Unkown checkpoint format")

        loaded_keys = merge_load_state_dict(state_dict, self)
        if self.freeze:
            for name, param in self.named_parameters():
                if name in loaded_keys:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

    def forward(self, input_dict):
        waveform = input_dict["wav"]
        wave_length = input_dict["wav_len"]
        specaug = input_dict["specaug"]
        x = self.melspec_extractor(waveform)
        x = self.db_transform(x)    # (batch_size, mel_bins, time_steps)
        x = x.transpose(1, 2)
        x = x.unsqueeze(1)      # (batch_size, 1, time_steps, mel_bins)

        # SpecAugment
        if self.training and specaug:
            x = self.spec_augmenter(x)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = torch.mean(x, dim=3)
        attn_emb = x.transpose(1, 2)
        wave_length = torch.as_tensor(wave_length)
        feat_length = torch.div(wave_length, self.hop_length,
            rounding_mode="floor") + 1
        feat_length = torch.div(feat_length, self.downsample_ratio,
            rounding_mode="floor")
        x_max = max_with_lens(attn_emb, feat_length)
        x_mean = mean_with_lens(attn_emb, feat_length)
        x = x_max + x_mean
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        fc_emb = F.dropout(x, p=0.5, training=self.training)

        return {
            "attn_emb": attn_emb,
            "fc_emb": fc_emb,
            "attn_emb_len": feat_length
        }


class Cnn10Encoder(nn.Module):

    def __init__(self, sample_rate=32000, freeze=False):
        super().__init__()

        sr_to_fmax = {
            32000: 14000,
            16000: 8000
        }
        # Logmel spectrogram extractor
        self.melspec_extractor = transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=32 * sample_rate // 1000,
            win_length=32 * sample_rate // 1000,
            hop_length=10 * sample_rate // 1000,
            f_min=50,
            f_max=sr_to_fmax[sample_rate],
            n_mels=64,
            norm="slaney",
            mel_scale="slaney"
        )
        self.hop_length = 10 * sample_rate // 1000
        self.db_transform = transforms.AmplitudeToDB()
        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64,
            time_stripes_num=2, freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.downsample_ratio = 16

        self.fc1 = nn.Linear(512, 512, bias=True)
        self.fc_emb_size = 512
        self.init_weight()
        self.freeze = freeze

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)

    def load_pretrained(self, pretrained):
        checkpoint = torch.load(pretrained, map_location="cpu")

        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            raise Exception("Unkown checkpoint format")

        loaded_keys = merge_load_state_dict(state_dict, self)
        if self.freeze:
            for name, param in self.named_parameters():
                if name in loaded_keys:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

    def forward(self, input_dict):
        waveform = input_dict["wav"]
        wave_length = input_dict["wav_len"]
        specaug = input_dict["specaug"]
        x = self.melspec_extractor(waveform)
        x = self.db_transform(x)    # (batch_size, mel_bins, time_steps)
        x = x.transpose(1, 2)
        x = x.unsqueeze(1)      # (batch_size, 1, time_steps, mel_bins)

        # SpecAugment
        if self.training and specaug:
            x = self.spec_augmenter(x)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = torch.mean(x, dim=3)
        attn_emb = x.transpose(1, 2)
        wave_length = torch.as_tensor(wave_length)
        feat_length = torch.div(wave_length, self.hop_length,
            rounding_mode="floor") + 1
        feat_length = torch.div(feat_length, self.downsample_ratio,
            rounding_mode="floor")
        x_max = max_with_lens(attn_emb, feat_length)
        x_mean = mean_with_lens(attn_emb, feat_length)
        x = x_max + x_mean
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        fc_emb = F.dropout(x, p=0.5, training=self.training)

        return {
            "attn_emb": attn_emb,
            "fc_emb": fc_emb,
            "attn_emb_len": feat_length
        }


class Cnn14Encoder(nn.Module):
    def __init__(self, sample_rate=32000, freeze=False):
        super().__init__()
        sr_to_fmax = {
            32000: 14000,
            16000: 8000
        }
        # Logmel spectrogram extractor
        self.melspec_extractor = transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=32 * sample_rate // 1000,
            win_length=32 * sample_rate // 1000,
            hop_length=10 * sample_rate // 1000,
            f_min=50,
            f_max=sr_to_fmax[sample_rate],
            n_mels=64,
            norm="slaney",
            mel_scale="slaney"
        )
        self.hop_length = 10 * sample_rate // 1000
        self.db_transform = transforms.AmplitudeToDB()
        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64,
            time_stripes_num=2, freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.downsample_ratio = 32

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_emb_size = 2048
        
        self.init_weight()
        self.freeze = freeze

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)

    def load_pretrained(self, pretrained):
        checkpoint = torch.load(pretrained, map_location="cpu")

        if "model" in checkpoint:
            state_keys = checkpoint["model"].keys()
            backbone = False
            for key in state_keys:
                if key.startswith("backbone."):
                    backbone = True
                    break

            if backbone: # COLA
                state_dict = {}
                for key, value in checkpoint["model"].items():
                    if key.startswith("backbone."):
                        model_key = key.replace("backbone.", "")
                        state_dict[model_key] = value
            else: # PANNs
                state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint: # CLAP
            state_dict = checkpoint["state_dict"]
            state_dict_keys = list(filter(
                lambda x: "audio_encoder" in x, state_dict.keys()))
            state_dict = {
                key.replace('audio_encoder.', ''): state_dict[key]
                    for key in state_dict_keys
            }
        else:
            raise Exception("Unkown checkpoint format")

        loaded_keys = merge_load_state_dict(state_dict, self)
        if self.freeze:
            for name, param in self.named_parameters():
                if name in loaded_keys:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
 
    def forward(self, input_dict):
        waveform = input_dict["wav"]
        wave_length = input_dict["wav_len"]
        specaug = input_dict["specaug"]
        x = self.melspec_extractor(waveform)
        x = self.db_transform(x)    # (batch_size, mel_bins, time_steps)
        x = x.transpose(1, 2)
        x = x.unsqueeze(1)      # (batch_size, 1, time_steps, mel_bins)

        # SpecAugment
        if self.training and specaug:
            x = self.spec_augmenter(x)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        attn_emb = x.transpose(1, 2)
        
        wave_length = torch.as_tensor(wave_length)
        feat_length = torch.div(wave_length, self.hop_length,
            rounding_mode="floor") + 1
        feat_length = torch.div(feat_length, self.downsample_ratio,
            rounding_mode="floor")
        x_max = max_with_lens(attn_emb, feat_length)
        x_mean = mean_with_lens(attn_emb, feat_length)
        x = x_max + x_mean
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        fc_emb = F.dropout(x, p=0.5, training=self.training)
        
        output_dict = {
            'fc_emb': fc_emb,
            'attn_emb': attn_emb,
            'attn_emb_len': feat_length
        }

        return output_dict


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            _layers = [
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False), 
                nn.AvgPool2d(stride), 
                nn.BatchNorm2d(hidden_dim), 
                nn.ReLU6(inplace=True), 
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), 
                nn.BatchNorm2d(oup)
                ]
            _layers = nn.Sequential(*_layers)
            init_layer(_layers[0])
            init_bn(_layers[2])
            init_layer(_layers[4])
            init_bn(_layers[5])
            self.conv = _layers
        else:
            _layers = [
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False), 
                nn.BatchNorm2d(hidden_dim), 
                nn.ReLU6(inplace=True), 
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False), 
                nn.AvgPool2d(stride), 
                nn.BatchNorm2d(hidden_dim), 
                nn.ReLU6(inplace=True), 
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), 
                nn.BatchNorm2d(oup)
                ]
            _layers = nn.Sequential(*_layers)
            init_layer(_layers[0])
            init_bn(_layers[1])
            init_layer(_layers[3])
            init_bn(_layers[5])
            init_layer(_layers[7])
            init_bn(_layers[8])
            self.conv = _layers

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, sample_rate):
        
        super().__init__()

        sr_to_fmax = {
            32000: 14000,
            16000: 8000
        }
        # Logmel spectrogram extractor
        self.melspec_extractor = transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=32 * sample_rate // 1000,
            win_length=32 * sample_rate // 1000,
            hop_length=10 * sample_rate // 1000,
            f_min=50,
            f_max=sr_to_fmax[sample_rate],
            n_mels=64,
            norm="slaney",
            mel_scale="slaney"
        )
        self.hop_length = 10 * sample_rate // 1000
        self.db_transform = transforms.AmplitudeToDB()
        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64,
            time_stripes_num=2, freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)
 
        width_mult=1.
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 2],
            [6, 160, 3, 1],
            [6, 320, 1, 1],
        ]

        self.downsample_ratio = 32

        def conv_bn(inp, oup, stride):
            _layers = [
                nn.Conv2d(inp, oup, 3, 1, 1, bias=False), 
                nn.AvgPool2d(stride), 
                nn.BatchNorm2d(oup), 
                nn.ReLU6(inplace=True)
                ]
            _layers = nn.Sequential(*_layers)
            init_layer(_layers[0])
            init_bn(_layers[2])
            return _layers


        def conv_1x1_bn(inp, oup):
            _layers = nn.Sequential(
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True)
            )
            init_layer(_layers[0])
            init_bn(_layers[1])
            return _layers

        # building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(1, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        self.fc1 = nn.Linear(1280, 1024, bias=True)
        
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
 
    def forward(self, input_dict):

        waveform = input_dict["wav"]
        wave_length = input_dict["wav_len"]
        specaug = input_dict["specaug"]
        x = self.melspec_extractor(waveform)
        x = self.db_transform(x)    # (batch_size, mel_bins, time_steps)
        x = x.transpose(1, 2)
        x = x.unsqueeze(1)      # (batch_size, 1, time_steps, mel_bins)
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training and specaug:
            x = self.spec_augmenter(x)
        
        x = self.features(x)
        
        x = torch.mean(x, dim=3)
        attn_emb = x.transpose(1, 2)
        
        wave_length = torch.as_tensor(wave_length)
        feat_length = torch.div(wave_length, self.hop_length,
            rounding_mode="floor") + 1
        feat_length = torch.div(feat_length, self.downsample_ratio,
            rounding_mode="floor")
        x_max = max_with_lens(attn_emb, feat_length)
        x_mean = mean_with_lens(attn_emb, feat_length)
        x = x_max + x_mean
        # TODO: the original PANNs code does not have dropout here, why?
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        fc_emb = F.dropout(x, p=0.5, training=self.training)
        
        output_dict = {
            'fc_emb': fc_emb,
            'attn_emb': attn_emb,
            'attn_emb_len': feat_length
        }

        return output_dict


class MobileNetV3(nn.Module):
    
    def __init__(self,
                 sample_rate,
                 model_name,
                 n_mels=64,
                 win_length=32,
                 pretrained=True,
                 freeze=False,
                 pooling="mean_max_fc"):

        from captioning.models.eff_at_encoder import get_model, NAME_TO_WIDTH

        super().__init__()
        sr_to_fmax = {
            32000: 14000,
            16000: 8000
        }
        self.n_mels = n_mels
        # Logmel spectrogram extractor
        self.melspec_extractor = transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=32 * sample_rate // 1000,
            win_length=win_length * sample_rate // 1000,
            hop_length=10 * sample_rate // 1000,
            f_min=50,
            f_max=sr_to_fmax[sample_rate],
            n_mels=n_mels,
            norm="slaney",
            mel_scale="slaney"
        )
        self.hop_length = 10 * sample_rate // 1000
        self.db_transform = transforms.AmplitudeToDB()
        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64,
            time_stripes_num=2, freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(n_mels)
        
        width_mult = NAME_TO_WIDTH(model_name)
        self.features = get_model(model_name=model_name,
                                  pretrained=pretrained,
                                  width_mult=width_mult).features
        self.downsample_ratio = 32

        if pooling == "mean_max_fc":
            self.fc_emb_size = 512
            self.fc1 = nn.Linear(self.features[-1].out_channels, 512, bias=True)
        elif pooling == "mean":
            self.fc_emb_size = self.features[-1].out_channels
        self.init_weight()

        if freeze:
            for param in self.parameters():
                param.requires_grad = False

        self.pooling = pooling

    def init_weight(self):
        init_bn(self.bn0)
        if hasattr(self, "fc1"):
            init_layer(self.fc1)

    def forward(self, input_dict):

        waveform = input_dict["wav"]
        wave_length = input_dict["wav_len"]
        specaug = input_dict["specaug"]
        x = self.melspec_extractor(waveform)
        x = self.db_transform(x)    # (batch_size, mel_bins, time_steps)
        x = x.transpose(1, 2)
        x = x.unsqueeze(1)      # (batch_size, 1, time_steps, mel_bins)
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training and specaug:
            x = self.spec_augmenter(x)
        
        x = self.features(x)
        
        x = torch.mean(x, dim=3)
        attn_emb = x.transpose(1, 2)
        
        wave_length = torch.as_tensor(wave_length)
        feat_length = torch.div(wave_length, self.hop_length,
            rounding_mode="floor") + 1
        feat_length = torch.div(feat_length, self.downsample_ratio,
            rounding_mode="floor")

        if self.pooling == "mean_max_fc":
            x_max = max_with_lens(attn_emb, feat_length)
            x_mean = mean_with_lens(attn_emb, feat_length)
            x = x_max + x_mean
            x = F.dropout(x, p=0.5, training=self.training)
            x = F.relu_(self.fc1(x))
            fc_emb = F.dropout(x, p=0.5, training=self.training)
        elif self.pooling == "mean":
            fc_emb = mean_with_lens(attn_emb, feat_length)
        
        output_dict = {
            'fc_emb': fc_emb,
            'attn_emb': attn_emb,
            'attn_emb_len': feat_length
        }

        return output_dict


class EfficientNetB2(nn.Module):

    def __init__(self,
                 n_mels: int = 64,
                 win_length: int = 32,
                 hop_length: int = 10,
                 f_min: int = 0,
                 pretrained: bool = False,
                 prune_ratio: float = 0.0,
                 prune_se: bool = True,
                 prune_start_layer: int = 0,
                 prune_method: str = "operator_norm",
                 freeze: bool = False,):
        from captioning.models.eff_latent_encoder import get_model, get_pruned_model
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
        self.spec_augmenter = SpecAugmentation(time_drop_width=64,
            time_stripes_num=2, freq_drop_width=8, freq_stripes_num=2)
        if prune_ratio > 0:
            self.backbone = get_pruned_model(pretrained=pretrained,
                                             prune_ratio=prune_ratio,
                                             prune_start_layer=prune_start_layer,
                                             prune_se=prune_se,
                                             prune_method=prune_method)
        else:
            self.backbone = get_model(pretrained=pretrained)
        self.fc_emb_size = self.backbone.eff_net._conv_head.out_channels
        self.downsample_ratio = 32
        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input_dict):
        
        waveform = input_dict["wav"]
        wave_length = input_dict["wav_len"]
        specaug = input_dict["specaug"]
        x = self.melspec_extractor(waveform)
        x = self.db_transform(x)    # (batch_size, mel_bins, time_steps)
        
        x = rearrange(x, 'b f t -> b 1 t f')
        if self.training and specaug:
            x = self.spec_augmenter(x)
        x = rearrange(x, 'b 1 t f -> b f t')
        
        x = self.backbone(x)
        attn_emb = x

        wave_length = torch.as_tensor(wave_length)
        feat_length = torch.div(wave_length, self.hop_length,
            rounding_mode="floor") + 1
        feat_length = torch.div(feat_length, self.downsample_ratio,
            rounding_mode="floor")
        fc_emb = mean_with_lens(attn_emb, feat_length)
        
        output_dict = {
            'fc_emb': fc_emb,
            'attn_emb': attn_emb,
            'attn_emb_len': feat_length
        }
        return output_dict


if __name__ == "__main__":
    encoder = MobileNetV3(32000, "mn10_as")
    print(encoder)
    input_dict = {
        "wav": torch.randn(4, 320000), 
        "wav_len": torch.tensor([320000, 280000, 160000, 300000]),
        "specaug": True
    }
    output_dict = encoder(input_dict)
    print("attn embed: ", output_dict["attn_emb"].shape)
    print("fc embed: ", output_dict["fc_emb"].shape)
    print("attn embed length: ", output_dict["attn_emb_len"])
