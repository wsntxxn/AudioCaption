import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio import transforms
from torchlibrosa.augmentation import SpecAugmentation


def generate_length_mask(lens, max_length=None):
    # lens = torch.as_tensor(lens)
    batch_size = lens.size(0)
    if max_length is None:
        max_length = max(lens)
    idxs = torch.arange(max_length).repeat(batch_size).view(
        batch_size, max_length).to(lens.device)
    mask = (idxs < lens.view(-1, 1))
    return mask


def mean_with_lens(features, lens):
    """
    features: [batch_size, time_steps, ...] 
        (assume the second dimension represents length)
    lens: [batch_size,]
    """
    # lens = torch.as_tensor(lens)
    if max(lens) != features.size(1):
        max_length = features.size(1)
        mask = generate_length_mask(lens, max_length)
    else:
        mask = generate_length_mask(lens)
    mask = mask.to(features.device) # [N, T]

    while mask.ndim < features.ndim:
        mask = mask.unsqueeze(-1)
    feature_mean = features * mask
    feature_mean = feature_mean.sum(1)
    while lens.ndim < feature_mean.ndim:
        lens = lens.unsqueeze(1)
    feature_mean = feature_mean / lens.to(features.device)
    # feature_mean = features * mask.unsqueeze(-1)
    # feature_mean = feature_mean.sum(1) / lens.unsqueeze(1).to(features.device)
    return feature_mean


def max_with_lens(features, lens):
    """
    features: [batch_size, time_steps, ...] 
        (assume the second dimension represents length)
    lens: [batch_size,]
    """
    # lens = torch.as_tensor(lens)
    if max(lens) != features.size(1):
        max_length = features.size(1)
        mask = generate_length_mask(lens, max_length)
    else:
        mask = generate_length_mask(lens)
    mask = mask.to(features.device) # [batch_size, time_steps]

    feature_max = features.clone()
    feature_max[~mask] = float("-inf")
    feature_max, _ = feature_max.max(1)
    return feature_max


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


class Cnn14(nn.Module):
    def __init__(self, sample_rate=32000, pretrained=None):
        
        super(Cnn14, self).__init__()

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
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
            freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.downsample_ratio = 32

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        
        if pretrained:
            state_dict = torch.load(pretrained, map_location="cpu")
            model_dict = self.state_dict()
            pretrained_dict = {
                k: v for k, v in state_dict.items() if (k in model_dict) and (
                    model_dict[k].shape == v.shape)
            }
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict, strict=True)
        else:
            self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
 
    def forward(self, waveform, wave_length, specaug=False):
        """
        Input: (batch_size, n_samples)"""
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
        time_emb = x.transpose(1, 2)
        
        feat_length = torch.div(wave_length, self.hop_length,
            rounding_mode="floor") + 1
        feat_length = torch.div(feat_length, self.downsample_ratio,
            rounding_mode="floor")
        x_max = max_with_lens(time_emb, feat_length)
        x_mean = mean_with_lens(time_emb, feat_length)
        x = x_max + x_mean
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        clip_emb = F.dropout(x, p=0.5, training=self.training)
        
        output_dict = {
            'clip_emb': clip_emb,
            'time_emb': time_emb,
            'length': feat_length
        }

        return output_dict


class ConvPreWavBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvPreWavBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=3, stride=1,
                              padding=1, bias=False)
                              
        self.conv2 = nn.Conv1d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=3, stride=1, dilation=2, 
                              padding=2, bias=False)
                              
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

        
    def forward(self, input, pool_size):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, kernel_size=pool_size)
        
        return x


class Wavegram_Logmel_Cnn14(nn.Module):
    def __init__(self, sample_rate=32000, pretrained=None):
        
        super(Wavegram_Logmel_Cnn14, self).__init__()

        assert sample_rate == 32000

        # Logmel spectrogram extractor
        self.melspec_extractor = transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=32 * sample_rate // 1000,
            win_length=32 * sample_rate // 1000,
            hop_length=10 * sample_rate // 1000,
            f_min=50,
            f_max=14000,
            n_mels=64,
            norm="slaney",
            mel_scale="slaney"
        )
        self.hop_length = 10 * sample_rate // 1000
        self.db_transform = transforms.AmplitudeToDB()
        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
            freq_drop_width=8, freq_stripes_num=2)

        self.pre_conv0 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=11, stride=5, padding=5, bias=False)
        self.pre_bn0 = nn.BatchNorm1d(64)
        self.pre_block1 = ConvPreWavBlock(64, 64)
        self.pre_block2 = ConvPreWavBlock(64, 128)
        self.pre_block3 = ConvPreWavBlock(128, 128)
        self.pre_block4 = ConvBlock(in_channels=4, out_channels=64)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=128, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.downsample_ratio = 32

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        
        if pretrained:
            state_dict = torch.load(pretrained, map_location="cpu")["model"]
            model_dict = self.state_dict()
            pretrained_dict = {
                k: v for k, v in state_dict.items() if (k in model_dict) and (
                    model_dict[k].shape == v.shape)
            }
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict, strict=True)
        else:
            self.init_weight()

    def init_weight(self):
        init_layer(self.pre_conv0)
        init_bn(self.pre_bn0)
        init_bn(self.bn0)
        init_layer(self.fc1)
 
    def forward(self, waveform, wave_length, specaug=False):
        """
        Input: (batch_size, data_length)"""

        # Wavegram
        a1 = F.relu_(self.pre_bn0(self.pre_conv0(waveform[:, None, :])))
        a1 = self.pre_block1(a1, pool_size=4)
        a1 = self.pre_block2(a1, pool_size=4)
        a1 = self.pre_block3(a1, pool_size=4)
        a1 = a1.reshape((a1.shape[0], -1, 32, a1.shape[-1])).transpose(2, 3)
        a1 = self.pre_block4(a1, pool_size=(2, 1))

        # Log mel spectrogram
        x = self.melspec_extractor(waveform)
        x = self.db_transform(x)    # (batch_size, mel_bins, time_steps)
        x = x.transpose(1, 2)
        x = x.unsqueeze(1)      # (batch_size, 1, time_steps, mel_bins)

        if self.training and specaug:
            x = self.spec_augmenter(x)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')

        # Concatenate Wavegram and Log mel spectrogram along the channel dimension
        if x.size(2) > a1.size(2):
            x = x[:, :, :a1.size(2), :]
        elif a1.size(2) > x.size(2):
            a1 = a1[:, :, :x.size(2), :]
        x = torch.cat((x, a1), dim=1)

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
        time_emb = x.transpose(1, 2)
        
        feat_length = torch.div(wave_length, self.hop_length,
            rounding_mode="floor") + 1
        feat_length = torch.div(feat_length, self.downsample_ratio,
            rounding_mode="floor")
        x_max = max_with_lens(time_emb, feat_length)
        x_mean = mean_with_lens(time_emb, feat_length)
        x = x_max + x_mean
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        clip_emb = F.dropout(x, p=0.5, training=self.training)
        
        output_dict = {
            'clip_emb': clip_emb,
            'time_emb': time_emb,
            'length': feat_length
        }

        return output_dict


class ClapWrapper(nn.Module):

    def __init__(self,
                 audio_encoder,
                 audio_dim,
                 shared_dim,
                 pretrained=None):
        super().__init__()

        self.audio_encoder = audio_encoder
        self.audio_proj = nn.Linear(audio_dim, shared_dim)
        if pretrained:
            checkpoint = torch.load(pretrained, map_location="cpu")
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                raise Exception
            model_dict = self.state_dict()
            pretrained_dict = {
                k: v for k, v in state_dict.items() if (k in model_dict) and (
                    model_dict[k].shape == v.shape)
            }
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict, strict=True)
    
    def forward(self, waveform, wave_length, specaug=False):
        output_dict = self.audio_encoder(waveform, wave_length, specaug)
        audio_emb = output_dict["clip_emb"]
        audio_emb = self.audio_proj(audio_emb)
        norm = audio_emb.norm(p=2, dim=-1, keepdim=True)
        audio_emb = audio_emb.div(norm + 1e-7).clip(-1e3, 1e3)
        output_dict["clip_emb"] = audio_emb
        return output_dict
