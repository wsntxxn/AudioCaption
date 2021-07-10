# -*- coding: utf-8 -*-
# @Author: xuenan xu
# @Date:   2021-06-14
# @Last Modified by:   xuenan xu
# @Last Modified time: 2021-07-02
import sys
import kaldiio
import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from pathlib import Path
from tqdm import tqdm as tqdm
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
import h5py
from pypeln import process as pr


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

        # self.init_weight()
        
    # def init_weight(self):
        # init_layer(self.conv1)
        # init_layer(self.conv2)
        # init_bn(self.bn1)
        # init_bn(self.bn2)

        
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

class Cnn10(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num):
        
        super(Cnn10, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # # Spec augmenter
        # self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            # freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(512, 512, bias=True)
        self.fc_audioset = nn.Linear(512, classes_num, bias=True)
        
        # self.init_weight()

    # def init_weight(self):
        # init_bn(self.bn0)
        # init_layer(self.fc1)
        # init_layer(self.fc_audioset)
 
    # def forward(self, input, mixup_lambda=None):
    def forward(self, input):
        """
        Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        # if self.training:
            # x = self.spec_augmenter(x)

        # # Mixup on spectrogram
        # if self.training and mixup_lambda is not None:
            # x = do_mixup(x, mixup_lambda)
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        attn_feats = x.transpose(1, 2)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))
        
        output_dict = {
            'clipwise_output': clipwise_output,
            'fc_feat': embedding,
            'attn_feat': attn_feats
        }

        return output_dict


class Cnn14(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num):
        
        super(Cnn14, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # # Spec augmenter
        # self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            # freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)
        
        # self.init_weight()

    # def init_weight(self):
        # init_bn(self.bn0)
        # init_layer(self.fc1)
        # init_layer(self.fc_audioset)
 
    def forward(self, input):
        """
        Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        # if self.training:
            # x = self.spec_augmenter(x)

        # # Mixup on spectrogram
        # if self.training and mixup_lambda is not None:
            # x = do_mixup(x, mixup_lambda)

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
        attn_feats = x.transpose(1, 2)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))
        
        output_dict = {
            'clipwise_output': clipwise_output, 
            'fc_feat': embedding,
            'attn_feat': attn_feats 
        }

        return output_dict

def load_audio(specifier: str, sr=None):
    if specifier.endswith("|"):
        fd = kaldiio.utils.open_like_kaldi(specifier, "rb")
        mat = kaldiio.matio._load_mat(fd, None)
        fd.close()
        sr, y = mat
        y = y.copy() / 2 ** 15
    else:
        assert Path(specifier).exists(), specifier + " not exists!"
        y, sr = librosa.load(specifier, sr=sr)
    return y, sr

parser = argparse.ArgumentParser()
parser.add_argument('wav_csv', type=str)
parser.add_argument('pretrained_model', type=str)
parser.add_argument('-sample_rate', type=int, default=32000)
parser.add_argument('-window_size', type=int, default=1024)
parser.add_argument('-hop_size', type=int, default=320)
parser.add_argument('-mel_bins', type=int, default=64)
parser.add_argument('-fmin', type=int, default=50)
parser.add_argument('-fmax', type=int, default=14000)
parser.add_argument('--cuda', default=False, action='store_true')
parser.add_argument('--model_type', type=str, default='Cnn10')
parser.add_argument('--process_num', type=int, default=4)
parser.add_argument('--fc_feat_h5', type=str)
parser.add_argument('--fc_feat_csv', type=str)
parser.add_argument('--attn_feat_h5', type=str)
parser.add_argument('--attn_feat_csv', type=str)

args = parser.parse_args()
if not args.fc_feat_h5:
    args.fc_feat_h5 = "panns_fc.h5"
args.fc_feat_h5 = Path(args.wav_csv).with_name(args.fc_feat_h5)
if not args.fc_feat_csv:
    args.fc_feat_csv = "panns_fc.csv"
args.fc_feat_csv = Path(args.wav_csv).with_name(args.fc_feat_csv)
if not args.attn_feat_h5:
    args.attn_feat_h5 = "panns_attn.h5"
args.attn_feat_h5 = Path(args.wav_csv).with_name(args.attn_feat_h5)
if not args.attn_feat_csv:
    args.attn_feat_csv = "panns_attn.csv"
args.attn_feat_csv = Path(args.wav_csv).with_name(args.attn_feat_csv)
argsdict = vars(args)

device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
device = torch.device(device)
model = eval(args.model_type)(
    sample_rate=args.sample_rate,
    window_size=args.window_size,
    hop_size=args.hop_size,
    mel_bins=args.mel_bins,
    fmin=args.fmin,
    fmax=args.fmax,
    classes_num=527)
checkpoint = torch.load(args.pretrained_model, map_location='cpu')
model.load_state_dict(checkpoint['model'])
model = model.to(device)
model.eval()

def extract_feature(row):
    row = row[1]
    waveform, _ = load_audio(row["file_name"], sr=args.sample_rate)
    waveform = waveform[None, :]
    waveform = torch.as_tensor(waveform).float().to(device)
    output_dict = model(waveform)
    fc_feat = output_dict["fc_feat"].cpu().numpy()[0]
    attn_feat = output_dict["attn_feat"].cpu().numpy()[0]
    return row["audio_id"], fc_feat, attn_feat

wav_df = pd.read_csv(args.wav_csv, sep="\t")
fc_feat_csv_data = []
attn_feat_csv_data = []

with h5py.File(args.fc_feat_h5, "w") as fc_store, \
    h5py.File(args.attn_feat_h5, "w") as attn_store, \
    tqdm(total=wav_df.shape[0]) as pbar, \
    torch.no_grad():
    # for audio_id, fc_feat, attn_feat in pr.map(extract_feature,
                                               # wav_df.iterrows(),
                                               # workers=args.process_num,
                                               # maxsize=4):
    for row in wav_df.iterrows():
        audio_id, fc_feat, attn_feat = extract_feature(row)
        fc_store[audio_id] = fc_feat
        attn_store[audio_id] = attn_feat
        fc_feat_csv_data.append({
            "audio_id": audio_id,
            "hdf5_path": str(Path(args.fc_feat_h5).absolute())
        })
        attn_feat_csv_data.append({
            "audio_id": audio_id,
            "hdf5_path": str(Path(args.attn_feat_h5).absolute())
        })
        pbar.update()

pd.DataFrame(fc_feat_csv_data).to_csv(args.fc_feat_csv, sep="\t", index=False)
pd.DataFrame(attn_feat_csv_data).to_csv(args.attn_feat_csv, sep="\t", index=False)

