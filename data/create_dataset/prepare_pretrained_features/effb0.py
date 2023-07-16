# -*- coding: utf-8 -*-
# @Author: xuenan xu
# @Date:   2021-12-09
# @Last Modified by:   xuenan xu
# @Last Modified time: 2021-12-09
import math
import multiprocessing
import kaldiio
import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from pathlib import Path
from tqdm import trange
import h5py

import efficient_latent
from efficient_latent.models import upsample as _upsample
from efficientnet_pytorch import utils as efficientnet_utils
from einops import reduce


class _EffiNet(efficient_latent.models._EffiNet):

    def extract_embedding(self, x, upsample=False):
        x, num_input_frames = self.forward(x)
        if upsample:
            x = _upsample(x, ratio=32, target_length=num_input_frames)
        segment_embedding = x
        clip_embedding = reduce(segment_embedding, 'b t d -> b d', 'mean')
        return {"attn_feat": segment_embedding, "fc_feat": clip_embedding}


def EfficientNet_B2(**kwargs) -> _EffiNet:
    blocks_args, global_params = efficientnet_utils.get_model_params(
        'efficientnet-b2', {'include_top': False})
    model = _EffiNet(blocks_args=blocks_args,
                     global_params=global_params,
                     embed_dim=1_408,
                     **kwargs)
    model._change_in_channels(1)
    return model


def load_model(model_file_path: str = None, device: str = None) -> nn.Module:
    if device is None:
        torch_device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
    else:
        torch_device = torch.device(device)

    # Instantiate model
    model = EfficientNet_B2()
    model = model.to(torch_device)

    if model_file_path is None:
        # Download model
        state_dict = torch.hub.load_state_dict_from_url(
            'https://github.com/richermans/HEAR2021_EfficientLatent/releases/download/v0.0.1/effb2.pt',
            progress=True)
        model.load_state_dict(state_dict, strict=True)
    else:
        # Set model weights using checkpoint file
        checkpoint = torch.load(model_file_path, map_location=device)
        model.load_state_dict(checkpoint, strict=True)

    model.sample_rate = 16000  # Input sample rate
    model.scene_embedding_size = 1408
    model.timestamp_embedding_size = 1408
    return model


def load_audio(specifier: str):
    if specifier.endswith("|"):
        fd = kaldiio.utils.open_like_kaldi(specifier, "rb")
        mat = kaldiio.matio._load_mat(fd, None)
        fd.close()
        sr, y = mat
        y = y.copy() / 2 ** 15
    else:
        assert Path(specifier).exists(), specifier + " not exists!"
        y, sr = librosa.core.load(specifier, sr=None)
    y = librosa.core.resample(y, sr, 16000)
    return y


def multiprocess_load_audio(file_names, num_process=8, min_length=0.32):
    pool = multiprocessing.Pool(num_process)
    waveforms = list(pool.imap(load_audio, file_names))
    pool.close()
    pool.join()
    filtered_waveforms = []
    lengths = []
    for waveform in waveforms:
        if len(waveform) < min_length * 16000:
            continue
        filtered_waveforms.append(waveform)
        lengths.append(len(waveform))
    np_waveforms = np.zeros((len(filtered_waveforms), max(lengths)))
    for idx, waveform in enumerate(filtered_waveforms):
        np_waveforms[idx, :len(waveform)] = waveform
    return np_waveforms, np.array(lengths)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('wav_csv', type=str)
    parser.add_argument('--cuda', default=False, action='store_true')
    parser.add_argument('--num_process', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--fc_feat_h5', type=str)
    parser.add_argument('--fc_feat_csv', type=str)
    parser.add_argument('--attn_feat_h5', type=str)
    parser.add_argument('--attn_feat_csv', type=str)
    parser.add_argument('--blacklist', type=str, default=None)

    args = parser.parse_args()
    if not args.fc_feat_h5:
        args.fc_feat_h5 = "efficient_latent_fc.h5"
    args.fc_feat_h5 = Path(args.wav_csv).with_name(args.fc_feat_h5)
    if not args.fc_feat_csv:
        args.fc_feat_csv = "efficient_latent_fc.csv"
    args.fc_feat_csv = Path(args.wav_csv).with_name(args.fc_feat_csv)
    if not args.attn_feat_h5:
        args.attn_feat_h5 = "efficient_latent_attn.h5"
    args.attn_feat_h5 = Path(args.wav_csv).with_name(args.attn_feat_h5)
    if not args.attn_feat_csv:
        args.attn_feat_csv = "efficient_latent_attn.csv"
    args.attn_feat_csv = Path(args.wav_csv).with_name(args.attn_feat_csv)
    argsdict = vars(args)
    print(args)
    
    num_process = args.num_process
    batch_size = args.batch_size
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model = load_model(device=device)
    model.eval()

    wav_df = pd.read_csv(args.wav_csv, sep="\t")
    if args.blacklist is not None:
        blacklist_samples = []
        with open(args.blacklist, "r") as reader:
            for line in reader.readlines():
                blacklist_samples.append(line.strip())
        wav_df = wav_df[~wav_df["audio_id"].isin(blacklist_samples)]

    fc_feat_csv_data = []
    attn_feat_csv_data = []
    file_names = wav_df["file_name"].unique()
    fname_to_aid = dict(zip(wav_df["file_name"], wav_df["audio_id"]))

    with h5py.File(args.fc_feat_h5, "w") as fc_store, \
        h5py.File(args.attn_feat_h5, "w") as attn_store, \
        torch.no_grad():
        for i in trange(0, len(file_names), batch_size):
            file_batch = file_names[i: i + batch_size]
            waveform_batch, length_batch = multiprocess_load_audio(file_batch, num_process=num_process)
            waveform_batch = torch.as_tensor(waveform_batch).float().to(device)
            output_dict = model.extract_embedding(waveform_batch)
            fc_feat_batch = output_dict["fc_feat"].cpu().numpy()
            attn_feat_batch = output_dict["attn_feat"].cpu().numpy()
            for sample_idx in range(len(file_batch)):
                audio_id = fname_to_aid[file_batch[sample_idx]]
                length = math.floor((length_batch[sample_idx] / 160 + 1) / 32)
                fc_store[audio_id] = fc_feat_batch[sample_idx]
                attn_store[audio_id] = attn_feat_batch[sample_idx][:length]
                fc_feat_csv_data.append({
                    "audio_id": audio_id,
                    "hdf5_path": str(Path(args.fc_feat_h5).absolute())
                })
                attn_feat_csv_data.append({
                    "audio_id": audio_id,
                    "hdf5_path": str(Path(args.attn_feat_h5).absolute())
                })

    pd.DataFrame(fc_feat_csv_data).to_csv(args.fc_feat_csv, sep="\t", index=False)
    pd.DataFrame(attn_feat_csv_data).to_csv(args.attn_feat_csv, sep="\t", index=False)

