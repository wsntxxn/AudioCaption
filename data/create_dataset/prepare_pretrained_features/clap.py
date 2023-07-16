# -*- coding: utf-8 -*-
# @Author: xuenan xu
# @Date:   2022-03-01
# @Last Modified by:   xuenan xu
# @Last Modified time: 2021-03-01
import multiprocessing
import argparse
from pathlib import Path
import math
import sys

import kaldiio
import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import trange
import h5py
from zsvision.zs_utils import load_json_config

import captioning.models.clap_audio_encoder as clap_models

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
    y = librosa.core.resample(y, sr, 32000)
    return y


def multiprocess_load_audio(file_names, num_process=8, min_length=0.32):
    pool = multiprocessing.Pool(num_process)
    waveforms = list(pool.imap(load_audio, file_names))
    pool.close()
    pool.join()
    filtered_waveforms = []
    lengths = []
    filtered_fnames = []
    for waveform, fname in zip(waveforms, file_names):
        if len(waveform) < min_length * 32000:
            continue
        filtered_waveforms.append(waveform)
        lengths.append(len(waveform))
        filtered_fnames.append(fname)
    np_waveforms = np.zeros((len(filtered_waveforms), max(lengths)))
    for idx, waveform in enumerate(filtered_waveforms):
        np_waveforms[idx, :len(waveform)] = waveform
    return filtered_fnames, np_waveforms, np.array(lengths)


def get_model(config, with_proj):
    model = getattr(clap_models, config["audio_encoder"]["type"])(
        **config["audio_encoder"]["args"])
    if with_proj:
        audio_encoder = model
        model = clap_models.ClapWrapper(
            audio_encoder=audio_encoder,
            **config["model"]["args"])
    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('wav_csv', type=str)
    parser.add_argument('experiment_path', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('--cuda', default=False, action='store_true')
    parser.add_argument('--with_proj', default=False, action='store_true')
    parser.add_argument('--fc_feat_h5', type=str)
    parser.add_argument('--fc_feat_csv', type=str)
    parser.add_argument('--attn_feat_h5', type=str)
    parser.add_argument('--attn_feat_csv', type=str)
    parser.add_argument('--blacklist', type=str, default=None)
    parser.add_argument('--num_process', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)

    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.fc_feat_h5:
        args.fc_feat_h5 = "clap_fc.h5"
    args.fc_feat_h5 = output_dir / args.fc_feat_h5
    if not args.fc_feat_csv:
        args.fc_feat_csv = "clap_fc.csv"
    args.fc_feat_csv = output_dir / args.fc_feat_csv
    if not args.attn_feat_h5:
        args.attn_feat_h5 = "clap_attn.h5"
    args.attn_feat_h5 = output_dir / args.attn_feat_h5
    if not args.attn_feat_csv:
        args.attn_feat_csv = "clap_attn.csv"
    args.attn_feat_csv = output_dir / args.attn_feat_csv
    argsdict = vars(args)
    print(args)

    experiment_path = Path(args.experiment_path)
    num_process = args.num_process
    batch_size = args.batch_size

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    config = load_json_config(experiment_path / "models" / "config.json")
    model = get_model(config["model"], args.with_proj)
    checkpoint = torch.load(experiment_path / "models" / "trained_model.pth",
                            map_location="cpu")

    state_dict = {}
    if args.with_proj:
        state_dict = checkpoint["state_dict"]
    else:
        for k, v in checkpoint["state_dict"].items():
            if k.startswith("audio_encoder."):
                state_dict[k.replace("audio_encoder.", "")] = v
    model.load_state_dict(state_dict, strict=True)

    model = model.to(device)
    model.eval()

    downsample_ratio = model.downsample_ratio


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
            fname_batch, waveform_batch, length_batch = multiprocess_load_audio(
                file_batch, num_process=num_process)
            waveform_batch = torch.as_tensor(waveform_batch).float().to(device)
            length_batch = torch.as_tensor(length_batch).long().to(device)
            output_dict = model(waveform_batch, length_batch)
            fc_feat_batch = output_dict["clip_emb"].cpu().numpy()
            attn_feat_batch = output_dict["time_emb"].cpu().numpy()
            attn_feat_length = output_dict["length"].cpu().numpy()
            for sample_idx in range(waveform_batch.shape[0]):
                audio_id = fname_to_aid[fname_batch[sample_idx]]
                # length = math.floor((length_batch[sample_idx] / args.hop_size + 1) / downsample_ratio)
                length = attn_feat_length[sample_idx]
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

