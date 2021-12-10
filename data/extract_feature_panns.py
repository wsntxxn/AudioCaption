# -*- coding: utf-8 -*-
# @Author: xuenan xu
# @Date:   2021-06-14
# @Last Modified by:   xuenan xu
# @Last Modified time: 2021-07-02
import multiprocessing
import argparse
from pathlib import Path
import math

import kaldiio
import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import trange
import h5py

import captioning.models.panns_inference_models as panns_inference_models


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
    for waveform in waveforms:
        if len(waveform) < min_length * 32000:
            continue
        filtered_waveforms.append(waveform)
        lengths.append(len(waveform))
    np_waveforms = np.zeros((len(filtered_waveforms), max(lengths)))
    for idx, waveform in enumerate(filtered_waveforms):
        np_waveforms[idx, :len(waveform)] = waveform
    return np_waveforms, np.array(lengths)


name_to_dr = {
    "Cnn10": 16,
    "Cnn14": 32,
    "Wavegram_Logmel_Cnn14": 32
}


if __name__ == "__main__":

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
    parser.add_argument('--model_type', type=str, default='Cnn14')
    parser.add_argument('--fc_feat_h5', type=str)
    parser.add_argument('--fc_feat_csv', type=str)
    parser.add_argument('--attn_feat_h5', type=str)
    parser.add_argument('--attn_feat_csv', type=str)
    parser.add_argument('--blacklist', type=str, default=None)
    parser.add_argument('--num_process', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)

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
    print(args)

    num_process = args.num_process
    batch_size = args.batch_size

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model = getattr(panns_inference_models, args.model_type)(
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

    downsample_ratio = name_to_dr[args.model_type]


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
            output_dict = model(waveform_batch)
            fc_feat_batch = output_dict["fc_feat"].cpu().numpy()
            attn_feat_batch = output_dict["attn_feat"].cpu().numpy()
            for sample_idx in range(len(file_batch)):
                audio_id = fname_to_aid[file_batch[sample_idx]]
                length = math.ceil((length_batch[sample_idx] / args.hop_size + 1) / downsample_ratio)
                fc_store[audio_id] = fc_feat_batch[sample_idx]
                attn_store[audio_id] = attn_feat_batch[sample_idx]
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

