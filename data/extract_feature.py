# -*- coding: utf-8 -*-
# @Author: richman
# @Date:   2018-03-29
# @Last Modified by:   xuenan xu
# @Last Modified time: 2021-07-02
import librosa
import kaldiio
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm as tqdm
import h5py
from pypeln import process as pr


def load_audio(specifier: str, mono=True):
    if specifier.endswith("|"):
        fd = kaldiio.utils.open_like_kaldi(specifier, "rb")
        mat = kaldiio.matio._load_mat(fd, None)
        fd.close()
        sr, y = mat
        y = y.copy() / 2 ** 15
    else:
        assert Path(specifier).exists(), specifier + " not exists!"
        y, sr = librosa.load(specifier, sr=None, mono=mono)
    return y, sr

def extractmfcc(y, fs=44100, **mfcc_params):
    eps = np.spacing(1)
    # Calculate Static Coefficients
    power_spectrogram = np.abs(librosa.stft(y + eps,
                                            center=True,
                                            n_fft=mfcc_params['n_fft'],
                                            win_length=mfcc_params[
                                                'win_length'],
                                            hop_length=mfcc_params[
                                                'hop_length'],
                                            ))**2
    mel_basis = librosa.filters.mel(sr=fs,
                                    n_fft=mfcc_params['n_fft'],
                                    n_mels=mfcc_params['n_mels'],
                                    fmin=mfcc_params['fmin'],
                                    fmax=mfcc_params['fmax'],
                                    htk=mfcc_params['htk'])
    mel_spectrum = np.dot(mel_basis, power_spectrogram)
    if mfcc_params['no_mfcc']:
        return np.log(mel_spectrum + eps)
    else:
        S = librosa.power_to_db(mel_spectrum)
        return librosa.feature.mfcc(S=S,
                                    n_mfcc=mfcc_params['n_mfcc'])

def extractlms(y, fs=44100, **lms_params):
    eps = np.spacing(1)
    mel_spectrum = np.log(librosa.feature.melspectrogram(
        y, 
        sr=fs, 
        n_fft=lms_params['n_fft'],
        n_mels=lms_params['n_mels'],
        hop_length=lms_params['hop_length'],
        win_length=lms_params['win_length'],
        fmin=lms_params['fmin'],
        fmax=lms_params['fmax'],
        htk=lms_params['htk']
    ) + eps)
    return mel_spectrum

def extractstft(y, fs=44100, ** params):
    """Extracts short time fourier transform with either energy or power

    Args:
        y (np.array float): input array, usually from librosa.load()
        fs (int, optional): Sampling Rate
        **params: Extra parameters

    Returns:
        numpy array: Log amplitude of either energy or power stft
    """
    eps = np.spacing(1)
    # Calculate Static Coefficients
    spectogram = np.abs(librosa.stft(y + eps, center=params['center'],
                                     n_fft=params['n_fft'], win_length=params[
                                         'win_length'],
                                     hop_length=params['hop_length'], ))
    if params['power']:
        spectogram = spectogram**2
    return librosa.logamplitude(spectogram)


def extractwavelet(y, fs, **params):
    import pywt
    frames = librosa.util.frame(
        y, frame_length=4096, hop_length=2048).transpose()
    features = []
    for frame in frames:
        res = []
        for _ in range(params['level']):
            frame, d = pywt.dwt(frame, params['type'])
            res.append(librosa.power_to_db(np.sum(d * d) / len(d)))
        res.append(librosa.power_to_db(np.sum(frame * frame) / len(frame)))
        features.append(res)
    return np.array(features)


def extractraw(y, fs, **params):
    return librosa.util.frame(y, frame_length=params['frame_length'], hop_length=params['hop_length']).transpose()



parser = argparse.ArgumentParser()
""" Arguments: wavfilelist, n_mfcc, n_fft, win_length, hop_length, htk, fmin, fmax """
parser.add_argument('wav_csv', type=str)
parser.add_argument('feat_h5', type=str)
parser.add_argument('feat_csv', type=str)
parser.add_argument('-nomono', default=False, action="store_true")
parser.add_argument('--process_num', type=int, default=4)
parser.add_argument('--blacklist', type=str, default=None)
subparsers = parser.add_subparsers(help="subcommand help")

stftparser = subparsers.add_parser('stft')
stftparser.add_argument('-n_fft', type=int, default=2048)
stftparser.add_argument('-win_length', type=int, default=2048)
stftparser.add_argument('-hop_length', type=int, default=1024)
stftparser.add_argument('-center', default=False, action="store_true")
stftparser.add_argument('-power', default=False, action="store_true")
stftparser.set_defaults(extractfeat=extractstft)

lmsparser = subparsers.add_parser('lms')
lmsparser.add_argument('-n_mels', type=int, default=128)
lmsparser.add_argument('-n_fft', type=int, default=2048)
lmsparser.add_argument('-win_length', type=int, default=2048)
lmsparser.add_argument('-hop_length', type=int, default=1024)
lmsparser.add_argument('-htk', default=False,
                        action="store_true", help="Uses htk formula for LMS est.")
lmsparser.add_argument('-fmin', type=int, default=0)
lmsparser.add_argument('-fmax', type=int, default=8000)
lmsparser.set_defaults(extractfeat=extractlms)

mfccparser = subparsers.add_parser('mfcc')
mfccparser.add_argument('-n_mfcc', type=int, default=20)
mfccparser.add_argument('-n_mels', type=int, default=128)
mfccparser.add_argument('-n_fft', type=int, default=2048)
mfccparser.add_argument('-win_length', type=int, default=2048)
mfccparser.add_argument('-hop_length', type=int, default=1024)
mfccparser.add_argument('-htk', default=True,
                        action="store_true", help="Uses htk formula for MFCC est.")
mfccparser.add_argument('-fmin', type=int, default=12)
mfccparser.add_argument('-fmax', type=int, default=8000)

rawparser = subparsers.add_parser('raw')
rawparser.add_argument('-hop_length', type=int, default=1024)
rawparser.add_argument('-frame_length', type=int, default=2048)
rawparser.set_defaults(extractfeat=extractraw)

waveletparser = subparsers.add_parser('wave')
waveletparser.add_argument('-level', default=10, type=int)
waveletparser.add_argument('-type', default='db4', type=str)
waveletparser.set_defaults(extractfeat=extractwavelet)

args = parser.parse_args()
if not hasattr(args, "feat_h5"):
    args.feat_h5 = Path(args.wav_csv).with_name("feat.h5")
if not hasattr(args, "feat_csv"):
    args.feat_csv = Path(args.wav_csv).with_name("feat.csv")
argsdict = vars(args).copy()

del argsdict["extractfeat"]

def pypeln_wrapper(extractfeat, **params):
    def extract(row):
        row = row[1]
        y, sr = load_audio(row["file_name"], mono=not args.nomono)
        # feature = extractfeat(row["file_name"], sr, **params)
        if y.ndim > 1:
            feat = np.array([extractfeat(i, sr, **params) for i in y])
        else:
            feat = extractfeat(y, sr, **params)
        return row["audio_id"], feat
    return extract

wav_df = pd.read_csv(args.wav_csv, sep="\t")
if args.blacklist is not None:
    blacklist_samples = []
    with open(args.blacklist, "r") as reader:
        for line in reader.readlines():
            blacklist_samples.append(line.strip())
    wav_df = wav_df[~wav_df["audio_id"].isin(blacklist_samples)]
feat_csv_data = []
with h5py.File(args.feat_h5, "w") as feat_store, tqdm(total=wav_df.shape[0]) as pbar:
    for audio_id, feat in pr.map(pypeln_wrapper(args.extractfeat, **argsdict),
                                 wav_df.iterrows(),
                                 workers=args.process_num,
                                 maxsize=4):
        # Transpose feat, nsamples to nsamples, feat
        feat = np.vstack(feat).transpose()
        feat_store[audio_id] = feat
        feat_csv_data.append({
            "audio_id": audio_id,
            "hdf5_path": str(Path(args.feat_h5).absolute())
        })
        pbar.update()

pd.DataFrame(feat_csv_data).to_csv(args.feat_csv, sep="\t", index=False)

