#!/usr/bin/env python3
import argparse
import logging
import os
from pathlib import Path
import math
import sys
from multiprocessing import Pool

import librosa
from tqdm import tqdm
import pandas as pd
import numpy as np
import soundfile as sf
from pypeln import process as pr
import h5py


parser = argparse.ArgumentParser()
parser.add_argument("input_csv")
parser.add_argument("-o", "--output", type=str, required=True)
parser.add_argument("-c", type=int, default=4)
parser.add_argument("--part_size", type=int, default=50000)
parser.add_argument("--sample_rate", type=int, default=None)

args = parser.parse_args()

print(args)


df = pd.read_csv(args.input_csv, sep="\t")

num_files = df.shape[0]


def load_audio(row):
    """extract_feature
    Extracts a log mel spectrogram feature from a filename:

    :param row: dataframe row containing "audio_id" and "file_name"
    """
    idx, item = row
    audio_id = item["audio_id"]
    fname = item["file_name"]
    try:
        y, sr = sf.read(fname, dtype="float32")
        if y.ndim > 1:
            y = y.mean(1)
        if args.sample_rate is not None:
            y = librosa.core.resample(y, sr, args.sample_rate)
        waveform = y.astype(np.float16)
        return audio_id, waveform
    except Exception as e:
        # Exception usually happens because some data has 6 channels, which librosa cant handle
        logging.error(e)
        logging.error(fname)
        return None, None

if num_files <= args.part_size: # single hdf5 file
    csv_data = []
    output_csv = Path(args.output).with_suffix(".csv")
    with h5py.File(args.output, "w") as store:
        for audio_id, waveform in tqdm(pr.map(load_audio,
                                             df.iterrows(),
                                             workers=args.c,
                                             maxsize=4),
                                      total=df.shape[0]):
            if audio_id is not None and waveform is not None:
                store[audio_id] = waveform
                csv_data.append({
                    "audio_id": audio_id,
                    "hdf5_path": str(Path(args.output).absolute())})
    pd.DataFrame(csv_data).to_csv(output_csv, sep="\t", index=False)

else: # multiple hdf5 files
    output = Path(args.output)
    if not output.exists():
        output.mkdir()
        (output / "hdf5").mkdir()
        (output / "csv").mkdir()
    def load_wrapper(index):
        start = (index - 1) * args.part_size
        end = index * args.part_size
        part_df = df[start: end]
        csv_data = []
        output_csv = output / f"csv/{index}.csv"
        with h5py.File(output / f"hdf5/{index}.h5", "w") as store, \
            tqdm(total=part_df.shape[0]) as pbar:
            for row in part_df.iterrows():
                audio_id, waveform = load_audio(row)
                if audio_id is not None and waveform is not None:
                    store[audio_id] = waveform
                    csv_data.append({
                        "audio_id": audio_id,
                        "hdf5_path": str((output / f"hdf5/{index}.h5").absolute())})
                pbar.update()
        pd.DataFrame(csv_data).to_csv(output_csv, sep="\t", index=False)

    pool = Pool(args.c)
    pool.map(load_wrapper, range(1, 1 + math.ceil(num_files / args.part_size)))
    pool.close()
    pool.join()
    os.system(f"concat_csv.py {output / 'csv/*.csv'} {output / 'waveform.csv'}")

