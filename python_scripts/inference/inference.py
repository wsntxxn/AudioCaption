import sys
import os
from pathlib import Path
import json

import fire
from tqdm import tqdm, trange
import librosa
import torchaudio
import numpy as np
import pandas as pd
import torch
import h5py

import captioning.utils.train_util as train_util


def load_model(cfg, ckpt_path, device):
    model = train_util.init_model_from_config(cfg["model"])
    ckpt = torch.load(ckpt_path, "cpu")
    train_util.load_pretrained_model(model, ckpt)
    model.eval()
    model = model.to(device)
    tokenizer = train_util.init_obj_from_dict(cfg["data"]["train"][
        "collate_fn"]["tokenizer"])
    if not tokenizer.loaded:
        tokenizer.load_state_dict(ckpt["tokenizer"])
    model.set_index(tokenizer.bos, tokenizer.eos, tokenizer.pad)
    return model, tokenizer


def load_audio(specifier: str, target_sr: int):
    assert Path(specifier).exists(), specifier + " not exists!"
    y, sr = librosa.core.load(specifier, sr=None)
    if y.shape[0] == 0:
        return None
    y = torch.as_tensor(y)
    y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=target_sr)
    return y


def read_from_h5(key, key_to_h5, cache):
    hdf5_path = key_to_h5[key]
    if hdf5_path not in cache:
        cache[hdf5_path] = h5py.File(hdf5_path, "r")
    return cache[hdf5_path][key][()]


class InferenceDataset(torch.utils.data.Dataset):

    def __init__(self, aid_to_fname, original_sr, target_sr):
        super().__init__()
        self.aid_to_fname = aid_to_fname
        self.aids = list(aid_to_fname.keys())
        first_fname = self.aid_to_fname[self.aids[0]]
        if first_fname.endswith(".h5") or first_fname.endswith(".hdf5"):
            self.source = "hdf5"
            assert original_sr is not None, "original sample rate must be provided"
        else:
            self.source = "wav"
        self.cache = {}
        self.original_sr = original_sr
        self.target_sr = target_sr
        if original_sr is not None:
            self.resampler = torchaudio.transforms.Resample(original_sr, target_sr)

    def __getitem__(self, idx):
        aid = self.aids[idx]
        if self.source == "hdf5":
            waveform = read_from_h5(aid, self.aid_to_fname, self.cache)
            waveform = torch.as_tensor(waveform)
            waveform = self.resampler(waveform)
        elif self.source == "wav":
            waveform = load_audio(self.aid_to_fname[aid], self.target_sr)
        return aid, waveform

    def __len__(self):
        return len(self.aids)


class WavPadCollate:

    def __init__(self, min_duration=0.32, sample_rate=32000):
        self.min_length = int(min_duration * sample_rate)

    def __call__(self, data_list):
        """
        data_list: [(aid1, waveform1), (aid2, waveform2), ...]
        """
        aids = []
        waveforms = []
        lengths = []
        blacklist_aids = []
        for item in data_list:
            waveform = item[1]
            audio_id = item[0]
            if waveform is None or len(waveform) < self.min_length:
                blacklist_aids.append(audio_id)
                continue
            aids.append(audio_id)
            waveforms.append(waveform)
            lengths.append(waveform.shape[0])
        np_waveforms = np.zeros((len(waveforms), max(lengths)))
        for idx, waveform in enumerate(waveforms):
            np_waveforms[idx, :len(waveform)] = waveform
        return {
            "aid": np.array(aids),
            "wav": np_waveforms,
            "wav_len": np.array(lengths), 
            "blacklist_aid": blacklist_aids
        }


def inference(input,
              output,
              checkpoint,
              batch_size=32,
              num_process=4,
              original_sr=None,
              target_sr=32000,
              min_duration=0.32,
              **sampling_kwargs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    exp_dir = Path(checkpoint).parent

    cfg = train_util.parse_config_or_kwargs(exp_dir / "config.yaml")
    model, tokenizer = load_model(cfg, checkpoint, device)

    if Path(input).suffix == ".csv":
        wav_df = pd.read_csv(input, sep="\t")
        if "file_name" in wav_df.columns:
            aid_to_fname = dict(zip(wav_df["audio_id"], wav_df["file_name"]))
        elif "hdf5_path" in wav_df.columns:
            aid_to_fname = dict(zip(wav_df["audio_id"], wav_df["hdf5_path"]))
    elif Path(input).suffix in [".wav", ".mp3", ".flac", ".mp4"]:
        audio_id = Path(input).name
        aid_to_fname = {audio_id: input}
    else:
        raise Exception("input should be wav.csv or audio file")

    dataloader = torch.utils.data.DataLoader(
        InferenceDataset(aid_to_fname, original_sr, target_sr),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=WavPadCollate(
            min_duration=min_duration,
            sample_rate=target_sr
        ),
        num_workers=num_process
    )

    captions = []
    audio_ids = []
    with torch.no_grad(), tqdm(total=len(dataloader)) as pbar:
        for batch in dataloader:
            wav = torch.as_tensor(batch["wav"]).float().to(device)
            input_dict = {
                "mode": "inference",
                "wav": wav,
                "wav_len": torch.as_tensor(batch["wav_len"], dtype=torch.long),
                "specaug": False,
                "sample_method": sampling_kwargs.get("sample_method", "beam"),
            }
            if input_dict["sample_method"] == "beam":
                input_dict["beam_size"] = sampling_kwargs.get("beam_size", 3)
            output_dict = model(input_dict)
            caption_batch = tokenizer.decode(output_dict["seq"].cpu().numpy())
            captions.extend(caption_batch)
            audio_ids.extend(batch["aid"])
            pbar.update()
    
    assert len(audio_ids) == len(captions)
    data = {aid: caption for aid, caption in zip(audio_ids, captions)}
    json.dump(data, open(output, "w"), indent=4)


if __name__ == "__main__":
    fire.Fire(inference)
