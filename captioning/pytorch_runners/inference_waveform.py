import sys
import os
from pathlib import Path
import json

import fire
from tqdm import tqdm, trange
import librosa
import numpy as np
import pandas as pd
import torch
import kaldiio
import h5py

import captioning.models
import captioning.models.encoder
import captioning.models.decoder
import captioning.utils.train_util as train_util


def print_pass(*args, **kwargs):
    pass


def load_model(config, checkpoint):
    ckpt = torch.load(checkpoint, "cpu")
    encoder_cfg = config["model"]["encoder"]
    encoder = train_util.init_obj(
        captioning.models.encoder,
        encoder_cfg
    )
    if "pretrained" in encoder_cfg:
        pretrained = encoder_cfg["pretrained"]
        train_util.load_pretrained_model(encoder,
                                         pretrained,
                                         print_pass)
    decoder_cfg = config["model"]["decoder"]
    if "vocab_size" not in decoder_cfg["args"]:
        decoder_cfg["args"]["vocab_size"] = len(ckpt["vocabulary"])
    decoder = train_util.init_obj(
        captioning.models.decoder,
        decoder_cfg
    )
    if "word_embedding" in decoder_cfg:
        decoder.load_word_embedding(**decoder_cfg["word_embedding"])
    if "pretrained" in decoder_cfg:
        pretrained = decoder_cfg["pretrained"]
        train_util.load_pretrained_model(decoder,
                                         pretrained,
                                         sys.stdout.write)
    model = train_util.init_obj(captioning.models, config["model"],
        encoder=encoder, decoder=decoder)
    train_util.load_pretrained_model(model, ckpt, print_pass)
    model.eval()
    return {
        "model": model,
        "vocabulary": ckpt["vocabulary"]
    }


def load_audio(specifier: str, target_sr: int):
    if specifier.endswith("|"):
        fd = kaldiio.utils.open_like_kaldi(specifier, "rb")
        mat = kaldiio.matio._load_mat(fd, None)
        fd.close()
        sr, y = mat
        y = y.copy() / 2 ** 15
    else:
        assert Path(specifier).exists(), specifier + " not exists!"
        y, sr = librosa.core.load(specifier, sr=None)
    if y.shape[0] == 0:
        return None
    y = librosa.core.resample(y, orig_sr=sr, target_sr=target_sr)

    # y = np.array(np.array(y, dtype=np.float16), dtype=np.float32)
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

    def __getitem__(self, idx):
        aid = self.aids[idx]
        if self.source == "hdf5":
            waveform = read_from_h5(aid, self.aid_to_fname, self.cache)
            waveform = np.array(waveform, dtype=np.float32)
            waveform = librosa.core.resample(waveform, orig_sr=self.original_sr, target_sr=self.target_sr)
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


def decode_caption(word_ids, vocabulary):
    candidate = []
    for word_id in word_ids:
        word = vocabulary[word_id]
        if word == "<end>":
            break
        elif word == "<start>":
            continue
        candidate.append(word)
    candidate = " ".join(candidate)
    return candidate


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

    experiment_path = Path(checkpoint).parent

    config = train_util.parse_config_or_kwargs(experiment_path / "config.yaml")
    resumed = load_model(config, checkpoint)
    model = resumed["model"]
    vocabulary = resumed["vocabulary"]
    model = model.to(device)

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
            caption_batch = [decode_caption(seq, vocabulary) for seq in \
                output_dict["seq"].cpu().numpy()]
            captions.extend(caption_batch)
            audio_ids.extend(batch["aid"])
            pbar.update()
    
    assert len(audio_ids) == len(captions)
    data = {aid: caption for aid, caption in zip(audio_ids, captions)}
    json.dump(data, open(output, "w"), indent=4)


if __name__ == "__main__":
    fire.Fire(inference)
