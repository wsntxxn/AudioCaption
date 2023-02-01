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

import captioning.models
import captioning.models.encoder
import captioning.models.decoder
import captioning.utils.train_util as train_util


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
                                         sys.stdout.write)
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
    train_util.load_pretrained_model(model, ckpt)
    model.eval()
    return {
        "model": model,
        "vocabulary": ckpt["vocabulary"]
    }


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
    if y.shape[0] == 0:
        return None
    y = librosa.core.resample(y, sr, 32000)
    return y


class InferenceDataset(torch.utils.data.Dataset):

    def __init__(self, aid_to_fname):
        super().__init__()
        self.aid_to_fname = aid_to_fname
        self.aids = list(aid_to_fname.keys())

    def __getitem__(self, idx):
        aid = self.aids[idx]
        waveform = load_audio(self.aid_to_fname[aid])
        return aid, waveform

    def __len__(self):
        return len(self.aids)


def collate_wrapper(min_length=0.32):
    def collate_fn(data_list):
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
            if waveform is None or len(waveform) < min_length * 32000:
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
    return collate_fn


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
              num_process=4):
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
        aid_to_fname = dict(zip(wav_df["audio_id"], wav_df["file_name"]))
    elif Path(input).suffix in [".wav", ".mp3", ".flac"]:
        audio_id = Path(input).name
        aid_to_fname = {audio_id: input}
    else:
        raise Exception("input should be wav.csv or audio file")

    dataloader = torch.utils.data.DataLoader(
        InferenceDataset(aid_to_fname),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_wrapper(0.32),
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
                "sample_method": "beam",
            }
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
