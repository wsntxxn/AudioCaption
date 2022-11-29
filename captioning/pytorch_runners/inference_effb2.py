import os
from pathlib import Path
import multiprocessing
import json

import fire
from tqdm import trange, tqdm
import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import captioning.models
import captioning.models.encoder
import captioning.models.decoder
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


def load_feature_extractor(model_file_path: str = None,
                           device: str = None) -> nn.Module:
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
    model.eval()
    return model


def load_model(config, checkpoint):
    encoder = getattr(
        captioning.models.encoder, config["encoder"]["type"])(
        **config["encoder"]["args"]
    )
    decoder = getattr(
        captioning.models.decoder, config["decoder"])(
        vocab_size=len(checkpoint["vocabulary"]),
        **config["decoder_args"]
    )
    model = getattr(
        captioning.models, config["model"])(
        encoder, decoder, **config["model_args"]
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint["vocabulary"]


def load_audio(specifier: str):
    assert Path(specifier).exists(), specifier + " not exists!"
    y, sr = librosa.core.load(specifier, sr=None)
    y = librosa.core.resample(y, sr, 32000)
    return y


class InferenceLogMelDataset(torch.utils.data.Dataset):

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
            if len(waveform) < min_length * 32000:
                blacklist_aids.append(audio_id)
                continue
            aids.append(audio_id)
            waveforms.append(waveform)
            lengths.append(waveform.shape[0])
        np_waveforms = np.zeros((len(waveforms), max(lengths)))
        for idx, waveform in enumerate(waveforms):
            np_waveforms[idx, :len(waveform)] = waveform
        return np.array(aids), np_waveforms, np.array(lengths), blacklist_aids
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
              batch_size=32,
              num_process=4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    config = {
        "encoder": {
            "type": "RnnEncoder",
            "args":{
                "bidirectional": True,
                "dropout": 0.5,
                "hidden_size": 256,
                "num_layers": 3
            },
        },
        "decoder": {
            "type": "TransformerDecoder",
            "args": {
                "attn_emb_dim": 512,
                "dropout": 0.2,
                "emb_dim": 256,
                "fc_emb_dim": 512,
                "nlayers": 2
            },
        },
        "type": "TransformerModel",
        "args": {}
    }
    checkpoint = torch.hub.load_state_dict_from_url(
        "https://github.com/wsntxxn/AudioCaption/releases/download/v0.0.1/audiocaps_effb2_rnn_trm.pth",
        progress=True,
        map_location="cpu")
    model, vocabulary = load_model(config, checkpoint)
    model = model.to(device)
    feature_extractor = load_feature_extractor()
    feature_extractor = feature_extractor.to(device)

    if Path(input).suffix == ".csv":
        wav_df = pd.read_csv(input, sep="\t")
        aid_to_fname = dict(zip(wav_df["audio_id"], wav_df["file_name"]))
    elif Path(input).suffix in [".wav", ".mp3", ".flac"]:
        audio_id = Path(input).name
        aid_to_fname = {audio_id: input}
    else:
        raise Exception("input should be wav.csv or audio file")

    dataloader = torch.utils.data.DataLoader(
        InferenceLogMelDataset(aid_to_fname),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_wrapper(0.32),
        num_workers=num_process
    )

    captions = []
    audio_ids = []
    with torch.no_grad(), tqdm(total=len(dataloader)) as pbar:
        for batch in dataloader:
            aids, waveforms, lengths, _ = batch
            waveforms = torch.as_tensor(waveforms).float().to(device)
            audio_feats = feature_extractor.extract_embedding(waveforms)
            attn_feat_lengths = np.floor((lengths / 160 + 1) / 32)
            input_dict = {
                "mode": "inference",
                "raw_feats": None,
                "raw_feat_lens": None,
                "fc_feats": audio_feats["fc_feat"],
                "attn_feats": audio_feats["attn_feat"],
                "attn_feat_lens": torch.as_tensor(attn_feat_lengths, dtype=torch.long),
                "sample_method": "beam"
            }
            output_dict = model(input_dict)
            caption_batch = [decode_caption(seq, vocabulary) for seq in output_dict["seqs"].cpu().numpy()]
            captions.extend(caption_batch)
            audio_ids.extend(aids)
            pbar.update()
    
    assert len(audio_ids) == len(captions)
    data = {aid: caption for aid, caption in zip(audio_ids, captions)}
    json.dump(data, open(output, "w"), indent=4)


if __name__ == "__main__":
    fire.Fire(inference)
