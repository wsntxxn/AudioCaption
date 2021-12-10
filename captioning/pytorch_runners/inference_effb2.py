import os
from pathlib import Path
import multiprocessing
import json

import fire
from tqdm import trange
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

def load_feature_extractor(model_file_path: str = None, device: str = None) -> nn.Module:
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
        captioning.models.encoder, config["encoder"])(
        config["data"]["raw_feat_dim"],
        config["data"]["fc_feat_dim"],
        config["data"]["attn_feat_dim"],
        **config["encoder_args"]
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

def load_audio(file_name):
    y, sr = librosa.core.load(file_name, sr=16000)
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
        "data": {
            "raw_feat_dim": 64,
            "fc_feat_dim": 1408,
            "attn_feat_dim": 1408
        },
        "encoder": "RnnEncoder",
        "encoder_args":{
            "bidirectional": True,
            "dropout": 0.5,
            "hidden_size": 256,
            "num_layers": 3
        },
        "decoder": "TransformerDecoder",
        "decoder_args": {
            "attn_emb_dim": 512,
            "dropout": 0.2,
            "emb_dim": 256,
            "fc_emb_dim": 512,
            "nlayers": 2
        },
        "model": "TransformerModel",
        "model_args": {}
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
        file_names = wav_df["file_name"].unique()
    elif Path(input).suffix in [".wav", ".mp3", ".flac"]:
        file_names = [input,]
    else:
        raise Exception("input should be wav.csv or audio file")

    captions = []
    with torch.no_grad():
        for i in trange(0, len(file_names), batch_size):
            file_batch = file_names[i: i + batch_size]
            waveform_batch, length_batch = multiprocess_load_audio(file_batch, num_process=num_process)
            waveform_batch = torch.as_tensor(waveform_batch).float().to(device)
            audio_feat_batch = feature_extractor.extract_embedding(waveform_batch)
            attn_feat_length_batch = np.floor((length_batch / 160 + 1) / 32)
            input_dict = {
                "mode": "inference",
                "raw_feats": None,
                "raw_feat_lens": None,
                "fc_feats": audio_feat_batch["fc_feat"],
                "attn_feats": audio_feat_batch["attn_feat"],
                "attn_feat_lens": torch.as_tensor(attn_feat_length_batch, dtype=torch.long),
                "sample_method": "beam"
            }
            output_dict = model(input_dict)
            caption_batch = [decode_caption(seq, vocabulary) for seq in output_dict["seqs"].cpu().numpy()]
            captions.extend(caption_batch)
    
    data = {os.path.basename(file_name): caption for file_name, caption in zip(file_names, captions)}
    json.dump(data, open(output, "w"), indent=4)


if __name__ == "__main__":
    fire.Fire(inference)
