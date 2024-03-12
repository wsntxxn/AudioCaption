from typing import Dict, Union, List
import random
import json

import torch
import librosa
from torch.utils.data import Dataset
import numpy as np
import torchaudio
from torchaudio.transforms import Resample
from torchaudio.functional import resample

from captioning.utils.train_util import load_dict_from_csv, init_obj_from_dict, \
    read_from_h5


def parse_transform(transforms: Dict):
    transform_fn = {}
    for feat_type, transform in transforms.items():
        if transform is None:
            transform_fn[feat_type] = None
        else:
            fns = []
            for fn_config in transform:
                fns.append(init_obj_from_dict(fn_config))
            transform_fn[feat_type] = fns
    return transform_fn


class InferenceDataset(Dataset):

    def __init__(self,
                 aid_to_wav,
                 features: Dict,
                 transforms: Dict,
                 audio_ids: Union[List, str] = None,
                 sample_rate: int = 32000,
                 duration: float = None,
                 ) -> None:
        super().__init__()
        if not isinstance(aid_to_wav, dict):
            aid_to_wav = load_dict_from_csv(aid_to_wav, ["audio_id", "file_name"])
        self.aid_to_wav = aid_to_wav
        self.sample_rate = sample_rate
        self.orig_sr = None
        self.resampler = None
        self.duration = duration
        if duration is not None:
            self.n_samples = int(duration * sample_rate)
        else:
            self.n_samples = None
        self.transforms = parse_transform(transforms)
        self.audio_ids = audio_ids
        if isinstance(self.audio_ids, str):
            audio_ids = []
            with open(self.audio_ids, "r") as reader:
                for line in reader.readlines():
                    audio_ids.append(line.strip())
            self.audio_ids = audio_ids

        # pre-computed featrues (except waveform)        
        self.feat_types = list(features.keys())
        self.aid_to_h5 = {}
        for feat_type, filename in features.items():
            if filename is not None:
                self.aid_to_h5[feat_type] = load_dict_from_csv(
                    filename, ("audio_id", "hdf5_path"))
                if self.audio_ids is None:
                    self.audio_ids = list(self.aid_to_h5[feat_type].keys())
        if self.audio_ids is None:
            self.audio_ids = list(self.aid_to_wav.keys())

        self.dataset_cache = {}
        first_audio_id = self.audio_ids[0]
        self.feat_dim = {}
        for feat_type in self.feat_types:
            if feat_type in self.aid_to_h5:
                self.feat_dim[feat_type] = read_from_h5(
                    first_audio_id, self.aid_to_h5[feat_type],
                    self.dataset_cache).shape[-1]

    def load_audio(self, fpath, sample_rate, n_samples):
        waveform, orig_sr = librosa.core.load(fpath, sr=None)
        waveform = torch.as_tensor(waveform)
        if self.orig_sr is None:
            self.orig_sr = orig_sr
            self.resampler = Resample(orig_sr, sample_rate)
        if orig_sr != sample_rate:
            if orig_sr != self.orig_sr:
                waveform = resample(waveform, orig_sr, sample_rate)
            else:
                waveform = self.resampler(waveform)
        if n_samples is not None:
            if len(waveform) < n_samples:
                waveform = torch.cat([waveform,
                                      torch.zeros(n_samples - len(waveform))])
            else:
                start = random.randint(0, len(waveform) - n_samples)
                waveform = waveform[start: start + n_samples]
        return waveform

    def load_feature(self, audio_id):
        output = {}
        for feat_type in self.feat_types:
            output[feat_type] = read_from_h5(
                audio_id, self.aid_to_h5[feat_type], self.dataset_cache)
            output[feat_type] = np.array(output[feat_type], dtype=np.float32)
        return output

    def load_wav_feat(self, audio_id):
        waveform = self.load_audio(self.aid_to_wav[audio_id],
                                   self.sample_rate,
                                   self.n_samples)
        feats = self.load_feature(audio_id)
        feats["wav"] = waveform

        for feat_type in self.feat_types:
            transform = self.transforms[feat_type]
            if transform is not None:
                for fn in transform:
                    feats[feat_type] = fn(feats[feat_type])
        feats["audio_id"] = audio_id
        return feats

    def __getitem__(self, index):
        audio_id = self.audio_ids[index]
        return self.load_wav_feat(audio_id)

    def __len__(self):
        return len(self.audio_ids)

    def __del__(self):
        for h5 in self.dataset_cache.values():
            if h5:
                try:
                    h5.close()
                except Exception:
                    pass


class CaptionDataset(InferenceDataset):

    def __init__(self,
                 aid_to_wav,
                 features: Dict,
                 transforms: Dict,
                 caption: str,
                 sample_rate: int = 32000,
                 duration: float = None,
                 ):
        self.caption_info = json.load(open(caption))["audios"]
        self.key_to_caps = {}
        self.keys = []
        audio_ids = []
        for item in self.caption_info:
            audio_id = item["audio_id"]
            audio_ids.append(audio_id)
            self.key_to_caps[audio_id] = {}
            for cap_idx, cap_item in enumerate(item["captions"]):
                if "cap_id" in cap_item:
                    cap_id = str(cap_item["cap_id"])
                else:
                    cap_id = str(cap_idx)
                self.key_to_caps[audio_id][cap_id] = cap_item["tokens"]
                self.keys.append((audio_id, cap_id))
        super().__init__(aid_to_wav,
                         features,
                         transforms,
                         audio_ids,
                         sample_rate,
                         duration)

    def __getitem__(self, index):
        audio_id, cap_id = self.keys[index]
        output = super().load_wav_feat(audio_id)

        cap = self.key_to_caps[audio_id][cap_id]
        output["cap"] = cap
        output["cap_id"] = cap_id
        return output

    def __len__(self):
        return len(self.keys)


class InferKdDataset(InferenceDataset):

    def __init__(self,
                 aid_to_wav,
                 features: Dict,
                 transforms: Dict,
                 audio_ids: Union[List, str] = None,
                 sample_rate: int = 32000,
                 teacher_sample_rate: int = 32000,
                 duration: float = None,
                 teacher_duration: float = None,
                 cntr_aug: bool = False) -> None:
        super().__init__(aid_to_wav,
                         features,
                         transforms,
                         audio_ids,
                         sample_rate,
                         duration)
        self.teacher_sample_rate = teacher_sample_rate
        self.teacher_resampler = None
        self.teacher_duration = teacher_duration
        if teacher_duration is not None:
            self.teacher_n_samples = int(teacher_duration * teacher_sample_rate)
        else:
            self.teacher_n_samples = None
        self.cntr_aug = cntr_aug
        if cntr_aug:
            from torchaudio_augmentations import RandomApply, PolarityInversion, Noise, Gain, \
                HighLowPass, Delay, PitchShift, Reverb, Compose
            augs = [
                RandomApply([PolarityInversion()], p=0.8),
                RandomApply([Noise(min_snr=0.001, max_snr=0.005)], p=0.3),
                RandomApply([Gain()], p=0.2),
                HighLowPass(sample_rate=sample_rate),
                RandomApply([Delay(sample_rate=sample_rate)], p=0.5),
                RandomApply([PitchShift(
                    n_samples=self.n_samples,
                    sample_rate=sample_rate,
                )], p=0.4),
                RandomApply([Reverb(sample_rate=sample_rate)], p=0.3)
            ]
            self.aug = Compose(augs)
            teacher_augs = [
                RandomApply([PolarityInversion()], p=0.8),
                RandomApply([Noise(min_snr=0.001, max_snr=0.005)], p=0.3),
                RandomApply([Gain()], p=0.2),
                HighLowPass(sample_rate=teacher_sample_rate),
                RandomApply([Delay(sample_rate=teacher_sample_rate)], p=0.5),
                RandomApply([PitchShift(
                    n_samples=self.teacher_n_samples,
                    sample_rate=teacher_sample_rate,
                )], p=0.4),
                RandomApply([Reverb(sample_rate=teacher_sample_rate)], p=0.3)
            ]
            self.teacher_aug = Compose(teacher_augs)

    def load_audio(self, fpath):
        waveform, orig_sr = librosa.core.load(fpath, sr=None)

        waveform = torch.as_tensor(waveform)
        teacher_waveform = waveform.clone() # current: self.sample_rate, target: self.teacher_sample_rate
        if self.orig_sr is None:
            self.orig_sr = orig_sr
            self.resampler = Resample(orig_sr, self.sample_rate)
            self.teacher_resampler = Resample(orig_sr, self.teacher_sample_rate)

        if orig_sr != self.sample_rate:
            if orig_sr != self.orig_sr:
                waveform = resample(waveform, orig_sr, self.sample_rate)
            else:
                waveform = self.resampler(waveform)
        if self.n_samples is not None:
            if len(waveform) < self.n_samples:
                waveform = torch.cat([waveform,
                                      torch.zeros(self.n_samples - len(waveform))])
            else:
                start = random.randint(0, len(waveform) - self.n_samples)
                waveform = waveform[start: start + self.n_samples]
        
        # resample teacher waveform
        if orig_sr != self.teacher_sample_rate:
            if orig_sr != self.orig_sr:
                teacher_waveform = resample(teacher_waveform, orig_sr, self.teacher_sample_rate)
            else:
                teacher_waveform = self.teacher_resampler(teacher_waveform)
        # pad or crop teacher waveform
        # when crop, avoid `waveform -> student sr -> teacher sr`, otherwise frequency will lose during resampling
        if self.teacher_n_samples is not None:
            if len(teacher_waveform) < self.teacher_n_samples:
                teacher_waveform = torch.cat([
                    teacher_waveform,
                    torch.zeros(self.teacher_n_samples - len(teacher_waveform))])
            else:
                teacher_start = self.teacher_sample_rate * start // self.sample_rate
                teacher_waveform = teacher_waveform[teacher_start: teacher_start + self.teacher_n_samples]
        
        if self.cntr_aug:
            waveform = self.aug(waveform.unsqueeze(0)).squeeze(0)
            teacher_waveform = self.teacher_aug(teacher_waveform.unsqueeze(0)).squeeze(0)
        return waveform, teacher_waveform

    def load_wav_feat(self, audio_id):
        waveform, teacher_waveform = self.load_audio(
            self.aid_to_wav[audio_id],
        )
        feats = self.load_feature(audio_id)
        feats["wav"] = waveform
        feats["teacher_wav"] = teacher_waveform

        for feat_type in self.feat_types:
            transform = self.transforms[feat_type]
            if transform is not None:
                for fn in transform:
                    feats[feat_type] = fn(feats[feat_type])
        feats["audio_id"] = audio_id
        return feats


class CaptionKdDataset(InferKdDataset):

    def __init__(self,
                 aid_to_wav,
                 features: Dict,
                 transforms: Dict,
                 caption: str,
                 sample_rate: int = 32000,
                 teacher_sample_rate: int = 32000,
                 teacher_duration: float = None,
                 duration: float = None,
                 ):
        self.caption_info = json.load(open(caption))["audios"]
        self.key_to_caps = {}
        self.keys = []
        audio_ids = []
        for item in self.caption_info:
            audio_id = item["audio_id"]
            audio_ids.append(audio_id)
            self.key_to_caps[audio_id] = {}
            for cap_idx, cap_item in enumerate(item["captions"]):
                if "cap_id" in cap_item:
                    cap_id = str(cap_item["cap_id"])
                else:
                    cap_id = str(cap_idx)
                self.key_to_caps[audio_id][cap_id] = cap_item["tokens"]
                self.keys.append((audio_id, cap_id))
        super().__init__(aid_to_wav,
                         features,
                         transforms,
                         audio_ids,
                         sample_rate,
                         teacher_sample_rate,
                         duration,
                         teacher_duration)

    def __getitem__(self, index):
        audio_id, cap_id = self.keys[index]
        output = super().load_wav_feat(audio_id)

        cap = self.key_to_caps[audio_id][cap_id]
        output["cap"] = cap
        output["cap_id"] = cap_id
        return output

    def __len__(self):
        return len(self.keys)


if __name__ == "__main__":
    import captioning.utils.train_util as train_util
    import argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")

    args = parser.parse_args()
    config = train_util.load_config(args.config)

    split = args.split
    dataset = train_util.init_obj_from_dict(config["data"][split]["dataset"])
    collate_fn = train_util.init_obj_from_dict(config["data"][split]["collate_fn"])
    dataloader = torch.utils.data.DataLoader(dataset,
                                             collate_fn=collate_fn,
                                             **config["data"][split]["dataloader_args"])
    for batch in tqdm(dataloader):
        pass
