from typing import Dict
import json
import pickle

import numpy as np

from captioning.utils.train_util import load_dict_from_csv
from captioning.datasets.caption_dataset import InferenceDataset, CaptionDataset


class ConditionCaptionDataset(CaptionDataset):

    def __init__(self,
                 features: Dict,
                 transforms: Dict,
                 caption: str,
                 vocabulary: str,
                 condition: str,
                 load_into_mem: bool = False):
        super().__init__(features=features,
                         transforms=transforms,
                         caption=caption,
                         vocabulary=vocabulary,
                         load_into_mem=load_into_mem)
        self.key_to_condition = load_dict_from_csv(condition, ("cap_id", "prob"))

    def __getitem__(self, index):
        output = super().__getitem__(index)
        audio_id = output["audio_id"]
        cap_id = output["cap_id"]
        condition = self.key_to_condition[f"{audio_id}_{cap_id}"]
        output["condition"] = np.array(condition)
        return output


class RandomConditionDataset(InferenceDataset):

    def __init__(self,
                 features: Dict,
                 transforms: Dict,
                 caption: str,
                 vocabulary: str,
                 condition: str,
                 load_into_mem: bool = False):
        super().__init__(features=features,
                         transforms=transforms,
                         load_into_mem=load_into_mem)
        self.caption_info = json.load(open(caption))["audios"]
        self.vocabulary = pickle.load(open(vocabulary, "rb"))
        self.bos = self.vocabulary('<start>')
        self.eos = self.vocabulary('<end>')

        key_to_condition = load_dict_from_csv(condition, ("cap_id", "prob"))
        self.captions = []
        cap_idx_to_condition = []
        for item in self.caption_info:
            audio_id = item["audio_id"]
            for cap_item in item["captions"]:
                cap_id = cap_item["cap_id"]
                self.captions.append(cap_item["tokens"])
                key = f"{audio_id}_{cap_id}"
                cap_idx_to_condition.append(key_to_condition[key])
        self.cap_idx_to_condition = np.array(cap_idx_to_condition)
        self.min_condition = min(cap_idx_to_condition)
        self.max_condition = max(cap_idx_to_condition)

    def __getitem__(self, index):
        audio_id = np.random.choice(self.audio_ids)
        output = super().load_audio(audio_id)
        output["audio_id"] = audio_id
        condition = np.random.rand() * (
            self.max_condition - self.min_condition) + self.min_condition
        cap_idx = np.argmin(np.abs(condition - self.cap_idx_to_condition))
        tokens = self.captions[cap_idx].split()
        caption = [self.bos] + [self.vocabulary(token) for token in tokens] + \
            [self.eos]
        caption = np.array(caption)
        condition = np.array(condition)
        output["cap"] = caption
        output["condition"] = condition
        return output

    def __len__(self):
        return len(self.captions)


class ConditionOverSampleDataset(ConditionCaptionDataset):

    def __init__(self, features: Dict, transforms: Dict, caption: str,
                 vocabulary: str, condition: str, load_into_mem: bool = False,
                 threshold: float = 0.9, times: int = 4):
        super().__init__(features, transforms, caption, vocabulary,
                         condition, load_into_mem=load_into_mem)
        self.keys = []
        for item in self.caption_info:
            audio_id = item["audio_id"]
            max_cap_num = len(item["captions"])
            for cap_idx in range(max_cap_num):
                cap_id = item["captions"][cap_idx]["cap_id"]
                cond = self.key_to_condition[f"{audio_id}_{cap_id}"]
                if cond < threshold:
                    for _ in range(times):
                        self.keys.append((audio_id, cap_id))
                else:
                    self.keys.append((audio_id, cap_id))
