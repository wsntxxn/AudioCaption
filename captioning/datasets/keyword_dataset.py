import pickle
from typing import List, Dict, Union

import numpy as np
import pandas as pd

from captioning.utils.train_util import load_dict_from_csv
from captioning.datasets.caption_dataset import InferenceDataset, CaptionDataset, \
    read_from_h5


class KeywordProbInferenceDataset(InferenceDataset):

    def __init__(self, features: Dict, transforms: Dict, keyword_prob: str,
                 load_into_mem: bool = False, audio_ids: List = None,
                 threshold: Union[float, str] = None):
        super().__init__(features, transforms, load_into_mem=load_into_mem,
            audio_ids=audio_ids)
        self.aid_to_h5["keyword"] = load_dict_from_csv(
            keyword_prob, ("audio_id", "hdf5_path"))
        self.threshold = threshold

    def load_keyword(self, audio_id):
        keyword = read_from_h5(audio_id,
                               self.aid_to_h5["keyword"],
                               self.dataset_cache)
        if self.threshold is not None:
            if isinstance(self.threshold, float):
                keyword = np.where(keyword < self.threshold, 0, 1)
            elif isinstance(self.threshold, str):
                assert self.threshold.startswith("top")
                k = int(self.threshold[3:])
                ind = keyword.argsort()
                keyword[ind[-k:]] = 1.0
                keyword[ind[:-k]] = 0.0
        return keyword
    
    def __getitem__(self, index):
        output = super().__getitem__(index)
        audio_id = output["audio_id"]
        output["keyword"] = self.load_keyword(audio_id)
        return output


class KeywordGtInferenceDataset(InferenceDataset):

    def __init__(self, features: Dict, transforms: Dict, keyword_prob: str,
            load_into_mem: bool = False, keyword_encoder: str = None,
            audio_ids: List = None):
        super().__init__(features, transforms, load_into_mem=load_into_mem,
            audio_ids=audio_ids)
        keyword_df = pd.read_csv(keyword_prob, sep="\t").fillna("")
        keyword_df["keywords"] = keyword_df["keywords"].apply(
            lambda x: x.split("; ")
        )
        self.aid_to_keywords = dict(zip(
            keyword_df["audio_id"], keyword_df["keywords"]))
        self.keyword_encoder = pickle.load(open(keyword_encoder, "rb"))

    def load_keyword(self, audio_id):
        keywords = self.aid_to_keywords[audio_id]
        keyword = self.keyword_encoder.transform([keywords])[0]
        return keyword
    
    def __getitem__(self, index):
        output = super().__getitem__(index)
        audio_id = output["audio_id"]
        output["keyword"] = self.load_keyword(audio_id)
        return output


class CaptionKeywordProbDataset(CaptionDataset):

    def __init__(self,
                 features: Dict,
                 transforms: Dict,
                 caption: str,
                 vocabulary: str,
                 keyword_prob: str,
                 load_into_mem: bool = False,
                 keyword_encoder: str = None):
        super().__init__(features, transforms, caption,
                         vocabulary, load_into_mem)
        with open(keyword_prob, "r") as reader:
            line = reader.readline()
            header = line.strip().split("\t")
        if header == ["audio_id", "hdf5_path"]:
            self.aid_to_h5["keyword"] = load_dict_from_csv(
                keyword_prob, ("audio_id", "hdf5_path"))
        elif header == ["cap_id", "keywords"]:
            keyword_df = pd.read_csv(keyword_prob, sep="\t").fillna("")
            keyword_df["keywords"] = keyword_df["keywords"].apply(
                lambda x: x.split("; ")
            )
            self.cid_to_keywords = dict(zip(
                keyword_df["cap_id"], keyword_df["keywords"]))

            assert keyword_encoder is not None, "keyword encoder must be provided"
            self.keyword_encoder = pickle.load(open(keyword_encoder, "rb"))
            # self.keyword_to_idx = {idx: keyword for idx, keyword in enumerate(
                # keyword_encoder.__class__)}
        else:
            raise Exception(f"unsupported keyword file header {header}")


    def load_audio_keyword(self, audio_id):
        keyword = read_from_h5(audio_id,
                               self.aid_to_h5["keyword"],
                               self.dataset_cache)
        return keyword

    def load_caption_keyword(self, key):
        keywords = self.cid_to_keywords[key]
        keyword = self.keyword_encoder.transform([keywords])[0]
        return keyword

    def __getitem__(self, index):
        output = super().__getitem__(index)
        audio_id = output["audio_id"]
        if "keyword" in self.aid_to_h5:
            output["keyword"] = self.load_audio_keyword(audio_id)
        else:
            cap_id = output["cap_id"]
            key = f"{audio_id}_{cap_id}"
            output["keyword"] = self.load_caption_keyword(key)
        
        return output


