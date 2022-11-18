import pickle
import random
from typing import List, Dict, Union

import numpy as np
import pandas as pd

from captioning.utils.train_util import load_dict_from_csv
from captioning.datasets.caption_dataset import InferenceDataset, \
    CaptionDataset, read_from_h5


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
                if self.threshold.startswith("top"):
                    # top k keywords
                    k = int(self.threshold[3:])
                    ind = keyword.argsort()
                    keyword[ind[-k:]] = 1.0
                    keyword[ind[:-k]] = 0.0
                else:
                    # top k + threshold
                    threshold, topk = self.threshold.split("_")
                    threshold = float(threshold)
                    onehot = np.where(keyword < threshold, 0, 1)
                    k = int(topk[3:])
                    if np.where(onehot == 1)[0].shape[0] > k:
                        ind = keyword.argsort()
                        keyword[ind[-k:]] = 1.0
                        keyword[ind[:-k]] = 0.0
                    else:
                        keyword = onehot
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
                 keyword_encoder: str = None,
                 dropout_prob: float = 0.0):
        super().__init__(features, transforms, caption,
                         vocabulary, load_into_mem)
        self.dropout_prob = dropout_prob
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
            keyword = self.load_caption_keyword(key)
            # random dropout keywords
            if self.dropout_prob > 0:
                keyword_idxs = np.where(keyword)[0]
                for idx in keyword_idxs:
                    if random.random() < self.dropout_prob:
                        keyword[idx] = 0
            output["keyword"] = keyword
        
        return output


class CaptionMergeNoKeywordDataset(CaptionKeywordProbDataset):

    def __getitem__(self, index):
        key_index = index  // 2
        output = super().__getitem__(key_index)
        if index % 2 == 0: # original data, without any keywords
            output["keyword"] = np.zeros_like(output["keyword"])
        return output

    def __len__(self):
        return 2 * len(self.keys)


class CaptionMergeAllKeywordDataset(CaptionKeywordProbDataset):

    def __init__(self, features: Dict, transforms: Dict, caption: str,
                 vocabulary: str, keyword_prob: str, load_into_mem: bool,
                 keyword_encoder: str, dropout_prob: float):
        assert dropout_prob > 0
        super().__init__(features, transforms, caption, vocabulary,
            keyword_prob, load_into_mem=load_into_mem,
            keyword_encoder=keyword_encoder, dropout_prob=dropout_prob)

    def __getitem__(self, index):
        key_index = index  // 2
        output = super().__getitem__(key_index)
        if index % 2 == 0: # all keywords
            audio_id = output["audio_id"]
            cap_id = output["cap_id"]
            key = f"{audio_id}_{cap_id}"
            keyword = self.load_caption_keyword(key)
            output["keyword"] = keyword
        return output

    def __len__(self):
        return 2 * len(self.keys)
