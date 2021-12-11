import math
import random
import sys
from typing import List, Optional, Dict, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import h5py

from captioning.utils.build_vocab import Vocabulary

class CaptionEvalDataset(torch.utils.data.Dataset):
    
    def __init__(self,
                 h5file_dict: Dict,
                 transform: Optional[List] = None):
        """audio captioning dataset object for inference and evaluation

        Args:
            h5file_dict (Dict): Dictionary (<audio_id>: <hdf5_path>)
            transform (List, optional): Defaults to None. Transformation onto the data (List of function)
        """
        self._h5file_dict = h5file_dict
        self._audio_ids = list(self._h5file_dict.keys())
        self._dataset_cache = {}
        self._transform = transform
        first_audio_id = next(iter(self._h5file_dict.keys()))
        with h5py.File(self._h5file_dict[first_audio_id], 'r') as store:
            self.data_dim = store[first_audio_id].shape[-1]

    def __getitem__(self, index):
        audio_id = self._audio_ids[index]
        h5_path = self._h5file_dict[audio_id]
        if not audio_id in self._dataset_cache:
            self._dataset_cache[audio_id] = h5py.File(h5_path, "r")
        feature = self._dataset_cache[audio_id][audio_id][()]
        if self._transform:
            for transform in self._transform:
                feature = transform(feature)
        return audio_id, torch.as_tensor(feature)

    def __len__(self):
        return len(self._audio_ids)

    def __del__(self):
        for k, cache in self._dataset_cache.items():
            if cache:
                try:
                    cache.close()
                except:
                    pass


class CaptionDataset(CaptionEvalDataset):

    def __init__(self,
                 h5file_dict: Dict,
                 caption_info: List,
                 vocabulary: Vocabulary,
                 transform: Optional[List] = None):
        """Dataloader for audio captioning dataset

        Args:
            h5file_dict (Dict): Dictionary (<audio_id>: <hdf5_path>)
            vocabulary (Vocabulary): Preloaded vocabulary object 
            transform (List, optional): Defaults to None. Transformation onto the data (List of function)
        """
        super().__init__(h5file_dict, transform)
        # Important!!! reset audio id list, otherwise there is problem in matching!
        self._audio_ids = [info["audio_id"] for info in caption_info]
        self._dataset_cache = {}
        self._caption_info = caption_info
        self._vocabulary = vocabulary

    def __getitem__(self, index: Tuple):
        """
        index: Tuple (<audio_idx>, <cap_idx>)
        """
        audio_idx, cap_idx = index
        audio_id, feature = super().__getitem__(audio_idx)
        if "raw_name" in self._caption_info[audio_idx]:
            audio_id = self._caption_info[audio_idx]["raw_name"]
        # cap_id = self._caption_info[audio_idx]["captions"][cap_idx]["cap_id"]
        tokens = self._caption_info[audio_idx]["captions"][cap_idx]["tokens"].split()
        caption = [self._vocabulary('<start>')] + \
            [self._vocabulary(token) for token in tokens] + \
            [self._vocabulary('<end>')]
        caption = torch.as_tensor(caption)
        return feature, caption, audio_id

    def __len__(self):
        length = 0
        for audio_item in self._caption_info:
            length += len(audio_item["captions"])
        return length


# class CaptionSentenceDataset(CaptionDataset):

    # def __init__(self, feature: str, caption_df: pd.DataFrame, vocabulary: Vocabulary,
            # sentence_embedding: np.ndarray, transform: Optional[List] = None):
        # super().__init__(feature,
            # caption_df, vocabulary, transform)
        # self.sentence_embedding = sentence_embedding

    # def __getitem__(self, index: int):
        # feature, caption, dataid = super().__getitem__(index)
        # caption_id = self._caption_df.iloc[index]["caption_index"]
        # sentence_embedding = self.sentence_embedding["{}_{}".format(dataid, caption_id)]
        # sentence_embedding = torch.as_tensor(sentence_embedding)
        # return feature, caption, sentence_embedding, dataid


# class CaptionInstanceDataset(CaptionDataset):

    # def __init__(self, feature: str, caption_df: pd.DataFrame,
            # vocabulary: Vocabulary, transform: Optional[List] = None):
        # super().__init__(feature,
            # caption_df, vocabulary, transform)
    
    # def __getitem__(self, index: int):
        # feature, caption, dataid = super().__getitem__(index)
        # caption_id = self._caption_df.iloc[index]["caption_index"] - 1
        # caption_id = torch.tensor(caption_id)
        # return feature, caption, caption_id, dataid

class CaptionSampler(torch.utils.data.Sampler):

    def __init__(self, data_source: CaptionDataset, audio_subset_indices: List = None, shuffle: bool = False):
        self._caption_info = data_source._caption_info
        self._audio_subset_indices = audio_subset_indices
        self._shuffle = shuffle
        self._num_sample = None

    def __iter__(self):
        elems = []
        if self._audio_subset_indices is not None:
            audio_idxs = self._audio_subset_indices
        else:
            audio_idxs = range(len(self._caption_info))
        for audio_idx in audio_idxs:
            for cap_idx in range(len(self._caption_info[audio_idx]["captions"])):
                elems.append((audio_idx, cap_idx))
        self._num_sample = len(elems)
        if self._shuffle:
            random.shuffle(elems)
        return iter(elems)

    def __len__(self):
        if self._num_sample is None:
            self.__iter__()
        return self._num_sample

class CaptionDistributedSampler(torch.utils.data.distributed.DistributedSampler):

    def __init__(self, dataset, audio_subset_indices: List = None, shuffle: bool = True):
        super().__init__(dataset, None, None, shuffle, 0, False)
        self._caption_info = dataset._caption_info
        self._audio_subset_indices = audio_subset_indices
        elems = []
        if self._audio_subset_indices is not None:
            audio_idxs = self._audio_subset_indices
        else:
            audio_idxs = range(len(self._caption_info))
        for audio_idx in audio_idxs:
            for cap_idx in range(len(self._caption_info[audio_idx]["captions"])):
                elems.append((audio_idx, cap_idx))
        self.indices = elems
        self.num_samples = len(elems)
        if self.drop_last and len(self.num_samples) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                # `type:ignore` is required because Dataset cannot provide a default __len__
                # see NOTE in pytorch/torch/utils/data/sampler.py
                (self.num_samples - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(self.num_samples / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            random.seed(self.seed + self.epoch)
            random.shuffle(self.indices)
        indices = self.indices

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size
        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)

def collate_fn(length_idxs: List = [], sort_idx = None):

    def collate_wrapper(data_batches):
        # x: [feature, caption]
        # data_batches: [[feat1, cap1], [feat2, cap2], ..., [feat_n, cap_n]]
        if sort_idx:
            data_batches.sort(key=lambda x: len(x[sort_idx]), reverse=True)

        def merge_seq(dataseq, dim=0):
            lengths = [seq.shape for seq in dataseq]
            # Assuming duration is given in the first dimension of each sequence
            maxlengths = tuple(np.max(lengths, axis=dim))
            # For the case that the lengths are 2 dimensional
            lengths = np.array(lengths)[:, dim]
            padded = torch.zeros((len(dataseq),) + maxlengths)
            for i, seq in enumerate(dataseq):
                end = lengths[i]
                padded[i, :end] = seq[:end]
            return padded, lengths
        
        data_out = []
        data_len = []
        for idx, data in enumerate(zip(*data_batches)):
            if isinstance(data[0], torch.Tensor):
                if len(data[0].shape) == 0:
                    data_seq = torch.as_tensor(data)
                elif data[0].size(0) > 1:
                    data_seq, tmp_len = merge_seq(data)
                    if idx in length_idxs:
                        # print(tmp_len)
                        data_len.append(tmp_len)
            else:
                data_seq = data
            data_out.append(data_seq)
        data_out.extend(data_len)

        return data_out

    return collate_wrapper


if __name__ == "__main__":
    import argparse
    import json
    import pickle
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'feature_csv',
        default='data/clotho_v2/dev/lms.csv',
        type=str,
        nargs="?")
    parser.add_argument(
        'vocab_file',
        default='data/clotho_v2/dev/vocab.pkl',
        type=str,
        nargs="?")
    parser.add_argument(
        'annotation_file',
        default='data/clotho_v2/dev/text.json',
        type=str,
        nargs="?")
    args = parser.parse_args()
    caption_info = json.load(open(args.annotation_file, "r"))["audios"]
    feature_df = pd.read_csv(args.feature_csv, sep="\t")
    vocabulary = pickle.load(open(args.vocab_file, "rb"))
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    dataset = CaptionDataset(
        dict(zip(feature_df["audio_id"], feature_df["hdf5_path"])),
        caption_info,
        vocabulary
    )
    # for feat, target in dataset:
        # print(feat.shape, target.shape)
    sampler = CaptionSampler(dataset, shuffle=True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        collate_fn=collate_fn([0, 1], 1),
        num_workers=4,
        sampler=sampler)
    agg = 0
    for feat, target, audio_ids, feat_lens, cap_lens in dataloader:
        agg += len(feat)
        print(feat.shape, target.shape, cap_lens)
        print(audio_ids)
        break
    print("Overall seen {} feats (of {})".format(agg, len(dataset)))


