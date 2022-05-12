import os
import math
import random
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import h5py
from tqdm import tqdm

from captioning.utils.build_vocab import Vocabulary


class CaptionEvalDataset(torch.utils.data.Dataset):
    
    def __init__(self,
                 raw_audio_to_h5: Dict,
                 fc_audio_to_h5: Dict,
                 attn_audio_to_h5: Dict,
                 load_into_mem: bool = False,
                 transform: Optional[List] = None):
        """audio captioning dataset object for inference and evaluation

        Args:
            raw_audio_to_h5 (Dict): Dictionary (<audio_id>: <hdf5_path>)
            fc_audio_to_h5 (Dict): Dictionary (<audio_id>: <hdf5_path>)
            attn_audio_to_h5 (Dict): Dictionary (<audio_id>: <hdf5_path>)
            transform (List, optional): Defaults to None. Transformation onto the data (List of function)
        """
        self._raw_audio_to_h5 = raw_audio_to_h5
        self._fc_audio_to_h5 = fc_audio_to_h5
        self._attn_audio_to_h5 = attn_audio_to_h5
        self._audio_ids = list(self._raw_audio_to_h5.keys())
        self._dataset_cache = {}
        self._transform = transform
        first_audio_id = next(iter(self._raw_audio_to_h5.keys()))
        with h5py.File(self._raw_audio_to_h5[first_audio_id], 'r') as store:
            self.raw_feat_dim = store[first_audio_id].shape[-1]
        with h5py.File(self._fc_audio_to_h5[first_audio_id], 'r') as store:
            self.fc_feat_dim = store[first_audio_id].shape[-1]
        with h5py.File(self._attn_audio_to_h5[first_audio_id], 'r') as store:
            self.attn_feat_dim = store[first_audio_id].shape[-1]

        self._load_into_mem = load_into_mem
        if self._load_into_mem:
            self.aid_to_raw = {}
            self.aid_to_fc = {}
            self.aid_to_attn = {}
            # TODO load into memory following the original sequence in HDF5
            for audio_id in tqdm(self._audio_ids, ascii=True):
                raw_feat_h5 = self._raw_audio_to_h5[audio_id]
                if not raw_feat_h5 in self._dataset_cache:
                    self._dataset_cache[raw_feat_h5] = h5py.File(raw_feat_h5, "r")
                raw_feat = self._dataset_cache[raw_feat_h5][audio_id][()]

                fc_feat_h5 = self._fc_audio_to_h5[audio_id]
                if not fc_feat_h5 in self._dataset_cache:
                    self._dataset_cache[fc_feat_h5] = h5py.File(fc_feat_h5, "r")
                fc_feat = self._dataset_cache[fc_feat_h5][audio_id][()]

                attn_feat_h5 = self._attn_audio_to_h5[audio_id]
                if not attn_feat_h5 in self._dataset_cache:
                    self._dataset_cache[attn_feat_h5] = h5py.File(attn_feat_h5, "r")
                attn_feat = self._dataset_cache[attn_feat_h5][audio_id][()]
                self.aid_to_raw[audio_id] = raw_feat
                self.aid_to_fc[audio_id] = fc_feat
                self.aid_to_attn[audio_id] = attn_feat

    def __getitem__(self, index):
        audio_id = self._audio_ids[index]

        if self._load_into_mem:
            raw_feat = self.aid_to_raw[audio_id]
            fc_feat = self.aid_to_fc[audio_id]
            attn_feat = self.aid_to_attn[audio_id]
        else:
            raw_feat_h5 = self._raw_audio_to_h5[audio_id]
            if not raw_feat_h5 in self._dataset_cache:
                self._dataset_cache[raw_feat_h5] = h5py.File(raw_feat_h5, "r")
            raw_feat = self._dataset_cache[raw_feat_h5][audio_id][()]

            fc_feat_h5 = self._fc_audio_to_h5[audio_id]
            if not fc_feat_h5 in self._dataset_cache:
                self._dataset_cache[fc_feat_h5] = h5py.File(fc_feat_h5, "r")
            fc_feat = self._dataset_cache[fc_feat_h5][audio_id][()]

            attn_feat_h5 = self._attn_audio_to_h5[audio_id]
            if not attn_feat_h5 in self._dataset_cache:
                self._dataset_cache[attn_feat_h5] = h5py.File(attn_feat_h5, "r")
            attn_feat = self._dataset_cache[attn_feat_h5][audio_id][()]

        if self._transform:
            # for transform in self._transform:
                # raw_feat = transform(raw_feat)
            for transform in self._transform:
                attn_feat = transform(attn_feat)
        return audio_id, torch.as_tensor(raw_feat), torch.as_tensor(fc_feat), torch.as_tensor(attn_feat)

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
                 raw_audio_to_h5: Dict,
                 fc_audio_to_h5: Dict,
                 attn_audio_to_h5: Dict,
                 caption_info: List,
                 vocabulary: Vocabulary,
                 load_into_mem: bool = False,
                 transform: Optional[List] = None):
        """Dataloader for audio captioning dataset

        Args:
            raw_audio_to_h5 (Dict): Mapping from audio id to raw feature hdf5 (<audio_id>: <hdf5_path>)
            fc_audio_to_h5 (Dict): Mapping from audio id to pre-trained fc hdf5 (<audio_id>: <hdf5_path>)
            attn_audio_to_h5 (Dict): Mapping from audio id to pre-trained attn feature hdf5 (<audio_id>: <hdf5_path>)
            vocabulary (Vocabulary): Preloaded vocabulary object 
            transform (List, optional): Defaults to None. Transformation onto the data (List of function)
        """
        audio_ids = [info["audio_id"] for info in caption_info]
        raw_audio_to_h5 = {aid: raw_audio_to_h5[aid] for aid in audio_ids}
        fc_audio_to_h5 = {aid: fc_audio_to_h5[aid] for aid in audio_ids}
        attn_audio_to_h5 = {aid: attn_audio_to_h5[aid] for aid in audio_ids}
        super().__init__(raw_audio_to_h5, fc_audio_to_h5, attn_audio_to_h5, load_into_mem, transform)
        # Important!!! reset audio id list, otherwise there is problem in matching!
        self._audio_ids = audio_ids
        self._caption_info = caption_info
        self._vocabulary = vocabulary

    def __getitem__(self, index: Tuple):
        """
        index: Tuple (<audio_idx>, <cap_idx>)
        """
        audio_idx, cap_idx = index
        audio_id, raw_feat, fc_feat, attn_feat = super().__getitem__(audio_idx)
        # if "raw_name" in self._caption_info[audio_idx]:
            # audio_id = self._caption_info[audio_idx]["raw_name"]
        cap_id = self._caption_info[audio_idx]["captions"][cap_idx]["cap_id"]
        cap_id = str(cap_id)
        tokens = self._caption_info[audio_idx]["captions"][cap_idx]["tokens"].split()
        caption = [self._vocabulary('<start>')] + \
            [self._vocabulary(token) for token in tokens] + \
            [self._vocabulary('<end>')]
        caption = torch.as_tensor(caption)
        return raw_feat, fc_feat, attn_feat, caption, audio_id, cap_id

    def __len__(self):
        length = 0
        for audio_item in self._caption_info:
            length += len(audio_item["captions"])
        return length


class CaptionConditionDataset(CaptionDataset):

    def __init__(self,
                 raw_audio_to_h5: Dict,
                 fc_audio_to_h5: Dict,
                 attn_audio_to_h5: Dict,
                 caption_info: List,
                 caption_to_condition: Dict,
                 vocabulary: Vocabulary,
                 transform: Optional[List] = None):
        """Dataloader for audio captioning dataset

        Args:
            h5file_dict (Dict): Dictionary (<audio_id>: <hdf5_path>)
            vocabulary (Vocabulary): Preloaded vocabulary object 
            transform (List, optional): Defaults to None. Transformation onto the data (List of function)
        """
        super().__init__(raw_audio_to_h5, fc_audio_to_h5,
                         attn_audio_to_h5, caption_info,
                         vocabulary, transform)
        self._caption_to_condition = caption_to_condition

    def __getitem__(self, index: Tuple):
        raw_feat, fc_feat, attn_feat, caption, \
            audio_id, cap_id = super().__getitem__(index)
        condition = self._caption_to_condition[f"{audio_id}_{cap_id}"]
        condition = torch.as_tensor(condition)
        return raw_feat, fc_feat, attn_feat, condition, caption, audio_id, cap_id


class CaptionStructureDataset(CaptionDataset):

    def __init__(self,
                 raw_audio_to_h5: Dict,
                 fc_audio_to_h5: Dict,
                 attn_audio_to_h5: Dict,
                 caption_info: List,
                 vocabulary: Vocabulary,
                 caption_to_structure: Dict,
                 transform: Optional[List] = None):
        """Dataloader for audio captioning dataset

        Args:
            h5file_dict (Dict): Dictionary (<audio_id>: <hdf5_path>)
            vocabulary (Vocabulary): Preloaded vocabulary object 
            transform (List, optional): Defaults to None. Transformation onto the data (List of function)
        """
        super().__init__(raw_audio_to_h5, fc_audio_to_h5,
                         attn_audio_to_h5, caption_info,
                         vocabulary, transform)
        self._caption_to_structure = caption_to_structure

    def __getitem__(self, index: Tuple):
        raw_feat, fc_feat, attn_feat, caption, \
            audio_id, cap_id = super().__getitem__(index)
        structure = self._caption_to_structure[f"{audio_id}_{cap_id}"]
        structure = torch.as_tensor(structure)
        return raw_feat, fc_feat, attn_feat, structure, caption, audio_id, cap_id


class RandomConditionDataset(CaptionEvalDataset):

    def __init__(self,
                 raw_audio_to_h5,
                 fc_audio_to_h5,
                 attn_audio_to_h5,
                 caption_info,
                 caption_to_condition,
                 vocabulary,
                 transform = None):
        super().__init__(raw_audio_to_h5, fc_audio_to_h5,
                         attn_audio_to_h5, transform)
        self._vocabulary = vocabulary
        self._num_audios = len(raw_audio_to_h5)
        self._captions = []
        cap_idx_to_condition = []
        for item in caption_info:
            audio_id = item["audio_id"]
            for cap_item in item["captions"]:
                cap_id = cap_item["cap_id"]
                self._captions.append(cap_item["tokens"])
                cap_idx_to_condition.append(caption_to_condition[f"{audio_id}_{cap_id}"])
        self._cap_idx_to_condition = np.array(cap_idx_to_condition)
        self._min_condition = min(cap_idx_to_condition)
        self._max_condition = max(cap_idx_to_condition)

    def __getitem__(self, index):
        audio_idx = np.random.randint(0, self._num_audios)
        audio_id, raw_feat, fc_feat, attn_feat = super().__getitem__(audio_idx)
        condition = np.random.rand() * (self._max_condition - self._min_condition) + self._min_condition
        cap_idx = np.argmin(np.abs(condition - self._cap_idx_to_condition))
        tokens = self._captions[cap_idx].split()
        caption = [self._vocabulary('<start>')] + \
            [self._vocabulary(token) for token in tokens] + \
            [self._vocabulary('<end>')]
        caption = torch.as_tensor(caption)
        condition = torch.as_tensor(condition)
        return raw_feat, fc_feat, attn_feat, condition, caption, audio_id, "dummy_id"

    def __len__(self):
        return len(self._captions)


class CaptionKeywordProbDataset(CaptionDataset):

    def __init__(self,
                 raw_audio_to_h5: Dict,
                 fc_audio_to_h5: Dict,
                 attn_audio_to_h5: Dict,
                 keyword_audio_to_h5: Dict,
                 caption_info: List,
                 vocabulary: Vocabulary,
                 load_into_mem: bool,
                 transform: Optional[List]):
        super().__init__(raw_audio_to_h5, fc_audio_to_h5, attn_audio_to_h5,
                         caption_info, vocabulary, load_into_mem=load_into_mem,
                         transform=transform)
        self._keyword_audio_to_h5 = keyword_audio_to_h5

    def __getitem__(self, index: Tuple):
        raw_feat, fc_feat, attn_feat, caption, \
            audio_id, cap_id = super().__getitem__(index)
        keyword_h5 = self._keyword_audio_to_h5[audio_id]
        if not keyword_h5 in self._dataset_cache:
            self._dataset_cache[keyword_h5] = h5py.File(keyword_h5, "r")
        keyword = self._dataset_cache[keyword_h5][audio_id][()]
        keyword = torch.as_tensor(keyword)
        return raw_feat, fc_feat, attn_feat, keyword, caption, audio_id 


class CaptionSampler(torch.utils.data.Sampler):

    def __init__(self, 
                 data_source: CaptionDataset, 
                 audio_subset_indices: List = None, 
                 shuffle: bool = False,
                 max_cap_num: int = None):
        self._caption_info = data_source._caption_info
        self._audio_subset_indices = audio_subset_indices
        self._shuffle = shuffle
        self._num_sample = None
        self._max_cap_num = max_cap_num

    def __iter__(self):
        elems = []
        if self._audio_subset_indices is not None:
            audio_idxs = self._audio_subset_indices
        else:
            audio_idxs = range(len(self._caption_info))
        for audio_idx in audio_idxs:
            if self._max_cap_num is None:
                max_cap_num = len(self._caption_info[audio_idx]["captions"])
            else:
                max_cap_num = self._max_cap_num
            for cap_idx in range(max_cap_num):
                elems.append((audio_idx, cap_idx))
        self._num_sample = len(elems)
        if self._shuffle:
            random.shuffle(elems)
        return iter(elems)

    def __len__(self):
        if self._num_sample is None:
            self.__iter__()
        return self._num_sample


class ConditionOverSampler(CaptionSampler):

    def __init__(self, 
                 data_source: CaptionConditionDataset,
                 shuffle: bool = True,
                 threshold: float = 0.9,
                 times: int = 4):
        super().__init__(data_source, None, shuffle)
        self._caption_to_condition = data_source._caption_to_condition
        self._threshold = threshold
        self._times = times

    def __iter__(self):
        elems = []
        audio_idxs = range(len(self._caption_info))
        for audio_idx in audio_idxs:
            audio_id = self._caption_info[audio_idx]["audio_id"]
            max_cap_num = len(self._caption_info[audio_idx]["captions"])
            for cap_idx in range(max_cap_num):
                cap_id = self._caption_info[audio_idx]["captions"][cap_idx]["cap_id"]
                condition = self._caption_to_condition[f"{audio_id}_{cap_id}"]
                if condition < self._threshold:
                    for _ in range(self._times):
                        elems.append((audio_idx, cap_idx))
                else:
                    elems.append((audio_idx, cap_idx))
        self._num_sample = len(elems)
        if self._shuffle:
            random.shuffle(elems)
        return iter(elems)

    

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
    import sys
    import json
    import pickle
    from tqdm import tqdm
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'raw_csv',
        default='data/clotho_v2/dev/lms.csv',
        type=str,
        nargs="?")
    parser.add_argument(
        'fc_csv',
        default='data/clotho_v2/dev/panns_wavegram_logmel_cnn14_fc.csv',
        type=str,
        nargs="?")
    parser.add_argument(
        'attn_csv',
        default='data/clotho_v2/dev/panns_wavegram_logmel_cnn14_attn.csv',
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
    raw_df = pd.read_csv(args.raw_csv, sep="\t")
    fc_df = pd.read_csv(args.fc_csv, sep="\t")
    attn_df = pd.read_csv(args.attn_csv, sep="\t")
    vocabulary = pickle.load(open(args.vocab_file, "rb"))
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    dataset = CaptionDataset(
        dict(zip(raw_df["audio_id"], raw_df["hdf5_path"])),
        dict(zip(fc_df["audio_id"], fc_df["hdf5_path"])),
        dict(zip(attn_df["audio_id"], attn_df["hdf5_path"])),
        caption_info,
        vocabulary,
        load_into_mem=True
    )
    # for feat, target in dataset:
        # print(feat.shape, target.shape)
    sampler = CaptionSampler(dataset, shuffle=True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        collate_fn=collate_fn([0, 2, 3], 3),
        num_workers=4,
        sampler=sampler)
    idx = 0
    import time
    start = time.time()
    for batch in tqdm(dataloader, unit="batch"):
        idx += 1
        end = time.time()
        # print("batch ", idx, "{:.3f} seconds".format(end - start))
        start = end
        sys.stdout.flush()
        pass


