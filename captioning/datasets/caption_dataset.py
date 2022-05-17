import os
import json
import pickle
import math
import random
from typing import List, Dict, Union

import numpy as np
import torch
import h5py
from tqdm import tqdm

from captioning.utils.train_util import load_dict_from_csv
import captioning.datasets.augment as augment


def read_from_h5(key: str, key_to_h5: Dict, cache: Dict):
    hdf5_path = key_to_h5[key]
    if hdf5_path not in cache:
        cache[hdf5_path] = h5py.File(hdf5_path, "r")
    return cache[hdf5_path][key][()]


def parse_transform(transforms: Dict):
    transform_fn = {}
    for feat_type, transform in transforms.items():
        if transform is None:
            transform_fn[feat_type] = None
        else:
            fns = []
            for fn_config in transform:
                fns.append(getattr(augment, fn_config["type"])(
                    **fn_config["args"]))
            transform_fn[feat_type] = fns
    return transform_fn


class InferenceDataset(torch.utils.data.Dataset):
    
    def __init__(self,
                 features: Dict,
                 transforms: Dict,
                 load_into_mem: bool = False,
                 audio_ids: Union[List, str] = None):
        self.feat_types = features.keys()
        self.transforms = parse_transform(transforms)
        self.aid_to_h5 = {}
        self.audio_ids = audio_ids
        if isinstance(self.audio_ids, str):
            audio_ids = []
            with open(self.audio_ids, "r") as reader:
                for line in reader.readlines():
                    audio_ids.append(line.strip())
            self.audio_ids = audio_ids

        for feat_type, filename in features.items():
            if filename is not None:
                self.aid_to_h5[feat_type] = load_dict_from_csv(
                    filename, ("audio_id", "hdf5_path"))
                if self.audio_ids is None:
                    self.audio_ids = list(self.aid_to_h5[feat_type].keys())
        if self.audio_ids is None:
            raise Exception("all provided feature csv is None")

        self.dataset_cache = {}
        first_audio_id = self.audio_ids[0]
        self.feat_dim = {}
        for feat_type in self.feat_types:
            if feat_type in self.aid_to_h5:
                self.feat_dim[feat_type] = read_from_h5(
                    first_audio_id, self.aid_to_h5[feat_type],
                    self.dataset_cache).shape[-1]

        self.load_into_mem = load_into_mem
        if self.load_into_mem:
            self.aid_to_feat = {feat_type: {} for feat_type in self.feat_types}
            # TODO load into memory following the original sequence in HDF5
            for audio_id in tqdm(self.audio_ids, ascii=True):
                for feat_type in self.feat_types:
                    feat = read_from_h5(audio_id,
                                        self.aid_to_h5[feat_type],
                                        self.dataset_cache)
                    self.aid_to_feat[feat_type][audio_id] = feat
        else:
            # ensure each process opens hdf5 after fork
            self.dataset_cache = {}

    
    def load_audio(self, audio_id):
        output = {}
        if self.load_into_mem:
            for feat_type in self.feat_types:
                output[feat_type] = self.aid_to_feat[feat_type][audio_id]
        else:
            for feat_type in self.feat_types:
                output[feat_type] = read_from_h5(
                    audio_id, self.aid_to_h5[feat_type], self.dataset_cache)
        return output


    def __getitem__(self, index):
        audio_id = self.audio_ids[index]
        output = self.load_audio(audio_id)

        for feat_type in self.feat_types:
            transform = self.transforms[feat_type]
            if transform is not None:
                for fn in transform:
                    output[feat_type] = fn(output[feat_type])
        output["audio_id"] = audio_id
        return output


    def __len__(self):
        return len(self.audio_ids)


    def __del__(self):
        for k, cache in self.dataset_cache.items():
            if cache:
                try:
                    cache.close()
                except:
                    pass


class CaptionDataset(InferenceDataset):

    def __init__(self,
                 features: Dict,
                 transforms: Dict,
                 caption: str,
                 vocabulary: str,
                 load_into_mem: bool = False):
        self.caption_info = json.load(open(caption))["audios"]
        self.vocabulary = pickle.load(open(vocabulary, "rb"))
        self.bos = self.vocabulary('<start>')
        self.eos = self.vocabulary('<end>')
        self.key_to_tokens = {}
        self.keys = []
        audio_ids = []
        for item in self.caption_info:
            audio_id = item["audio_id"]
            audio_ids.append(audio_id)
            self.key_to_tokens[audio_id] = {}
            for cap_item in item["captions"]:
                cap_id = str(cap_item["cap_id"])
                self.key_to_tokens[audio_id][cap_id] = cap_item["tokens"]
                self.keys.append((audio_id, cap_id))
        super().__init__(features, transforms, load_into_mem=load_into_mem,
            audio_ids=audio_ids)

    def __getitem__(self, index):
        audio_id, cap_id = self.keys[index]
        output = super().load_audio(audio_id)
        tokens = self.key_to_tokens[audio_id][cap_id].split()
        caption = [self.bos] + [self.vocabulary(token) for token in tokens] + \
            [self.eos]
        caption = np.array(caption)
        output["cap"] = caption
        output["audio_id"] = audio_id
        output["cap_id"] = cap_id
        return output

    def __len__(self):
        return len(self.keys)


class IterationBatchSampler(object):

    def __init__(self, data_source, batch_size, num_samples, shuffle=False):
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.shuffle = shuffle
        self.data_source = data_source
        self.index_queue = list(range(len(self.data_source)))
        if self.shuffle:
            np.random.shuffle(self.index_queue)
        self.pointer = 0

    def __iter__(self):
        batch = []
        num_samples = 0

        while num_samples < self.num_samples:

            indexes = []
            left = min(self.batch_size, self.num_samples - num_samples)

            for i in range(left):
                if self.pointer >= len(self.index_queue):
                    self.index_queue = list(range(len(self.data_source)))
                    self.pointer = 0
                    if self.shuffle:
                        np.random.shuffle(self.index_queue)
                indexes.append(self.index_queue[self.pointer])
                self.pointer += 1

            batch.append(indexes)
            num_samples += len(indexes)

        return iter(batch)

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size



def pad_sequence(data):
    if isinstance(data[0], np.ndarray):
        data = [torch.as_tensor(arr) for arr in data]
    padded_seq = torch.nn.utils.rnn.pad_sequence(data,
                                                 batch_first=True)
    length = [x.shape[0] for x in data]
    return padded_seq, length


def collate_fn(pad_keys=[], sort_key=None):

    def wrapper(data_batch):
        if sort_key:
            data_batch.sort(key=lambda x: len(x[sort_key]), reverse=True)
        
        output = {}
        for data in data_batch:
            for key in data:
                if key not in output:
                    output[key] = []
                output[key].append(data[key])

        for key in data_batch[0].keys():
            try:
                if key in pad_keys:
                    padded_seq, length = pad_sequence(output[key])
                    output[key] = padded_seq
                    output[f"{key}_len"] = np.array(length)
                else:
                    data = np.array(output[key])
                    if isinstance(output[key][0], np.ndarray):
                        output[key] = torch.as_tensor(data)
                    else:
                        output[key] = data
            except Exception:
                print(f"error occurred when collating {key}")
                import ipdb; ipdb.set_trace()

        return output

    return wrapper


if __name__ == "__main__":
    import sys
    from tqdm import tqdm
    from captioning.utils.build_vocab import Vocabulary
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    args = {
        "features": {
            "spec": "data/clotho_v2/dev/lms.csv",
            "fc": "data/clotho_v2/dev/clap_features/cnn14/fc.csv",
            "attn": "data/clotho_v2/dev/clap_features/cnn14/attn.csv"
        },
        "transforms": {
            "spec": None,
            "fc": None,
            "attn": None
        },
        "load_into_mem": False,
        "caption": "data/clotho_v2/dev/text.json",
        "vocabulary": "data/clotho_v2/dev/vocab.pkl"
    }
    dataset = CaptionDataset(**args)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        collate_fn=collate_fn(["spec", "attn", "cap"], "cap"),
        num_workers=4)
    idx = 0
    import time
    start = time.time()
    for batch in tqdm(dataloader, unit="batch"):
        idx += 1
        end = time.time()
        print("batch ", idx, "{:.3f} seconds".format(end - start))
        start = end
        pass


