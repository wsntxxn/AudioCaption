import warnings

import numpy as np
import torch

from captioning.utils.train_util import pad_sequence



class VarLenPadCollate:

    def __init__(self, pad_keys=[], sort_key=None):
        self.pad_keys = pad_keys
        self.sort_key = sort_key

    def __call__(self, data_batch):
        if self.sort_key:
            data_batch.sort(key=lambda x: len(x[self.sort_key]), reverse=True)
        
        output = {}
        for data in data_batch:
            for key in data:
                if key not in output:
                    output[key] = []
                output[key].append(data[key])

        for key in data_batch[0].keys():
            try:
                if key in self.pad_keys:
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


class TextCollate(VarLenPadCollate):

    def __init__(self, tokenizer, text_key="text", pad_keys=[], sort_key=None):
        super().__init__(pad_keys=pad_keys, sort_key=sort_key)
        self.tokenizer = tokenizer
        self.text_key = text_key

    def __call__(self, data_batch):
        if self.sort_key:
            data_batch.sort(key=lambda x: len(x[self.sort_key]), reverse=True)
        
        output = {}
        for data in data_batch:
            for key in data:
                if key not in output:
                    output[key] = []
                output[key].append(data[key])

        for key in data_batch[0].keys():
            try:
                if key == self.text_key:
                    output.update(self.tokenizer(output[key]))
                elif key in self.pad_keys:
                    padded_seq, length = pad_sequence(output[key])
                    output[key] = padded_seq
                    output[f"{key}_len"] = np.array(length)
                else:
                    if isinstance(output[key][0], (np.ndarray, torch.Tensor)) and \
                        output[key][0].ndim > 1:
                        warnings.warn(f"collating multi-dimensional {key} as a single tensor")
                    data = np.array(output[key])
                    if isinstance(output[key][0], np.ndarray):
                        output[key] = torch.as_tensor(data)
                    else:
                        output[key] = data
            except Exception:
                print(f"error occurred when collating {key}")
                import ipdb; ipdb.set_trace()
        return output
