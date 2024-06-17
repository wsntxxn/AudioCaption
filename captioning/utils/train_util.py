# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import os
import sys
import logging
import random
from typing import Callable, Dict, Union
import importlib
import yaml
import toml
import torch
from torch.optim.swa_utils import AveragedModel as torch_average_model
import numpy as np
import pandas as pd
from pprint import pformat
import h5py


def load_dict_from_csv(csv, cols):
    df = pd.read_csv(csv, sep="\t")
    output = dict(zip(df[cols[0]], df[cols[1]]))
    return output

def pad_sequence(data, pad_value=0):
    if isinstance(data[0], (np.ndarray, torch.Tensor)):
        data = [torch.as_tensor(arr) for arr in data]
    padded_seq = torch.nn.utils.rnn.pad_sequence(data,
                                                 batch_first=True,
                                                 padding_value=pad_value)
    length = np.array([x.shape[0] for x in data])
    return padded_seq, length

def init_logger(filename, level="INFO"):
    formatter = logging.Formatter(
        "[ %(levelname)s : %(asctime)s ] - %(message)s")
    logger = logging.getLogger(__name__ + "." + filename)
    logger.setLevel(getattr(logging, level))
    # Log results to std
    # stdhandler = logging.StreamHandler(sys.stdout)
    # stdhandler.setFormatter(formatter)
    # Dump log to file
    filehandler = logging.FileHandler(filename)
    filehandler.setFormatter(formatter)
    logger.addHandler(filehandler)
    # logger.addHandler(stdhandler)
    return logger

def init_obj(module, config, **kwargs):
    obj_args = config["args"].copy()
    obj_args.update(kwargs)
    return getattr(module, config["type"])(**obj_args)

def read_from_h5(key: str, key_to_h5: Dict, cache: Dict):
    hdf5_path = key_to_h5[key]
    if hdf5_path not in cache:
        cache[hdf5_path] = h5py.File(hdf5_path, "r")
    try:
        return cache[hdf5_path][key][()]
    except KeyError: # audiocaps compatibility
        key = "Y" + key + ".wav"
        return cache[hdf5_path][key][()]

def get_cls_from_str(string, reload=False):
    module_name, cls_name = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module_name)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module_name, package=None), cls_name)

def init_obj_from_dict(config, **kwargs):
    obj_args = config["args"].copy()
    obj_args.update(kwargs)
    for k in config:
        if k not in ["type", "args"] and isinstance(config[k], dict) and k not in kwargs:
            obj_args[k] = init_obj_from_dict(config[k])
    try:
        obj = get_cls_from_str(config["type"])(**obj_args)
        return obj
    except Exception as e:
        print(f"Initializing {config} failed, detailed error stack: ")
        raise e

def init_model_from_config(config, print_fn=sys.stdout.write):
    kwargs = {}
    for k in config:
        if k not in ["type", "args", "pretrained"]:
            sub_model = init_model_from_config(config[k], print_fn)
            if "pretrained" in config[k]:
                load_pretrained_model(sub_model,
                                      config[k]["pretrained"],
                                      print_fn)
            kwargs[k] = sub_model
    model = init_obj_from_dict(config, **kwargs)
    return model

def pprint_dict(in_dict, print_fn=sys.stdout.write, formatter='yaml'):
    """pprint_dict

    :param outputfun: function to use, defaults to sys.stdout
    :param in_dict: dict to print
    """
    if formatter == 'yaml':
        format_fn = yaml.dump
    elif formatter == 'pretty':
        format_fn = pformat
    else:
        raise NotImplementedError
    for line in format_fn(in_dict).split('\n'):
        print_fn(line)

def merge_a_into_b(a, b):
    # merge dict a into dict b. values in a will overwrite b.
    for k, v in a.items():
        if isinstance(v, dict) and k in b:
            assert isinstance(
                b[k], dict
            ), "Cannot inherit key '{}' from base!".format(k)
            merge_a_into_b(v, b[k])
        else:
            b[k] = v

def load_config(config_file):
    with open(config_file, "r") as reader:
        config = yaml.load(reader, Loader=yaml.FullLoader)
    if "inherit_from" in config:
        base_config_file = config["inherit_from"]
        base_config_file = os.path.join(
            os.path.dirname(config_file), base_config_file
        )
        assert not os.path.samefile(config_file, base_config_file), \
            "inherit from itself"
        base_config = load_config(base_config_file)
        del config["inherit_from"]
        merge_a_into_b(config, base_config)
        return base_config
    return config

def parse_config_or_kwargs(config_file, **kwargs):
    toml_list = []
    for k, v in kwargs.items():
        if isinstance(v, str):
            toml_list.append(f"{k}='{v}'")
        elif isinstance(v, bool):
            toml_list.append(f"{k}={str(v).lower()}")
        else:
            toml_list.append(f"{k}={v}")
    toml_str = "\n".join(toml_list)
    cmd_config = toml.loads(toml_str)
    yaml_config = load_config(config_file)
    merge_a_into_b(cmd_config, yaml_config)
    return yaml_config

def store_yaml(config, config_file):
    with open(config_file, "w") as con_writer:
        yaml.dump(config, con_writer, indent=4, default_flow_style=False)


class MetricImprover:

    def __init__(self, mode):
        assert mode in ("min", "max")
        self.mode = mode
        # min: lower -> better; max: higher -> better
        self.best_value = np.inf if mode == "min" else -np.inf

    def compare(self, x, best_x):
        return x < best_x if self.mode == "min" else x > best_x

    def __call__(self, x):
        if self.compare(x, self.best_value):
            self.best_value = x
            return True
        return False

    def state_dict(self):
        return self.__dict__

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

def fix_batchnorm(model: torch.nn.Module):
    def inner(module):
        class_name = module.__class__.__name__
        if class_name.find("BatchNorm") != -1:
            module.eval()
    model.apply(inner)

def merge_load_state_dict(state_dict,
                          model: torch.nn.Module,
                          output_fn: Callable = sys.stdout.write):
    model_dict = model.state_dict()
    pretrained_dict = {}
    mismatch_keys = []
    for key, value in state_dict.items():
        if key in model_dict and model_dict[key].shape == value.shape:
            pretrained_dict[key] = value
        else:
            mismatch_keys.append(key)
    output_fn(f"Loading pre-trained model, with mismatched keys {mismatch_keys}")
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=True)
    return pretrained_dict.keys()

def load_pretrained_model(model: torch.nn.Module,
                          pretrained: Union[str, Dict],
                          output_fn: Callable = sys.stdout.write):
    if not isinstance(pretrained, dict) and not os.path.exists(pretrained):
        output_fn(f"pretrained {pretrained} not exist!")
        return
    
    if hasattr(model, "load_pretrained"):
        model.load_pretrained(pretrained, output_fn)
        return

    if isinstance(pretrained, dict):
        state_dict = pretrained
    else:
        state_dict = torch.load(pretrained, map_location="cpu")

    if "model" in state_dict:
        state_dict = state_dict["model"]
    
    merge_load_state_dict(state_dict, model, output_fn)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AveragedModel(torch_average_model):

    def update_parameters(self, model):
        for p_swa, p_model in zip(self.parameters(), model.parameters()):
            device = p_swa.device
            p_model_ = p_model.detach().to(device)
            if self.n_averaged == 0:
                p_swa.detach().copy_(p_model_)
            else:
                p_swa.detach().copy_(self.avg_fn(p_swa.detach(), p_model_,
                                                 self.n_averaged.to(device)))

        for b_swa, b_model in zip(list(self.buffers())[1:], model.buffers()):
            device = b_swa.device
            b_model_ = b_model.detach().to(device)
            if self.n_averaged == 0:
                b_swa.detach().copy_(b_model_)
            else:
                b_swa.detach().copy_(self.avg_fn(b_swa.detach(), b_model_,
                                                 self.n_averaged.to(device)))
        self.n_averaged += 1
