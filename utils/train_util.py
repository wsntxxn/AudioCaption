# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import os
import sys
import logging
import yaml
import torch
import numpy as np
import pandas as pd
import sklearn.preprocessing as pre
from pprint import pformat

def genlogger(outputfile, level="INFO"):
    formatter = logging.Formatter(
        "[ %(levelname)s : %(asctime)s ] - %(message)s")
    logger = logging.getLogger(__name__ + "." + outputfile)
    logger.setLevel(getattr(logging, level))
    # Log results to std
    # stdhandler = logging.StreamHandler(sys.stdout)
    # stdhandler.setFormatter(formatter)
    # Dump log to file
    filehandler = logging.FileHandler(outputfile)
    filehandler.setFormatter(formatter)
    logger.addHandler(filehandler)
    # logger.addHandler(stdhandler)
    return logger

def pprint_dict(in_dict, outputfun=sys.stdout.write, formatter='yaml'):
    """pprint_dict

    :param outputfun: function to use, defaults to sys.stdout
    :param in_dict: dict to print
    """
    if formatter == 'yaml':
        format_fun = yaml.dump
    elif formatter == 'pretty':
        format_fun = pformat
    for line in format_fun(in_dict).split('\n'):
        outputfun(line)

def encode_labels(labels: pd.Series, encoder=None):
    """encode_labels

    Encodes labels

    :param labels: pd.Series representing the raw labels e.g., Speech, Water
    :param encoder (optional): Encoder already fitted 
    returns encoded labels (one hot) and the encoder
    """
    assert isinstance(labels, pd.Series), "Labels need to series"
    if not encoder:
        encoder = pre.LabelEncoder()
        encoder.fit(labels)
    labels_encoded = encoder.transform(labels)
    return labels_encoded.tolist(), encoder

def parse_config_or_kwargs(config_file, **kwargs):
    default_args = {
        "distributed": False,
        "swa": True,
        "swa_start": 21
    }
    with open(config_file) as con_reader:
        yaml_config = yaml.load(con_reader, Loader=yaml.FullLoader)
    # passed kwargs will override yaml config
    args = dict(yaml_config, **kwargs)
    for key, value in default_args.items():
        args.setdefault(key, value)
    return args

def store_yaml(config, config_file):
    with open(config_file, "w") as con_writer:
        yaml.dump(config, con_writer, default_flow_style=False)

def parse_augments(augment_list):
    """parse_augments
    parses the augmentation string in configuration file to corresponding methods

    :param augment_list: list
    """
    from captioning.datasets import augment

    specaug_kwargs = {"timemask": False, "freqmask": False, "timewarp": False}
    augments = []
    for transform in augment_list:
        if transform == "timemask":
            specaug_kwargs["timemask"] = True
        elif transform == "freqmask":
            specaug_kwargs["freqmask"] = True
        elif transform == "timewarp":
            specaug_kwargs["timewarp"] = True
        elif transform == "randomcrop":
            augments.append(augment.random_crop)
        elif transform == "timeroll":
            augments.append(augment.time_roll)
    augments.append(augment.spec_augment(**specaug_kwargs))
    return augments

def criterion_improver(mode):
    assert mode in ("loss", "acc", "score")
    best_value = np.inf if mode == "loss" else 0

    def comparator(x, best_x):
        return x < best_x if mode == "loss" else x > best_x

    def inner(x):
        nonlocal best_value

        if comparator(x, best_value):
            best_value = x
            return True
        return False
    return inner

def run_val(engine, evaluator, dataloader):
    evaluator.run(dataloader)

# def generate_length_mask(lens):
    # lens = torch.as_tensor(lens)
    # N = lens.size(0)
    # T = max(lens)
    # idxs = torch.arange(T).repeat(N).view(N, T)
    # mask = (idxs < lens.view(-1, 1))
    # return mask

# def mean_with_lens(features, lens):
    # """
    # features: [N, T, ...] (assume the second dimension represents length)
    # lens: [N,]
    # """
    # lens = torch.as_tensor(lens)
    # mask = generate_length_mask(lens).to(features.device) # [N, T]

    # feature_mean = features * mask.unsqueeze(-1)
    # feature_mean = feature_mean.sum(1) / lens.unsqueeze(1).to(features.device)
    # return feature_mean

# def max_with_lens(features, lens):
    # """
    # features: [N, T, ...] (assume the second dimension represents length)
    # lens: [N,]
    # """
    # lens = torch.as_tensor(lens)
    # mask = generate_length_mask(lens).to(features.device) # [N, T]

    # feature_max = features.clone()
    # feature_max[~mask] = float("-inf")
    # feature_max, _ = feature_max.max(1)
    # return feature_max

def fix_batchnorm(model: torch.nn.Module):
    # classname = model.__class__.__name__
    # if classname.find("BatchNorm") != -1:
        # model.eval()
    def inner(module):
        class_name = module.__class__.__name__
        if class_name.find("BatchNorm") != -1:
            module.eval()
    model.apply(inner)

def load_pretrained_model(model: torch.nn.Module, pretrained, outputfun):
    if not os.path.exists(pretrained):
        outputfun(f"Loading pretrained model from {pretrained} failed!")
        return
    state_dict = torch.load(pretrained, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
    model_dict = model.state_dict()
    pretrained_dict = {
        k: v for k, v in state_dict.items() if (k in model_dict) and (
            model_dict[k].shape == v.shape)
    }
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=True)


