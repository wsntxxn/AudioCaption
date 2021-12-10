# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import os
import sys
import logging
import yaml
import torch
from torch.optim.swa_utils import AveragedModel as torch_average_model
import numpy as np
import pandas as pd
import sklearn.preprocessing as pre
from pprint import pformat


def load_dict_from_csv(csv, cols):
    df = pd.read_csv(csv, sep="\t")
    output = dict(zip(df[cols[0]], df[cols[1]]))
    return output

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
        "swa_start": 21,
        "sampler_args": {"max_cap_num": None},
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
