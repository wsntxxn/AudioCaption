import os

import torch
import torch.nn as nn
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.model import MBConvBlock
from efficientnet_pytorch import utils as efficientnet_utils
from efficientnet_pytorch.utils import (
    round_filters,
    round_repeats,
    get_same_padding_conv2d,
    calculate_output_image_size,
    MemoryEfficientSwish,
)
from einops import rearrange, reduce
from torch.hub import load_state_dict_from_url


model_dir = "/hpc_stor03/sjtu_home/xuenan.xu/workspace/audio_captioning/experiments/pretrained_encoder/"


class _EffiNet(nn.Module):
    """A proxy for efficient net models"""
    def __init__(self,
                 blocks_args=None,
                 global_params=None,
                 prune_start_layer: int = 0,
                 prune_se: bool = True,
                 prune_ratio: float = 0.0
                 ) -> None:
        super().__init__()
        if prune_ratio > 0:
            self.eff_net = EfficientNetB2Pruned(blocks_args=blocks_args,
                                                global_params=global_params,
                                                prune_start_layer=prune_start_layer,
                                                prune_se=prune_se,
                                                prune_ratio=prune_ratio)
        else:
            self.eff_net = EfficientNet(blocks_args=blocks_args,
                                        global_params=global_params)
        

    def forward(self, x: torch.Tensor): 
        x = rearrange(x, 'b f t -> b 1 f t')
        x = self.eff_net.extract_features(x)
        return reduce(x, 'b c f t -> b t c', 'mean')


def get_model(pretrained=True) -> _EffiNet:
    blocks_args, global_params = efficientnet_utils.get_model_params(
        'efficientnet-b2', {'include_top': False})
    model = _EffiNet(blocks_args=blocks_args,
                     global_params=global_params)
    model.eff_net._change_in_channels(1)
    if pretrained:
        model_path = os.path.join(model_dir, "effb2.pt")
        if not os.path.exists(model_path):
            state_dict = load_state_dict_from_url(
                'https://github.com/richermans/HEAR2021_EfficientLatent/releases/download/v0.0.1/effb2.pt',
                progress=True,
                model_dir=model_dir)
        else:
            state_dict = torch.load(model_path)
        del_keys = [key for key in state_dict if key.startswith("front_end")]
        for key in del_keys:
            del state_dict[key]
        model.eff_net.load_state_dict(state_dict)
    return model


class MBConvBlockPruned(MBConvBlock):

    def __init__(self, block_args, global_params, image_size=None, prune_ratio=0.5, prune_se=True):
        super(MBConvBlock, self).__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum  # pytorch's difference from tensorflow
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # whether to use skip connection and drop connect

        # Expansion phase (Inverted Bottleneck)
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            oup = int(oup * (1 - prune_ratio))
            Conv2d = get_same_padding_conv2d(image_size=image_size)
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
            # image_size = calculate_output_image_size(image_size, 1) <-- this wouldn't modify image_size

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        image_size = calculate_output_image_size(image_size, s)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            Conv2d = get_same_padding_conv2d(image_size=(1, 1))
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            if prune_se:
                num_squeezed_channels = int(num_squeezed_channels * (1 - prune_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Pointwise convolution phase
        final_oup = self._block_args.output_filters
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()


class EfficientNetB2Pruned(EfficientNet):

    def __init__(self, blocks_args=None, global_params=None,
                 prune_start_layer=0, prune_ratio=0.5, prune_se=True):
        super(EfficientNet, self).__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Get stem static or dynamic convolution depending on image size
        image_size = global_params.image_size
        Conv2d = get_same_padding_conv2d(image_size=image_size)

        n_build_blks = 0
        # Stem
        in_channels = 1  # spectrogram

        p = 0.0 if n_build_blks < prune_start_layer else prune_ratio
        out_channels = round_filters(32 * (1 - p),
                                     self._global_params)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        image_size = calculate_output_image_size(image_size, 2)
        n_build_blks += 1

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            p = 0.0 if n_build_blks < prune_start_layer else prune_ratio
            orig_input_filters = block_args.input_filters
            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(
                    block_args.input_filters * (1 - p),
                    self._global_params),
                output_filters=round_filters(
                    block_args.output_filters * (1 - p),
                    self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            if n_build_blks == prune_start_layer:
                block_args = block_args._replace(input_filters=round_filters(
                    orig_input_filters,
                    self._global_params)
                )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlockPruned(block_args, self._global_params,
                                                  image_size=image_size, prune_ratio=p,
                                                  prune_se=prune_se))
            n_build_blks += 1

            image_size = calculate_output_image_size(image_size, block_args.stride)
            if block_args.num_repeat > 1:  # modify block_args to keep same output size
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlockPruned(block_args,
                                                      self._global_params,
                                                      image_size=image_size,
                                                      prune_ratio=p,
                                                      prune_se=prune_se))
                # image_size = calculate_output_image_size(image_size, block_args.stride)  # stride = 1

        # Head
        in_channels = block_args.output_filters  # output of final block
        p = 0.0 if n_build_blks < prune_start_layer else prune_ratio
        out_channels = round_filters(1280 * (1 - p), self._global_params)
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        if self._global_params.include_top:
            self._dropout = nn.Dropout(self._global_params.dropout_rate)
            self._fc = nn.Linear(out_channels, self._global_params.num_classes)

        # set activation to memory efficient swish by default
        self._swish = MemoryEfficientSwish()


def get_pruned_model(pretrained: bool = True,
                     prune_ratio: float = 0.5,
                     prune_start_layer: int = 0,
                     prune_se: bool = True,
                     prune_method: str = "operator_norm") -> _EffiNet:
    
    import captioning.models.conv_filter_pruning as pruning_lib

    blocks_args, global_params = efficientnet_utils.get_model_params(
        'efficientnet-b2', {'include_top': False})
    # print("num blocks: ", len(blocks_args))
    # print("block args: ")
    # for block_arg in blocks_args:
    #     print(block_arg)
    model = _EffiNet(blocks_args=blocks_args,
                     global_params=global_params,
                     prune_start_layer=prune_start_layer,
                     prune_se=prune_se,
                     prune_ratio=prune_ratio)
    
    if prune_method == "operator_norm":
        filter_pruning = pruning_lib.operator_norm_pruning
    elif prune_method == "interspeech":
        filter_pruning = pruning_lib.cs_interspeech
    elif prune_method == "iclr_l1":
        filter_pruning = pruning_lib.iclr_l1
    elif prune_method == "iclr_gm":
        filter_pruning = pruning_lib.iclr_gm
    elif prune_method == "cs_waspaa":
        filter_pruning = pruning_lib.cs_waspaa
        

    if isinstance(pretrained, str):
        ckpt = torch.load(pretrained, "cpu")
        state_dict = {}
        for key in ckpt["model"].keys():
            if key.startswith("model.encoder.backbone"):
                state_dict[key[len("model.encoder.backbone.eff_net."):]] = ckpt["model"][key]
    elif isinstance(pretrained, bool):
        model_path = os.path.join(model_dir, "effb2.pt")
        if not os.path.exists(model_path):
            state_dict = load_state_dict_from_url(
                'https://github.com/richermans/HEAR2021_EfficientLatent/releases/download/v0.0.1/effb2.pt',
                progress=True,
                model_dir=model_dir)
        else:
            state_dict = torch.load(model_path)
        del_keys = [key for key in state_dict if key.startswith("front_end")]
        for key in del_keys:
            del state_dict[key]
    
    # load pretrained model with corresponding filters
    # rule:
    # * depthwise_conv: in_ch_idx = out_ch_idx = prev_conv_idx
    mod_dep_path = [
        "_conv_stem",
    ]
    conv_to_bn = {"_conv_stem": "_bn0"}
    for i in range(2):
        mod_dep_path.extend([
            f"_blocks.{i}._depthwise_conv",
            f"_blocks.{i}._se_reduce",
            f"_blocks.{i}._se_expand",
            f"_blocks.{i}._project_conv", 
        ])
        conv_to_bn[f"_blocks.{i}._depthwise_conv"] = f"_blocks.{i}._bn1"
        conv_to_bn[f"_blocks.{i}._project_conv"] = f"_blocks.{i}._bn2"
    
    for i in range(2, 23):
        mod_dep_path.extend([
            f"_blocks.{i}._expand_conv",
            f"_blocks.{i}._depthwise_conv", 
            f"_blocks.{i}._se_reduce",
            f"_blocks.{i}._se_expand",
            f"_blocks.{i}._project_conv"
        ])
        conv_to_bn[f"_blocks.{i}._expand_conv"] = f"_blocks.{i}._bn0"
        conv_to_bn[f"_blocks.{i}._depthwise_conv"] = f"_blocks.{i}._bn1"
        conv_to_bn[f"_blocks.{i}._project_conv"] = f"_blocks.{i}._bn2"

    mod_dep_path.append("_conv_head") 
    conv_to_bn["_conv_head"] = "_bn1"

    # print(mod_dep_path)
    # print(conv_to_bn)
    
    key_to_w_b_idx = {}
    model_dict = model.eff_net.state_dict()
    for conv_key in tqdm(mod_dep_path):
        weight = state_dict[f"{conv_key}.weight"]
        ptr_n_filter = weight.size(0)
        model_n_filter = model_dict[f"{conv_key}.weight"].size(0)
        if model_n_filter < ptr_n_filter:
            key_to_w_b_idx[conv_key] = filter_pruning(weight.numpy())[:model_n_filter]
        else:
            key_to_w_b_idx[conv_key] = slice(None)

    pruned_state_dict = {}
    for conv_key, prev_conv_key in zip(mod_dep_path, [None] + mod_dep_path[:-1]):
    
        for sub_key in ["weight", "bias"]: # adjust the conv layer
            cur_key = f"{conv_key}.{sub_key}"

            if cur_key not in state_dict:
                continue

            if prev_conv_key is None or conv_key.endswith("_depthwise_conv"):
                conv_in_idx = slice(None)
            else:
                conv_in_idx = key_to_w_b_idx[prev_conv_key]

            # the first pruned layer
            if model_dict[cur_key].ndim > 1 and model_dict[cur_key].size(1) == state_dict[cur_key].size(1):
                conv_in_idx = slice(None)
            
            if conv_key.endswith("_depthwise_conv"):
                conv_out_idx = key_to_w_b_idx[prev_conv_key]
            else:
                conv_out_idx = key_to_w_b_idx[conv_key]
            
            # if conv_key == "_blocks.16._se_reduce":
            #     print(len(conv_out_idx), len(conv_in_idx))

            if sub_key == "weight":
                pruned_state_dict[cur_key] = state_dict[cur_key][
                    conv_out_idx, ...][:, conv_in_idx, ...]
            else:
                pruned_state_dict[cur_key] = state_dict[cur_key][
                    conv_out_idx, ...]

        if conv_key in conv_to_bn: # adjust the corresponding bn layer
            for sub_key in ["weight", "bias", "running_mean", "running_var"]:
                cur_key = f"{conv_to_bn[conv_key]}.{sub_key}"
                if cur_key not in state_dict:
                    continue
                pruned_state_dict[cur_key] = state_dict[cur_key][
                    key_to_w_b_idx[conv_key], ...]
    
    model.eff_net.load_state_dict(pruned_state_dict)

    return model
