from functools import partial
import math
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union, Dict
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision.ops.misc import ConvNormActivation
from torch.hub import load_state_dict_from_url
import urllib.parse


def NAME_TO_WIDTH(name):
    mn_map = {
        'mn01': 0.1,
        'mn02': 0.2,
        'mn04': 0.4,
        'mn05': 0.5,
        'mn06': 0.6,
        'mn08': 0.8,
        'mn10': 1.0,
        'mn12': 1.2,
        'mn14': 1.4,
        'mn16': 1.6,
        'mn20': 2.0,
        'mn30': 3.0,
        'mn40': 4.0,
    }

    dymn_map = {
        'dymn04': 0.4,
        'dymn10': 1.0,
        'dymn20': 2.0
    }

    try:
        if name.startswith('dymn'):
            w = dymn_map[name[:6]]
        else:
            w = mn_map[name[:4]]
    except:
        w = 1.0

    return w


def cnn_out_size(in_size, padding, dilation, kernel, stride):
    s = in_size + 2 * padding - dilation * (kernel - 1) - 1
    return math.floor(s / stride + 1)


def make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def collapse_dim(x: Tensor, dim: int, mode: str = "pool", pool_fn:  Callable[[Tensor, int], Tensor] = torch.mean,
                 combine_dim: int = None):
    """
    Collapses dimension of multi-dimensional tensor by pooling or combining dimensions
    :param x: input Tensor
    :param dim: dimension to collapse
    :param mode: 'pool' or 'combine'
    :param pool_fn: function to be applied in case of pooling
    :param combine_dim: dimension to join 'dim' to
    :return: collapsed tensor
    """
    if mode == "pool":
        return pool_fn(x, dim)
    elif mode == "combine":
        s = list(x.size())
        s[combine_dim] *= dim
        s[dim] //= dim
        return x.view(s)


class MultiHeadAttentionPooling(nn.Module):
    """Multi-Head Attention as used in PSLA paper (https://arxiv.org/pdf/2102.01243.pdf)
    """
    def __init__(self, in_dim, out_dim, att_activation: str = 'sigmoid',
                 clf_activation: str = 'ident', num_heads: int = 4, epsilon: float = 1e-7):
        super(MultiHeadAttentionPooling, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.epsilon = epsilon

        self.att_activation = att_activation
        self.clf_activation = clf_activation

        # out size: out dim x 2 (att and clf paths) x num_heads
        self.subspace_proj = nn.Linear(self.in_dim, self.out_dim * 2 * self.num_heads)
        self.head_weight = nn.Parameter(torch.tensor([1.0 / self.num_heads] * self.num_heads).view(1, -1, 1))

    def activate(self, x, activation):
        if activation == 'linear':
            return x
        elif activation == 'relu':
            return F.relu(x)
        elif activation == 'sigmoid':
            return torch.sigmoid(x)
        elif activation == 'softmax':
            return F.softmax(x, dim=1)
        elif activation == 'ident':
            return x

    def forward(self, x) -> Tensor:
        """x: Tensor of size (batch_size, channels, frequency bands, sequence length)
        """
        x = collapse_dim(x, dim=2)  # results in tensor of size (batch_size, channels, sequence_length)
        x = x.transpose(1, 2)  # results in tensor of size (batch_size, sequence_length, channels)
        b, n, c = x.shape

        x = self.subspace_proj(x).reshape(b, n, 2, self.num_heads, self.out_dim).permute(2, 0, 3, 1, 4)
        att, val = x[0], x[1]
        val = self.activate(val, self.clf_activation)
        att = self.activate(att, self.att_activation)
        att = torch.clamp(att, self.epsilon, 1. - self.epsilon)
        att = att / torch.sum(att, dim=2, keepdim=True)

        out = torch.sum(att * val, dim=2) * self.head_weight
        out = torch.sum(out, dim=1)
        return out


class ConcurrentSEBlock(torch.nn.Module):
    def __init__(
        self,
        c_dim: int,
        f_dim: int,
        t_dim: int,
        se_cnf: Dict
    ) -> None:
        super().__init__()
        dims = [c_dim, f_dim, t_dim]
        self.conc_se_layers = nn.ModuleList()
        for d in se_cnf['se_dims']:
            input_dim = dims[d-1]
            squeeze_dim = make_divisible(input_dim // se_cnf['se_r'], 8)
            self.conc_se_layers.append(SqueezeExcitation(input_dim, squeeze_dim, d))
        if se_cnf['se_agg'] == "max":
            self.agg_op = lambda x: torch.max(x, dim=0)[0]
        elif se_cnf['se_agg'] == "avg":
            self.agg_op = lambda x: torch.mean(x, dim=0)
        elif se_cnf['se_agg'] == "add":
            self.agg_op = lambda x: torch.sum(x, dim=0)
        elif se_cnf['se_agg'] == "min":
            self.agg_op = lambda x: torch.min(x, dim=0)[0]
        else:
            raise NotImplementedError(f"SE aggregation operation '{self.agg_op}' not implemented")

    def forward(self, input: Tensor) -> Tensor:
        # apply all concurrent se layers
        se_outs = []
        for se_layer in self.conc_se_layers:
            se_outs.append(se_layer(input))
        out = self.agg_op(torch.stack(se_outs, dim=0))
        return out


class SqueezeExcitation(torch.nn.Module):
    """
    This block implements the Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507.
    Args:
        input_dim (int): Input dimension
        squeeze_dim (int): Size of Bottleneck
        activation (Callable): activation applied to bottleneck
        scale_activation (Callable): activation applied to the output
    """

    def __init__(
        self,
        input_dim: int,
        squeeze_dim: int,
        se_dim: int,
        activation: Callable[..., torch.nn.Module] = torch.nn.ReLU,
        scale_activation: Callable[..., torch.nn.Module] = torch.nn.Sigmoid,
    ) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, squeeze_dim)
        self.fc2 = torch.nn.Linear(squeeze_dim, input_dim)
        assert se_dim in [1, 2, 3]
        self.se_dim = [1, 2, 3]
        self.se_dim.remove(se_dim)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, input: Tensor) -> Tensor:
        scale = torch.mean(input, self.se_dim, keepdim=True)
        shape = scale.size()
        scale = self.fc1(scale.squeeze(2).squeeze(2))
        scale = self.activation(scale)
        scale = self.fc2(scale)
        scale = scale
        return self.scale_activation(scale).view(shape)

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input)
        return scale * input


class InvertedResidualConfig:
    # Stores information listed at Tables 1 and 2 of the MobileNetV3 paper
    def __init__(
        self,
        input_channels: int,
        kernel: int,
        expanded_channels: int,
        out_channels: int,
        use_se: bool,
        activation: str,
        stride: int,
        dilation: int,
        width_mult: float,
    ):
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride
        self.dilation = dilation
        self.f_dim = None
        self.t_dim = None

    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        return make_divisible(channels * width_mult, 8)

    def out_size(self, in_size):
        padding = (self.kernel - 1) // 2 * self.dilation
        return cnn_out_size(in_size, padding, self.dilation, self.kernel, self.stride)


class InvertedResidual(nn.Module):
    def __init__(
        self,
        cnf: InvertedResidualConfig,
        se_cnf: Dict,
        norm_layer: Callable[..., nn.Module],
        depthwise_norm_layer: Callable[..., nn.Module]
    ):
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        # expand
        if cnf.expanded_channels != cnf.input_channels:
            layers.append(
                ConvNormActivation(
                    cnf.input_channels,
                    cnf.expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # depthwise
        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(
            ConvNormActivation(
                cnf.expanded_channels,
                cnf.expanded_channels,
                kernel_size=cnf.kernel,
                stride=stride,
                dilation=cnf.dilation,
                groups=cnf.expanded_channels,
                norm_layer=depthwise_norm_layer,
                activation_layer=activation_layer,
            )
        )
        if cnf.use_se and se_cnf['se_dims'] is not None:
            layers.append(ConcurrentSEBlock(cnf.expanded_channels, cnf.f_dim, cnf.t_dim, se_cnf))

        # project
        layers.append(
            ConvNormActivation(
                cnf.expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None
            )
        )

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels
        self._is_cn = cnf.stride > 1

    def forward(self, inp: Tensor) -> Tensor:
        result = self.block(inp)
        if self.use_res_connect:
            result += inp
        return result


# Adapted version of MobileNetV3 pytorch implementation
# https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py

# points to github releases
model_url = "https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/"
# folder to store downloaded models to
model_dir = "/mnt/fast/nobackup/scratch4weeks/xx00336/workspace/audio_caption_xnx/experiments/pretrained_models/eff_at"


pretrained_models = {
    # pytorch ImageNet pre-trained model
    # own ImageNet pre-trained models will follow
    # NOTE: for easy loading we provide the adapted state dict ready for AudioSet training (1 input channel,
    # 527 output classes)
    # NOTE: the classifier is just a random initialization, feature extractor (conv layers) is pre-trained
    "mn10_im_pytorch": urllib.parse.urljoin(model_url, "mn10_im_pytorch.pt"),
    # self-trained models on ImageNet
    "mn01_im": urllib.parse.urljoin(model_url, "mn01_im.pt"),
    "mn02_im": urllib.parse.urljoin(model_url, "mn02_im.pt"),
    "mn04_im": urllib.parse.urljoin(model_url, "mn04_im.pt"),
    "mn05_im": urllib.parse.urljoin(model_url, "mn05_im.pt"),
    "mn10_im": urllib.parse.urljoin(model_url, "mn10_im.pt"),
    "mn20_im": urllib.parse.urljoin(model_url, "mn20_im.pt"),
    "mn30_im": urllib.parse.urljoin(model_url, "mn30_im.pt"),
    "mn40_im": urllib.parse.urljoin(model_url, "mn40_im.pt"),
    # Models trained on AudioSet
    "mn01_as": urllib.parse.urljoin(model_url, "mn01_as_mAP_298.pt"),
    "mn02_as": urllib.parse.urljoin(model_url, "mn02_as_mAP_378.pt"),
    "mn04_as": urllib.parse.urljoin(model_url, "mn04_as_mAP_432.pt"),
    "mn05_as": urllib.parse.urljoin(model_url, "mn05_as_mAP_443.pt"),
    "mn10_as": urllib.parse.urljoin(model_url, "mn10_as_mAP_471.pt"),
    "mn20_as": urllib.parse.urljoin(model_url, "mn20_as_mAP_478.pt"),
    "mn30_as": urllib.parse.urljoin(model_url, "mn30_as_mAP_482.pt"),
    "mn40_as": urllib.parse.urljoin(model_url, "mn40_as_mAP_484.pt"),
    "mn40_as(2)": urllib.parse.urljoin(model_url, "mn40_as_mAP_483.pt"),
    "mn40_as(3)": urllib.parse.urljoin(model_url, "mn40_as_mAP_483(2).pt"),
    "mn40_as_no_im_pre": urllib.parse.urljoin(model_url, "mn40_as_no_im_pre_mAP_483.pt"),
    "mn40_as_no_im_pre(2)": urllib.parse.urljoin(model_url, "mn40_as_no_im_pre_mAP_483(2).pt"),
    "mn40_as_no_im_pre(3)": urllib.parse.urljoin(model_url, "mn40_as_no_im_pre_mAP_482.pt"),
    "mn40_as_ext": urllib.parse.urljoin(model_url, "mn40_as_ext_mAP_487.pt"),
    "mn40_as_ext(2)": urllib.parse.urljoin(model_url, "mn40_as_ext_mAP_486.pt"),
    "mn40_as_ext(3)": urllib.parse.urljoin(model_url, "mn40_as_ext_mAP_485.pt"),
    # varying hop size (time resolution)
    "mn10_as_hop_5": urllib.parse.urljoin(model_url, "mn10_as_hop_5_mAP_475.pt"),
    "mn10_as_hop_15": urllib.parse.urljoin(model_url, "mn10_as_hop_15_mAP_463.pt"),
    "mn10_as_hop_20": urllib.parse.urljoin(model_url, "mn10_as_hop_20_mAP_456.pt"),
    "mn10_as_hop_25": urllib.parse.urljoin(model_url, "mn10_as_hop_25_mAP_447.pt"),
    # varying n_mels (frequency resolution)
    "mn10_as_mels_40": urllib.parse.urljoin(model_url, "mn10_as_mels_40_mAP_453.pt"),
    "mn10_as_mels_64": urllib.parse.urljoin(model_url, "mn10_as_mels_64_mAP_461.pt"),
    "mn10_as_mels_256": urllib.parse.urljoin(model_url, "mn10_as_mels_256_mAP_474.pt"),
    # fully-convolutional head
    "mn10_as_fc": urllib.parse.urljoin(model_url, "mn10_as_fc_mAP_465.pt"),
    "mn10_as_fc_s2221": urllib.parse.urljoin(model_url, "mn10_as_fc_s2221_mAP_466.pt"),
    "mn10_as_fc_s2211": urllib.parse.urljoin(model_url, "mn10_as_fc_s2211_mAP_466.pt"),
}


class MN(nn.Module):
    def __init__(
        self,
        inverted_residual_setting: List[InvertedResidualConfig],
        last_channel: int,
        num_classes: int = 1000,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.2,
        in_conv_kernel: int = 3,
        in_conv_stride: int = 2,
        in_channels: int = 1,
        **kwargs: Any,
    ) -> None:
        """
        MobileNet V3 main class

        Args:
            inverted_residual_setting (List[InvertedResidualConfig]): Network structure
            last_channel (int): The number of channels on the penultimate layer
            num_classes (int): Number of classes
            block (Optional[Callable[..., nn.Module]]): Module specifying inverted residual building block for models
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
            dropout (float): The droupout probability
            in_conv_kernel (int): Size of kernel for first convolution
            in_conv_stride (int): Size of stride for first convolution
            in_channels (int): Number of input channels
        """
        super(MN, self).__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])
        ):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        if block is None:
            block = InvertedResidual

        depthwise_norm_layer = norm_layer = \
            norm_layer if norm_layer is not None else partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers: List[nn.Module] = []

        kernel_sizes = [in_conv_kernel]
        strides = [in_conv_stride]

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            ConvNormActivation(
                in_channels,
                firstconv_output_channels,
                kernel_size=in_conv_kernel,
                stride=in_conv_stride,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
            )
        )

        # get squeeze excitation config
        se_cnf = kwargs.get('se_conf', None)

        # building inverted residual blocks
        # - keep track of size of frequency and time dimensions for possible application of Squeeze-and-Excitation
        # on the frequency/time dimension
        # - applying Squeeze-and-Excitation on the time dimension is not recommended as this constrains the network to
        # a particular length of the audio clip, whereas Squeeze-and-Excitation on the frequency bands is fine,
        # as the number of frequency bands is usually not changing
        f_dim, t_dim = kwargs.get('input_dims', (128, 1000))
        # take into account first conv layer
        f_dim = cnn_out_size(f_dim, 1, 1, 3, 2)
        t_dim = cnn_out_size(t_dim, 1, 1, 3, 2)
        for cnf in inverted_residual_setting:
            f_dim = cnf.out_size(f_dim)
            t_dim = cnf.out_size(t_dim)
            cnf.f_dim, cnf.t_dim = f_dim, t_dim  # update dimensions in block config
            layers.append(block(cnf, se_cnf, norm_layer, depthwise_norm_layer))
            kernel_sizes.append(cnf.kernel)
            strides.append(cnf.stride)

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(
            ConvNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
            )
        )

        self.features = nn.Sequential(*layers)
        self.head_type = kwargs.get("head_type", False)
        if self.head_type == "multihead_attention_pooling":
            self.classifier = MultiHeadAttentionPooling(lastconv_output_channels, num_classes,
                                                        num_heads=kwargs.get("multihead_attention_heads"))
        elif self.head_type == "fully_convolutional":
            self.classifier = nn.Sequential(
                nn.Conv2d(
                    lastconv_output_channels,
                    num_classes,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=(0, 0),
                    bias=False),
                nn.BatchNorm2d(num_classes),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
        elif self.head_type == "mlp":
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(start_dim=1),
                nn.Linear(lastconv_output_channels, last_channel),
                nn.Hardswish(inplace=True),
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(last_channel, num_classes),
            )
        else:
            raise NotImplementedError(f"Head '{self.head_type}' unknown. Must be one of: 'mlp', "
                                      f"'fully_convolutional', 'multihead_attention_pooling'")

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor, return_fmaps: bool = False) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, List[Tensor]]]:
        fmaps = []
        
        for i, layer in enumerate(self.features):
            x = layer(x)
            if return_fmaps:
                fmaps.append(x)
        
        features = F.adaptive_avg_pool2d(x, (1, 1)).squeeze()
        x = self.classifier(x).squeeze()
        
        if features.dim() == 1 and x.dim() == 1:
            # squeezed batch dimension
            features = features.unsqueeze(0)
            x = x.unsqueeze(0)
        
        if return_fmaps:
            return x, fmaps
        else:
            return x, features

    def forward(self, x: Tensor) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, List[Tensor]]]:
        return self._forward_impl(x)


def _mobilenet_v3_conf(
        width_mult: float = 1.0,
        reduced_tail: bool = False,
        dilated: bool = False,
        strides: Tuple[int] = (2, 2, 2, 2),
        **kwargs: Any
):
    reduce_divider = 2 if reduced_tail else 1
    dilation = 2 if dilated else 1

    bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_mult=width_mult)

    # InvertedResidualConfig:
    # input_channels, kernel, expanded_channels, out_channels, use_se, activation, stride, dilation
    inverted_residual_setting = [
        bneck_conf(16, 3, 16, 16, False, "RE", 1, 1),
        bneck_conf(16, 3, 64, 24, False, "RE", strides[0], 1),  # C1
        bneck_conf(24, 3, 72, 24, False, "RE", 1, 1),
        bneck_conf(24, 5, 72, 40, True, "RE", strides[1], 1),  # C2
        bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
        bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
        bneck_conf(40, 3, 240, 80, False, "HS", strides[2], 1),  # C3
        bneck_conf(80, 3, 200, 80, False, "HS", 1, 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
        bneck_conf(80, 3, 480, 112, True, "HS", 1, 1),
        bneck_conf(112, 3, 672, 112, True, "HS", 1, 1),
        bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", strides[3], dilation),  # C4
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
    ]
    last_channel = adjust_channels(1280 // reduce_divider)

    return inverted_residual_setting, last_channel


def _mobilenet_v3(
    inverted_residual_setting: List[InvertedResidualConfig],
    last_channel: int,
    model_name: str,
    pretrained: bool,
    **kwargs: Any,
):
    model = MN(inverted_residual_setting, last_channel, **kwargs)

    if pretrained:
        if model_name in pretrained_models:
            model_url = pretrained_models.get(model_name)
            state_dict = load_state_dict_from_url(model_url, model_dir=model_dir, map_location="cpu")
            if kwargs['head_type'] == "mlp":
                num_classes = state_dict['classifier.5.bias'].size(0)
            elif kwargs['head_type'] == "fully_convolutional":
                num_classes = state_dict['classifier.1.bias'].size(0)
            else:
                print("Loading weights for classifier only implemented for head types 'mlp' and 'fully_convolutional'")
                num_classes = -1
            if kwargs['num_classes'] != num_classes:
                # if the number of logits is not matching the state dict,
                # drop the corresponding pre-trained part
                pretrain_logits = state_dict['classifier.5.bias'].size(0) if kwargs['head_type'] == "mlp" \
                    else state_dict['classifier.1.bias'].size(0)
                print(f"Number of classes defined: {kwargs['num_classes']}, "
                      f"but try to load pre-trained layer with logits: {pretrain_logits}\n"
                      "Dropping last layer.")
                if kwargs['head_type'] == "mlp":
                    del state_dict['classifier.5.weight']
                    del state_dict['classifier.5.bias']
                else:
                    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('classifier')}
            try:
                model.load_state_dict(state_dict)
                print("Loading pre-trained weights successful.")
            except RuntimeError as e:
                print(str(e))
                print("Loading weights pre-trained weights in a non-strict manner.")
                model.load_state_dict(state_dict, strict=False)
        else:
            raise NotImplementedError(f"Model name '{model_name}' unknown.")
    return model


def mobilenet_v3(model_name: str = None, pretrained: bool = True, **kwargs: Any) \
        -> MN:
    """
    Constructs a MobileNetV3 architecture from
    "Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>".
    """
    inverted_residual_setting, last_channel = _mobilenet_v3_conf(**kwargs)
    return _mobilenet_v3(inverted_residual_setting, last_channel, model_name, pretrained, **kwargs)


def get_model(num_classes: int = 527, model_name: str = None, pretrained: bool = True,
              width_mult: float = 1.0, reduced_tail: bool = False, dilated: bool = False,
              strides: Tuple[int, int, int, int] = (2, 2, 2, 2),
              head_type: str = "mlp", multihead_attention_heads: int = 4, input_dim_f: int = 128,
              input_dim_t: int = 1000, se_dims: str = 'c', se_agg: str = "max", se_r: int = 4):
    """
        Arguments to modify the instantiation of a MobileNetv3

        Args:
            num_classes (int): Specifies number of classes to predict
            pretrained_name (str): Specifies name of pre-trained model to load
            width_mult (float): Scales width of network
            reduced_tail (bool): Scales down network tail
            dilated (bool): Applies dilated convolution to network tail
            strides (Tuple): Strides that are set to '2' in original implementation;
                might be changed to modify the size of receptive field and the downsampling factor in
                time and frequency dimension
            head_type (str): decides which classification head to use
            multihead_attention_heads (int): number of heads in case 'multihead_attention_heads' is used
            input_dim_f (int): number of frequency bands
            input_dim_t (int): number of time frames
            se_dims (Tuple): choose dimension to apply squeeze-excitation on, if multiple dimensions are chosen, then
                squeeze-excitation is applied concurrently and se layer outputs are fused by se_agg operation
            se_agg (str): operation to fuse output of concurrent se layers
            se_r (int): squeeze excitation bottleneck size
            se_dims (str): contains letters corresponding to dimensions 'c' - channel, 'f' - frequency, 't' - time
        """

    dim_map = {'c': 1, 'f': 2, 't': 3}
    assert len(se_dims) <= 3 and all([s in dim_map.keys() for s in se_dims]) or se_dims == 'none'
    input_dims = (input_dim_f, input_dim_t)
    if se_dims == 'none':
        se_dims = None
    else:
        se_dims = [dim_map[s] for s in se_dims]
    se_conf = dict(se_dims=se_dims, se_agg=se_agg, se_r=se_r)
    m = mobilenet_v3(model_name=model_name, pretrained=pretrained,
                     num_classes=num_classes, width_mult=width_mult,
                     reduced_tail=reduced_tail, dilated=dilated, strides=strides,
                     head_type=head_type, multihead_attention_heads=multihead_attention_heads,
                     input_dims=input_dims, se_conf=se_conf
                     )
    return m
