from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from captioning.models.base_model import CaptionMetaMixin
from captioning.models.utils import init


class WmlEncoderKdWrapper(nn.Module, CaptionMetaMixin):

    def __init__(self,
                 model: nn.Module,
                 shared_dim: int,
                 tchr_layer_to_dims: Dict[str, int],
                 loss_type: str = "mse",):
        super().__init__()
        self.model = model
        self.tchr_layers = list(tchr_layer_to_dims.keys())
        self.stdnt_qv_proj = nn.Linear(model.encoder.fc_emb_size,
                                       2 * shared_dim)
        self.stdnt_qv_proj.apply(init)
        for layer, dim in tchr_layer_to_dims.items():
            self.add_module(f'tchr_kv_proj_{layer}', nn.Linear(dim, 2 * shared_dim))
            getattr(self, f'tchr_kv_proj_{layer}').apply(init)
        if loss_type == "mse":
            self.loss_fn = nn.MSELoss(reduction="none")
    
    def forward(self, input_dict: Dict):
        output_dict = self.model(input_dict)
        if "tchr_output" in input_dict:
            stdnt_emb = output_dict["fc_emb"]
            stdnt_qv = self.stdnt_qv_proj(stdnt_emb)
            stdnt_q, stdnt_v = torch.chunk(stdnt_qv, 2, dim=-1)
            
            tchr_output = input_dict["tchr_output"]
            layer_ks, layer_vs = [], []
            for layer in self.tchr_layers:
                layer_kv = getattr(self, f'tchr_kv_proj_{layer}')(tchr_output[layer])
                layer_k, layer_v = torch.chunk(layer_kv, 2, dim=-1)
                layer_ks.append(layer_k)
                layer_vs.append(layer_v)
            layer_ks = torch.stack(layer_ks, dim=1)
            layer_vs = torch.stack(layer_vs, dim=1)
            weights = torch.softmax(stdnt_q.unsqueeze(1) @ layer_ks.transpose(1, 2), dim=-1)
            stdnt_v = repeat(stdnt_v, 'b d -> b n d', n=len(self.tchr_layers))
            loss = self.loss_fn(stdnt_v, layer_vs).mean(dim=-1, keepdim=True)
            loss = (weights @ loss).mean()
            output_dict["enc_kd_loss"] = loss
        return output_dict


class MseEncoderKdWrapper(nn.Module, CaptionMetaMixin):

    def __init__(self,
                 model: nn.Module,
                 shared_dim: int,
                 tchr_dim: int,
                 use_tchr_proj: bool = True,
                 l2_norm: bool = False,
                 ):
        super().__init__()
        self.model = model
        self.use_tchr_proj = use_tchr_proj
        if not use_tchr_proj:
            assert shared_dim == tchr_dim
        self.tchr_dim = tchr_dim
        self.l2_norm = l2_norm
        if hasattr(model, "encoder"):
            self.stdnt_proj = nn.Linear(model.encoder.fc_emb_size,
                                        shared_dim)
        else:
            self.stdnt_proj = nn.Linear(model.fc_emb_size,
                                        shared_dim)
        self.stdnt_proj.apply(init)
        if use_tchr_proj:
            self.tchr_proj = nn.Linear(tchr_dim, shared_dim)
            self.tchr_proj.apply(init)
        else:
            self.tchr_proj = nn.Identity()

    def forward(self, input_dict: Dict):
        unsup = input_dict.get("unsup", False)
        if unsup is False:
            if self.use_tchr_proj:
                output_dict = self.model(input_dict)
                stdnt_emb = output_dict["fc_emb"]
            else:
                encoder_output = self.model.encoder(input_dict)
                stdnt_emb = encoder_output["fc_emb"]
                encoder_output["fc_emb"] = self.stdnt_proj(encoder_output["fc_emb"])
                encoder_output["attn_emb"] = self.stdnt_proj(encoder_output["attn_emb"])
                output_dict = self.model.forward_decoder(input_dict, encoder_output)
        else:
            output_dict = self.model.encoder(input_dict)
            stdnt_emb = output_dict["fc_emb"]
        if "tchr_output" in input_dict:
            stdnt_emb = self.stdnt_proj(stdnt_emb)
            tchr_emb = input_dict["tchr_output"]["embedding"]
            thcr_emb = self.tchr_proj(tchr_emb)

            if self.l2_norm:
                stdnt_emb = F.normalize(stdnt_emb, dim=-1)
                thcr_emb = F.normalize(thcr_emb, dim=-1)

            loss = F.mse_loss(stdnt_emb, thcr_emb)
            output_dict["enc_kd_loss"] = loss
        return output_dict


class ContraEncoderKdWrapper(nn.Module, CaptionMetaMixin):

    def __init__(self,
                 model: nn.Module,
                 shared_dim: int,
                 tchr_dim: int,
                 ):
        super().__init__()
        self.model = model
        self.tchr_dim = tchr_dim
        if hasattr(model, "encoder"):
            self.stdnt_proj = nn.Linear(model.encoder.fc_emb_size,
                                        shared_dim)
        else:
            self.stdnt_proj = nn.Linear(model.fc_emb_size,
                                        shared_dim)
        self.stdnt_proj.apply(init)
        self.tchr_proj = nn.Linear(tchr_dim, shared_dim)
        self.tchr_proj.apply(init)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, input_dict: Dict):
        unsup = input_dict.get("unsup", False)
        if unsup is False:
            output_dict = self.model(input_dict)
        else:
            output_dict = self.model.encoder(input_dict)
        if "tchr_output" in input_dict:
            stdnt_emb = output_dict["fc_emb"]
            stdnt_emb = self.stdnt_proj(stdnt_emb)
            tchr_emb = input_dict["tchr_output"]["embedding"]
            thcr_emb = self.tchr_proj(tchr_emb)

            stdnt_emb = F.normalize(stdnt_emb, dim=-1)
            thcr_emb = F.normalize(thcr_emb, dim=-1)

            unscaled_logit = stdnt_emb @ thcr_emb.transpose(0, 1)
            logit = self.logit_scale * unscaled_logit
            label = torch.arange(logit.shape[0]).to(logit.device)
            loss1 = F.cross_entropy(logit, label)
            loss2 = F.cross_entropy(logit.transpose(0, 1), label)
            loss = (loss1 + loss2) / 2
            output_dict["enc_kd_loss"] = loss
        return output_dict


class ContraMseEncoderKdWrapper(nn.Module, CaptionMetaMixin):

    def __init__(self,
                 model: nn.Module,
                 shared_dim: int,
                 tchr_dim: int,
                 use_tchr_proj: bool = True,
                 l2_norm: bool = False,
                 ):
        super().__init__()
        self.model = model
        self.use_tchr_proj = use_tchr_proj
        if not use_tchr_proj:
            assert shared_dim == tchr_dim
        self.tchr_dim = tchr_dim
        self.l2_norm = l2_norm
        if hasattr(model, "encoder"):
            self.stdnt_proj = nn.Linear(model.encoder.fc_emb_size,
                                        shared_dim)
        else:
            self.stdnt_proj = nn.Linear(model.fc_emb_size,
                                        shared_dim)
        self.stdnt_proj.apply(init)
        if use_tchr_proj:
            self.tchr_proj = nn.Linear(tchr_dim, shared_dim)
            self.tchr_proj.apply(init)
        else:
            self.tchr_proj = nn.Identity()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, input_dict: Dict):
        unsup = input_dict.get("unsup", False)
        if unsup is False:
            if self.use_tchr_proj:
                output_dict = self.model(input_dict)
                stdnt_emb = output_dict["fc_emb"]
            else:
                encoder_output = self.model.encoder(input_dict)
                stdnt_emb = encoder_output["fc_emb"]
                encoder_output["fc_emb"] = self.stdnt_proj(encoder_output["fc_emb"])
                encoder_output["attn_emb"] = self.stdnt_proj(encoder_output["attn_emb"])
                output_dict = self.model.forward_decoder(input_dict, encoder_output)
        else:
            output_dict = self.model.encoder(input_dict)
            stdnt_emb = output_dict["fc_emb"]
        if "tchr_output" in input_dict:
            stdnt_emb = self.stdnt_proj(stdnt_emb)
            tchr_emb = input_dict["tchr_output"]["embedding"]
            thcr_emb = self.tchr_proj(tchr_emb)

            if self.l2_norm:
                stdnt_emb = F.normalize(stdnt_emb, dim=-1)
                thcr_emb = F.normalize(thcr_emb, dim=-1)

            mse_loss = F.mse_loss(stdnt_emb, thcr_emb)

            stdnt_emb = F.normalize(stdnt_emb, dim=-1)
            thcr_emb = F.normalize(thcr_emb, dim=-1)
            unscaled_logit = stdnt_emb @ thcr_emb.transpose(0, 1)
            logit = self.logit_scale * unscaled_logit
            label = torch.arange(logit.shape[0]).to(logit.device)
            loss1 = F.cross_entropy(logit, label)
            loss2 = F.cross_entropy(logit.transpose(0, 1), label)
            cntr_loss = (loss1 + loss2) / 2
            output_dict["enc_kd_loss"] = mse_loss + cntr_loss

        return output_dict
