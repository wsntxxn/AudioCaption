import math
from functools import partial
import torch
import torch.nn as nn

from captioning.models.base_model import CaptionModel
from captioning.models.utils import init, repeat_tensor


class Attention(nn.Module):

    def __init__(self, kv_dim, q_dim, d_model):
        super().__init__()
        self.q_proj = nn.Linear(q_dim, d_model)
        self.k_proj = nn.Linear(kv_dim, d_model)
        self.v_proj = nn.Linear(kv_dim, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, q = None, k = None, v = None,
                weight = None):
        """
        Args:
            q: [bs, T_tgt, q_dim]
            kv: [bs, T_src, kv_dim]
        """
        if weight is None:
            d_k = k.size(-1)
            q = self.q_proj(q) # [bs, T_tgt, d_model]
            k = self.k_proj(k) # [bs, T_src, d_model]
            v = self.v_proj(v)
            score = torch.matmul(q, k.transpose(-2, -1)) \
                    / math.sqrt(d_k) # [bs, T_tgt, T_src]
            weight = torch.softmax(score, dim=-1) # [bs, T_tgt, T_src]
        out = torch.matmul(weight, v) # [bs, T_tgt, d_model]
        out = self.out_proj(out)

        return out, weight


class StyleCaptionModel(CaptionModel):

    def __init__(self, encoder: nn.Module, decoder: nn.Module,
                 ref_encoder: nn.Module, **kwargs):
        super().__init__(encoder, decoder, **kwargs)
        self.ref_encoder = ref_encoder
        n_style = kwargs.get("n_style", 2)
        style_embed_dim = kwargs.get("style_embed_dim", self.decoder.emb_dim)
        # n_head = kwargs.get("style_attn_heads", 4)
        self.style_embeddings = nn.Parameter(torch.randn(n_style, style_embed_dim))
        assert ref_encoder.embed_dim == self.decoder.emb_dim
        self.style_attn = Attention(style_embed_dim, ref_encoder.embed_dim,
                                    ref_encoder.embed_dim)
        self.inference_forward_keys += ["style_weight"]
        self.init_params()

    def init_params(self):
        nn.init.kaiming_uniform_(self.style_embeddings)
        for p in self.style_attn.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode_style(self, input_dict):
        if input_dict["mode"] == "train":
            cap = input_dict["cap"] # [batch_size, max_len] <bos> + cap + <eos>
            cap_len = input_dict["cap_len"] # [batch_size,] max_len = max_cap_len + 2
            cap_emb = self.ref_encoder(
                {"cap": cap[:, 1: -1], "cap_len": cap_len - 2}
            )["ref_emb"] # [batch_size, embed_dim]
            style_embeddings = repeat_tensor(self.style_embeddings, cap.size(0))
            # [batch_size, n_style, style_embed_dim]
            style_emb, style_weight = self.style_attn(
                cap_emb.unsqueeze(1), style_embeddings, style_embeddings)
            style_emb = style_emb.squeeze(1)
            style_weight = style_weight.squeeze(1)
        elif input_dict["mode"] == "inference":
            weight = input_dict["style_weight"]
            weight = weight.to(self.style_embeddings.device)
            style_emb, _ = self.style_attn(v=self.style_embeddings, weight=weight)
            style_emb = repeat_tensor(style_emb, input_dict["fc_emb"].size(0))
            # [batch_size, style_emb_dim]
        else:
            raise Exception("mode should be either 'train' or 'inference'")
        return style_emb

    def train_forward(self, input_dict):
        input_dict["style"] = self.encode_style(input_dict)
        return super().train_forward(input_dict)

    def inference_forward(self, input_dict):
        input_dict["style"] = self.encode_style(input_dict)
        return super().inference_forward(input_dict)
