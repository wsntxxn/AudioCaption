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
        n_head = kwargs.get("style_attn_heads", 4)
        self.cls_idx = self.ref_encoder.vocab_size - 1
        self.style_embeddings = nn.Parameter(torch.randn(n_style, style_embed_dim))
        assert ref_encoder.embed_dim == self.decoder.emb_dim
        self.style_attn = Attention(style_embed_dim, ref_encoder.embed_dim,
                                    ref_encoder.embed_dim)
        self.init_params()

    def init_params(self):
        nn.init.kaiming_uniform_(self.style_embeddings)
        for p in self.style_attn.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode_style(self, input_dict):
        if input_dict["mode"] == "train":
            caps = input_dict["caps"] # [N, T] <bos> + cap + <eos>
            cap_lens = input_dict["cap_lens"] # [N,] max_length = max_cap_length + 2
            cls_tokens = torch.empty(
                caps.size(0), 1, dtype=torch.long).fill_(self.cls_idx).to(caps.device)
            caps = torch.cat((cls_tokens, caps[:, 1:-1]), dim=-1)
            cap_embeds = self.ref_encoder(
                {"caps": caps, "cap_lens": cap_lens - 1}
            )["ref_embs"] # [N, embed_dim]
            style_embeddings = repeat_tensor(self.style_embeddings, caps.size(0)) # [N, n_style, style_embed_dim]
            style_embeds, style_weights = self.style_attn(
                cap_embeds.unsqueeze(1), style_embeddings, style_embeddings)
            style_embeds = style_embeds.squeeze(1)
            style_weights = style_weights.squeeze(1)
        else:
            weights = input_dict["style_weights"]
            weights = weights.to(self.style_embeddings.device)
            style_embeds, _ = self.style_attn(v=self.style_embeddings, weight=weights)
            style_embeds = repeat_tensor(style_embeds, input_dict["raw_feats"].size(0)) # [N, style_embed_dim]
        return style_embeds

    def forward(self, input_dict):
        """
        input_dict: {
            (required)
            mode: train/inference,
            raw_feats,
            raw_feat_lens,
            fc_feats,
            attn_feats,
            attn_feat_lens,
            [sample_method: greedy],
            [temp: 1.0] (in case of no teacher forcing)

            (optional, mode=train)
            caps,
            cap_lens,
            ss_ratio,

            (optional, mode=inference)
            sample_method: greedy/beam,
            max_length,
            temp,
            beam_size (optional, sample_method=beam),
        }
        """
        encoder_input_keys = ["raw_feats", "raw_feat_lens", "fc_feats", "attn_feats", "attn_feat_lens"]
        encoder_input = { key: input_dict[key] for key in encoder_input_keys }
        encoder_output_dict = self.encoder(encoder_input)
        if input_dict["mode"] == "train":
            forward_dict = { "mode": "train", "sample_method": "greedy", "temp": 1.0 }
            for key in self.train_forward_keys:
                forward_dict[key] = input_dict[key]
            forward_dict.update(encoder_output_dict)
            forward_dict["styles"] = self.encode_style(input_dict)
            output = self.train_forward(forward_dict)
        elif input_dict["mode"] == "inference":
            forward_dict = {"mode": "inference"}
            default_args = { "sample_method": "greedy", "max_length": self.max_length, "temp": 1.0 }
            for key in self.inference_forward_keys:
                if key in input_dict:
                    forward_dict[key] = input_dict[key]
                else:
                    forward_dict[key] = default_args[key]
            if forward_dict["sample_method"] == "beam":
                if "beam_size" in input_dict:
                    forward_dict["beam_size"] = input_dict["beam_size"]
                else:
                    forward_dict["beam_size"] = 3
            forward_dict.update(encoder_output_dict)
            forward_dict["styles"] = self.encode_style(input_dict)
            output = self.inference_forward(forward_dict)
        else:
            raise Exception("mode should be either 'train' or 'inference'")

        return output
