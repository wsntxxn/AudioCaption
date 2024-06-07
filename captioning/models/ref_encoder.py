import torch
import math
import torch.nn as nn

from captioning.utils.model_util import generate_length_mask, PositionalEncoding


class TransformerEncoder(nn.Module):

    def __init__(self, vocab_size, d_model, embed_dim, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.cls_idx = self.vocab_size - 1
        self.embed_dim = embed_dim
        self.word_embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        dropout = kwargs.get("dropout", 0.2)
        self.in_dropout = nn.Dropout(dropout)
        self.pos_encoder = PositionalEncoding(self.d_model, dropout)
        self.nhead = kwargs.get("nhead", self.d_model // 64)
        self.nlayers = kwargs.get("nlayers", 2)
        self.dim_feedforward = kwargs.get("dim_feedforward", self.d_model * 4)

        layer = nn.TransformerEncoderLayer(d_model=self.d_model,
                                           nhead=self.nhead,
                                           dim_feedforward=self.dim_feedforward,
                                           dropout=dropout)
        self.model = nn.TransformerEncoder(layer, self.nlayers)
        self.out_transform = nn.Linear(self.d_model, self.embed_dim)
        self.init_params()

    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_dict):
        cap = input_dict["cap"]
        cap_len = input_dict["cap_len"]
        cap_len = torch.as_tensor(cap_len)

        cls_tokens = torch.empty(
            cap.size(0), 1, dtype=torch.long).fill_(self.cls_idx).to(cap.device)
        cap = torch.cat((cls_tokens, cap), dim=-1)
        cap_len = cap_len + 1

        embed = self.in_dropout(self.word_embedding(cap)) * math.sqrt(self.d_model)
        embed = embed.transpose(0, 1)
        embed = self.pos_encoder(embed)

        src_key_padding_mask = ~generate_length_mask(cap_len, embed.size(0)).to(embed.device)
        output = self.model(embed, src_key_padding_mask=src_key_padding_mask)
        output = output.transpose(0, 1) # [N, T, embed_dim]
        cls_emb = output[:, 0, :]
        ref_emb = self.out_transform(cls_emb)

        return {
            "ref_emb": ref_emb,
        }
