import torch
import math
import torch.nn as nn

from captioning.models.utils import generate_length_mask

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [T, N, E]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):

    def __init__(self, vocab_size, d_model, embed_dim, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
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
        caps = input_dict["caps"]
        cap_lens = input_dict["cap_lens"]
        cap_lens = torch.as_tensor(cap_lens)

        embeds = self.in_dropout(self.word_embedding(caps)) * math.sqrt(self.d_model)
        embeds = embeds.transpose(0, 1)
        embeds = self.pos_encoder(embeds)

        src_key_padding_mask = ~generate_length_mask(cap_lens, embeds.size(0)).to(embeds.device)
        output = self.model(embeds, src_key_padding_mask=src_key_padding_mask)
        output = output.transpose(0, 1) # [N, T, embed_dim]
        cls_embeds = output[:, 0, :]
        ref_embeds = self.out_transform(cls_embeds)

        return {
            "ref_embs": ref_embeds,
        }
