# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from captioning.models import BaseEncoder, embedding_pooling
from captioning.utils.model_util import init, pack_wrapper


class RnnEncoder(BaseEncoder):

    def __init__(self, spec_dim, fc_feat_dim, attn_feat_dim,
                 pooling="mean", **kwargs):
        super().__init__(spec_dim, fc_feat_dim, attn_feat_dim)
        self.pooling = pooling
        self.hidden_size = kwargs.get('hidden_size', 512)
        self.bidirectional = kwargs.get('bidirectional', False)
        self.num_layers = kwargs.get('num_layers', 1)
        self.dropout = kwargs.get('dropout', 0.2)
        self.rnn_type = kwargs.get('rnn_type', "GRU")
        self.in_bn = kwargs.get('in_bn', False)
        self.embed_dim = self.hidden_size * (self.bidirectional + 1)
        self.network = getattr(nn, self.rnn_type)(
            attn_feat_dim,
            self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            dropout=self.dropout,
            batch_first=True)
        if self.in_bn:
            self.bn = nn.BatchNorm1d(self.embed_dim)
        self.apply(init)

    def forward(self, input_dict):
        x = input_dict["attn"]
        lens = input_dict["attn_len"]
        lens = torch.as_tensor(lens)
        # x: [N, T, E]
        if self.in_bn:
            x = pack_wrapper(self.bn, x, lens)
        out = pack_wrapper(self.network, x, lens)
        # out: [N, T, hidden]
        attn_emb = out
        fc_emb = embedding_pooling(out, lens, self.pooling)
        return {
            "attn_emb": attn_emb,
            "fc_emb": fc_emb,
            "attn_emb_len": lens
        }
