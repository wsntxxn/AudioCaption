import math

import torch
import torch.nn as nn

from captioning.models import BaseDecoder
from captioning.utils.model_util import generate_length_mask, PositionalEncoding
from captioning.utils.train_util import merge_load_state_dict



class TransformerDecoder(BaseDecoder):

    def __init__(self,
                 emb_dim,
                 vocab_size,
                 fc_emb_dim,
                 attn_emb_dim,
                 dropout,
                 freeze=False,
                 tie_weights=False,
                 **kwargs):
        super().__init__(emb_dim, vocab_size, fc_emb_dim, attn_emb_dim,
                         dropout=dropout, tie_weights=tie_weights)
        self.d_model = emb_dim
        self.nhead = kwargs.get("nhead", self.d_model // 64)
        self.nlayers = kwargs.get("nlayers", 2)
        self.dim_feedforward = kwargs.get("dim_feedforward", self.d_model * 4)

        self.pos_encoder = PositionalEncoding(self.d_model, dropout)
        layer = nn.TransformerDecoderLayer(d_model=self.d_model,
                                           nhead=self.nhead,
                                           dim_feedforward=self.dim_feedforward,
                                           dropout=dropout)
        self.model = nn.TransformerDecoder(layer, self.nlayers)
        self.classifier = nn.Linear(self.d_model, vocab_size, bias=False)
        if tie_weights:
            self.classifier.weight = self.word_embedding.weight
        self.attn_proj = nn.Sequential(
            nn.Linear(self.attn_emb_dim, self.d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(self.d_model)
        )
        self.init_params()

        self.freeze = freeze
        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def load_pretrained(self, pretrained, output_fn):
        checkpoint = torch.load(pretrained, map_location="cpu")

        if "model" in checkpoint:
            checkpoint = checkpoint["model"]
            if next(iter(checkpoint)).startswith("decoder."):
                state_dict = {}
                for k, v in checkpoint.items():
                    state_dict[k[8:]] = v

        loaded_keys = merge_load_state_dict(state_dict, self, output_fn)
        if self.freeze:
            for name, param in self.named_parameters():
                if name in loaded_keys:
                    param.requires_grad = False
                else:
                    param.requires_grad = True


    def generate_square_subsequent_mask(self, max_length):
        mask = (torch.triu(torch.ones(max_length, max_length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, input_dict):
        word = input_dict["word"]
        attn_emb = input_dict["attn_emb"]
        attn_emb_len = input_dict["attn_emb_len"]
        cap_padding_mask = input_dict["cap_padding_mask"]

        p_attn_emb = self.attn_proj(attn_emb)
        p_attn_emb = p_attn_emb.transpose(0, 1) # [T_src, N, emb_dim]
        word = word.to(attn_emb.device)
        embed = self.in_dropout(self.word_embedding(word)) * math.sqrt(self.emb_dim) # [N, T, emb_dim]
        embed = embed.transpose(0, 1) # [T, N, emb_dim]
        embed = self.pos_encoder(embed)

        tgt_mask = self.generate_square_subsequent_mask(embed.size(0)).to(attn_emb.device)
        memory_key_padding_mask = ~generate_length_mask(attn_emb_len, attn_emb.size(1)).to(attn_emb.device)
        output = self.model(embed, p_attn_emb, tgt_mask=tgt_mask,
                            tgt_key_padding_mask=cap_padding_mask,
                            memory_key_padding_mask=memory_key_padding_mask)
        output = output.transpose(0, 1)
        output = {
            "embed": output,
            "logit": self.classifier(output),
        }
        return output


class M2TransformerDecoder(BaseDecoder):

    def __init__(self, vocab_size, fc_emb_dim, attn_emb_dim, dropout=0.1, **kwargs):
        super().__init__(attn_emb_dim, vocab_size, fc_emb_dim, attn_emb_dim, dropout=dropout,)
        try:
            from m2transformer.models.transformer import MeshedDecoder
        except:
            raise ImportError("meshed-memory-transformer not installed; please run `pip install git+https://github.com/ruotianluo/meshed-memory-transformer.git`")
        del self.word_embedding
        del self.in_dropout

        self.d_model = attn_emb_dim
        self.nhead = kwargs.get("nhead", self.d_model // 64)
        self.nlayers = kwargs.get("nlayers", 2)
        self.dim_feedforward = kwargs.get("dim_feedforward", self.d_model * 4)
        self.model = MeshedDecoder(vocab_size, 100, self.nlayers, 0,
                                   d_model=self.d_model,
                                   h=self.nhead,
                                   d_ff=self.dim_feedforward,
                                   dropout=dropout)
        self.init_params()

    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_dict):
        word = input_dict["word"]
        attn_emb = input_dict["attn_emb"]
        attn_emb_mask = input_dict["attn_emb_mask"]
        word = word.to(attn_emb.device)
        embed, logit = self.model(word, attn_emb, attn_emb_mask)
        output = {
            "embed": embed,
            "logit": logit,
        }
        return output


class EventTransformerDecoder(TransformerDecoder):

    def forward(self, input_dict):
        word = input_dict["word"] # index of word embeddings
        attn_emb = input_dict["attn_emb"]
        attn_emb_len = input_dict["attn_emb_len"]
        cap_padding_mask = input_dict["cap_padding_mask"]
        event_emb = input_dict["event"] # [N, emb_dim]

        p_attn_emb = self.attn_proj(attn_emb)
        p_attn_emb = p_attn_emb.transpose(0, 1) # [T_src, N, emb_dim]
        word = word.to(attn_emb.device)
        embed = self.in_dropout(self.word_embedding(word)) * math.sqrt(self.emb_dim) # [N, T, emb_dim]

        embed = embed.transpose(0, 1) # [T, N, emb_dim]
        embed += event_emb
        embed = self.pos_encoder(embed)

        tgt_mask = self.generate_square_subsequent_mask(embed.size(0)).to(attn_emb.device)
        memory_key_padding_mask = ~generate_length_mask(attn_emb_len, attn_emb.size(1)).to(attn_emb.device)
        output = self.model(embed, p_attn_emb, tgt_mask=tgt_mask,
                            tgt_key_padding_mask=cap_padding_mask,
                            memory_key_padding_mask=memory_key_padding_mask)
        output = output.transpose(0, 1)
        output = {
            "embed": output,
            "logit": self.classifier(output),
        }
        return output


class KeywordProbTransformerDecoder(TransformerDecoder):

    def __init__(self, emb_dim, vocab_size, fc_emb_dim, attn_emb_dim,
                 dropout, keyword_classes_num, **kwargs):
        super().__init__(emb_dim, vocab_size, fc_emb_dim, attn_emb_dim,
                         dropout, **kwargs)
        self.keyword_proj = nn.Linear(keyword_classes_num, self.d_model)
        self.word_keyword_norm = nn.LayerNorm(self.d_model)

    def forward(self, input_dict):
        word = input_dict["word"] # index of word embeddings
        attn_emb = input_dict["attn_emb"]
        attn_emb_len = input_dict["attn_emb_len"]
        cap_padding_mask = input_dict["cap_padding_mask"]
        keyword = input_dict["keyword"] # [N, keyword_classes_num]

        p_attn_emb = self.attn_proj(attn_emb)
        p_attn_emb = p_attn_emb.transpose(0, 1) # [T_src, N, emb_dim]
        word = word.to(attn_emb.device)
        embed = self.in_dropout(self.word_embedding(word)) * math.sqrt(self.emb_dim) # [N, T, emb_dim]

        embed = embed.transpose(0, 1) # [T, N, emb_dim]
        embed += self.keyword_proj(keyword)
        embed = self.word_keyword_norm(embed)

        embed = self.pos_encoder(embed)

        tgt_mask = self.generate_square_subsequent_mask(embed.size(0)).to(attn_emb.device)
        memory_key_padding_mask = ~generate_length_mask(attn_emb_len, attn_emb.size(1)).to(attn_emb.device)
        output = self.model(embed, p_attn_emb, tgt_mask=tgt_mask,
                            tgt_key_padding_mask=cap_padding_mask,
                            memory_key_padding_mask=memory_key_padding_mask)
        output = output.transpose(0, 1)
        output = {
            "embed": output,
            "logit": self.classifier(output),
        }
        return output
