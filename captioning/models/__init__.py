import numpy as np
import torch
import torch.nn as nn

from captioning.models.utils import max_with_lens, mean_with_lens


def embedding_pooling(x, lens, pooling="mean"):
    if pooling == "max":
        fc_embs = max_with_lens(x, lens)
    elif pooling == "mean":
        fc_embs = mean_with_lens(x, lens)
    elif pooling == "mean+max":
        x_mean = mean_with_lens(x, lens)
        x_max = max_with_lens(x, lens)
        fc_embs = x_mean + x_max
    elif pooling == "last":
        indices = (lens - 1).reshape(-1, 1, 1).repeat(1, 1, x.size(-1))
        # indices: [N, 1, hidden]
        fc_embs = torch.gather(x, 1, indices).squeeze(1)
    else:
        raise Exception(f"pooling method {pooling} not support")
    return fc_embs


class BaseEncoder(nn.Module):
    
    """
    Encode the given audio into embedding
    Base encoder class, cannot be called directly
    All encoders should inherit from this class
    """

    def __init__(self, spec_dim, fc_feat_dim, attn_feat_dim):
        super(BaseEncoder, self).__init__()
        self.spec_dim = spec_dim
        self.fc_feat_dim = fc_feat_dim
        self.attn_feat_dim = attn_feat_dim


    def forward(self, x):
        #########################
        # Arguments:
        # `x`: {
        #     (may contain)
        #     wav: [batch_size, n_samples],
        #     spec: [batch_size, n_frames, spec_dim],
        #     fc: [batch_size, fc_feat_dim],
        #     attn: [batch_size, attn_max_len, attn_feat_dim],
        #     attn_len: [batch_size,]
        #     ......
        #  }
        #
        # Returns:
        # `encoded`: {
        #     fc_emb: [batch_size, fc_emb_dim],
        #     attn_emb: [batch_size, attn_max_len, attn_emb_dim],
        #     attn_emb_lens: [batch_size,]
        # }
        #########################
        raise NotImplementedError


class BaseDecoder(nn.Module):
    """
    Take word/audio embeddings and output the next word probs
    """
    def __init__(self, emb_dim, vocab_size, fc_emb_dim,
                 attn_emb_dim, dropout=0.2, tie_weights=False):
        super().__init__()
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.fc_emb_dim = fc_emb_dim
        self.attn_emb_dim = attn_emb_dim
        self.tie_weights = tie_weights
        self.word_embedding = nn.Embedding(vocab_size, emb_dim)
        self.in_dropout = nn.Dropout(dropout)

    def forward(self, x):
        raise NotImplementedError

    def load_word_embedding(self, weight, freeze=True):
        embedding = np.load(weight)
        assert embedding.shape[0] == self.vocab_size, "vocabulary size mismatch"
        assert embedding.shape[1] == self.emb_dim, "embed size mismatch"
        
        # embeddings = torch.as_tensor(embeddings).float()
        # self.word_embeddings.weight = nn.Parameter(embeddings)
        # for para in self.word_embeddings.parameters():
            # para.requires_grad = tune
        self.word_embedding = nn.Embedding.from_pretrained(embedding,
            freeze=freeze)
