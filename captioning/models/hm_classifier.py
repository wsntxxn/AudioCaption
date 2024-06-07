import torch
import torch.nn as nn

from captioning.utils.model_util import mean_with_lens, max_with_lens, init, pack_wrapper, generate_length_mask


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
        indices = (lens - 1).reshape(-1, 1, 1).repeat(1, 1, x.size(-1)).to(x.device)
        # indices: [N, 1, hidden]
        fc_embs = torch.gather(x, 1, indices).squeeze(1)
    else:
        raise Exception(f"pooling method {pooling} not support")
    return fc_embs


class RnnEncoder(nn.Module):

    def __init__(self, vocab_size, embed_dim, pooling="mean", **kwargs):
        super().__init__()
        self.pooling = pooling
        self.hidden_size = kwargs.get("hidden_size", 512)
        self.bidirectional = kwargs.get("bidirectional", True)
        self.num_layers = kwargs.get("num_layers", 1)
        self.dropout = kwargs.get("dropout", 0.2)
        self.rnn_type = kwargs.get("rnn_type", "LSTM")
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.network = getattr(nn, self.rnn_type)(
            embed_dim,
            self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            dropout=self.dropout,
            batch_first=True)
        self.classifier = nn.Linear(
            self.hidden_size * (self.bidirectional + 1), 1)
        self.apply(init)

    def forward(self, input_dict):
        caps = input_dict["caps"]
        lens = input_dict["lens"]
        lens = torch.as_tensor(lens)
        if isinstance(caps, (torch.LongTensor, torch.cuda.LongTensor)):
            embeds = self.embedding(caps)
        else:
            embeds = torch.matmul(caps, self.embedding.weight)
        # x: [N, T, E]
        out = pack_wrapper(self.network, embeds, lens)
        # out: [N, T, hidden*n_direction]
        fc_embs = embedding_pooling(out, lens, self.pooling)
        output = torch.sigmoid(self.classifier(fc_embs)).squeeze(-1)
        return output
