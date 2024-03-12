import pickle
from pathlib import Path

import numpy as np
from captioning.utils.train_util import pad_sequence


class DictTokenizer:

    def __init__(self,
                 tokenizer_path: str = None,
                 max_length: int = 20) -> None:
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.add_word("<pad>")
        self.add_word("<start>")
        self.add_word("<end>")
        self.add_word("<unk>")
        if tokenizer_path is not None and Path(tokenizer_path).exists():
            state_dict = pickle.load(open(tokenizer_path, "rb"))
            self.load_state_dict(state_dict)
            self.loaded = True
        else:
            self.loaded = False
        self.bos, self.eos = self.word2idx["<start>"], self.word2idx["<end>"]
        self.pad = self.word2idx["<pad>"]
        self.max_length = max_length

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def encode_word(self, word):
        if word in self.word2idx:
            return self.word2idx[word]
        else:
            return self.word2idx["<unk>"]

    def __call__(self, texts):
        assert isinstance(texts, list), "the input must be List[str]"
        batch_tokens = []
        for text in texts:
            tokens = [self.encode_word(token) for token in text.split()][:self.max_length]
            tokens = [self.bos] + tokens + [self.eos]
            tokens = np.array(tokens)
            batch_tokens.append(tokens)
        caps, cap_lens = pad_sequence(batch_tokens, self.pad)
        return {
            "cap": caps,
            "cap_len": cap_lens
        }

    def decode(self, batch_token_ids):
        output = []
        for token_ids in batch_token_ids:
            tokens = []
            for token_id in token_ids:
                if token_id == self.eos:
                    break
                elif token_id == self.bos:
                    continue
                tokens.append(self.idx2word[token_id])
            output.append(" ".join(tokens))
        return output

    def __len__(self):
        return len(self.word2idx)

    def state_dict(self):
        return self.word2idx
    
    def load_state_dict(self, state_dict):
        self.word2idx = state_dict
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.idx = len(self.word2idx)


class HuggingfaceTokenizer:

    def __init__(self,
                 model_name_or_path,
                 max_length) -> None:
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.max_length = max_length
        self.bos, self.eos = self.tokenizer.bos_token_id, self.tokenizer.eos_token_id
        self.pad = self.tokenizer.pad_token_id
        self.loaded = True

    def __call__(self, texts):
        assert isinstance(texts, list), "the input must be List[str]"
        batch_token_dict = self.tokenizer(texts,
                                          padding=True,
                                          truncation=True,
                                          max_length=self.max_length,
                                          return_tensors="pt")
        batch_token_dict["cap"] = batch_token_dict["input_ids"]
        cap_lens = batch_token_dict["attention_mask"].sum(dim=1)
        cap_lens = cap_lens.numpy().astype(np.int32)
        batch_token_dict["cap_len"] = cap_lens
        return batch_token_dict

    def decode(self, batch_token_ids):
        return self.tokenizer.batch_decode(batch_token_ids, skip_special_tokens=True)
