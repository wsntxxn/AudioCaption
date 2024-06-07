# coding=utf-8
#!/usr/bin/env python3

import json
import numpy as np
import gensim
from gensim.models import Word2Vec
from tqdm import tqdm
import fire

from captioning.datasets.text_tokenizer import DictTokenizer


def create_embedding(vocab_file: str,
                     embed_size: int,
                     output: str,
                     caption_file: str = None,
                     pretrained_weights_path: str = None,
                     **word2vec_kwargs):

    tokenizer = DictTokenizer(tokenizer_path=vocab_file)

    if pretrained_weights_path:
        model = gensim.models.KeyedVectors.load_word2vec_format(
            fname=pretrained_weights_path,
            binary=True,
        )
        if model.vector_size != embed_size:
            assert embed_size < model.vector_size, f"only reduce dimension, cannot add dimesion {model.vector_size} to {embed_size}"
            from sklearn.decomposition import PCA
            pca = PCA(n_components=embed_size)
            model.vectors = pca.fit_transform(model.vectors)
    else:
        cap_data = json.load(open(caption_file))
        sentences = []
        for item in cap_data:
            for cap_item in item["captions"]:
                sentences.append(cap_item["tokens"])
        epochs = word2vec_kwargs.get("epochs", 10)
        if "epochs" in word2vec_kwargs:
            del word2vec_kwargs["epochs"]
        model = Word2Vec(size=embed_size, min_count=1, **word2vec_kwargs)
        model.build_vocab(sentences=sentences)
        model.train(sentences=sentences, total_examples=len(sentences), epochs=epochs)
    
    word_embeddings = np.random.randn(len(tokenizer), embed_size)
    
    if isinstance(model, gensim.models.word2vec.Word2Vec):
        model = model.wv
    with tqdm(total=len(tokenizer), ascii=True) as pbar:
        for word, idx in tokenizer.word2idx.items():
            try:
                word_embeddings[idx] = model.get_vector(word)
            except KeyError:
                print(f"word {word} not found in word2vec model, it is random initialized!")
            pbar.update()

    np.save(output, word_embeddings)

    print("Finish writing word2vec embeddings to " + output)


if __name__ == "__main__":
    fire.Fire(create_embedding)



