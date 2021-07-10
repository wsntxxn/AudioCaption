import pickle
import fire
import numpy as np
import pandas as pd
from tqdm import tqdm


class EmbeddingExtractor(object):

    def extract_sentbert(self, caption_file: str, output: str, dev: bool=True, zh: bool=False):
        from sentence_transformers import SentenceTransformer
        lang2model = {
            "zh": "distiluse-base-multilingual-cased",
            "en": "bert-base-nli-mean-tokens"
        }
        lang = "zh" if zh else "en"
        model = SentenceTransformer(lang2model[lang])

        self.extract(caption_file, model, output, dev)

    def extract_originbert(self, caption_file: str, output: str, dev: bool=True, ip="localhost"):
        from bert_serving.client import BertClient
        caption_df = pd.read_json(caption_file, dtype={"key": str})
        client = BertClient(ip)
        
        self.extract(caption_file, client, output, dev)

    def extract(self, caption_file: str, model, output, dev: bool):
        caption_df = pd.read_json(caption_file, dtype={"key": str})
        embeddings = {}

        if dev:
            with tqdm(total=caption_df.shape[0], ascii=True) as pbar:
                for idx, row in caption_df.iterrows():
                    caption = row["caption"]
                    key = row["key"]
                    caption_index = row["caption_index"]
                    embeddings["{}_{}".format(key, caption_index)] = np.array(model.encode([caption])).reshape(-1)
                    pbar.update()

        else:
            dump = {}

            with tqdm(total=caption_df.shape[0], ascii=True) as pbar:
                for idx, row in caption_df.iterrows():
                    key = row["key"]
                    caption = row["caption"]
                    value = np.array(model.encode([caption])).reshape(-1)

                    if key not in embeddings.keys():
                        embeddings[key] = [value]
                    else:
                        embeddings[key].append(value)

                    pbar.update()
                
            for key in embeddings:
                dump[key] = np.stack(embeddings[key])

            embeddings = dump

        with open(output, "wb") as f:
            pickle.dump(embeddings, f)
        
    def extract_sbert(self, 
                      input_json: str, 
                      output: str):
        from sentence_transformers import SentenceTransformer
        import torch
        from h5py import File
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SentenceTransformer("bert-base-nli-mean-tokens")
        model = model.to(device)
        model.eval()

        df = pd.read_json(input_json)

        with torch.no_grad(), tqdm(total=df.shape[0], ascii=True) as pbar, File(output, "w") as store:
            for idx, row in df.iterrows():
                caption = row["caption"]
                store[row["caption_key"]] = model.encode([caption]).squeeze(0)
                pbar.update()


if __name__ == "__main__":
    fire.Fire(EmbeddingExtractor)
