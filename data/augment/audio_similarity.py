import os
import pdb
from h5py import File
from tqdm import tqdm
import sklearn.metrics
import numpy as np
import pandas as pd

THRESHOLD = 0.95

dev_val_embeds = []
dev_val_idx2key = []
with File("clotho_v2/panns_dev_val.h5", "r") as panns_store:
    for key, embedding in tqdm(panns_store.items()):
        dev_val_embeds.append(embedding[()])
        dev_val_idx2key.append(key)
dev_val_embeds = np.stack(dev_val_embeds)

fsd50k_embeds = []
fsd50k_idx2key = []
with File("clotho_v2/panns_fsd50k.h5", "r") as panns_store:
    for key, embedding in tqdm(panns_store.items()):
        fsd50k_embeds.append(embedding[()])
        fsd50k_idx2key.append(key)
fsd50k_embeds = np.stack(fsd50k_embeds)

similarity = sklearn.metrics.pairwise.cosine_similarity(fsd50k_embeds, dev_val_embeds)

dev_val_df = pd.read_json("clotho_v2/dev_val.json")
data = []
fsd50k_meta_df = pd.read_csv("/mnt/lustre/sjtu/shared/data/raa/SceneEventData/FSD50K/FSD50K.ground_truth/dev.csv", dtype={"fname": str})
dev_fnames = fsd50k_meta_df["fname"].unique()

with tqdm(total=similarity.shape[0]) as pbar:
    for fsd50k_idx, (sim, clotho_idx) in enumerate(zip(similarity.max(axis=1), similarity.argmax(axis=1))):
        if sim > THRESHOLD:
            clotho_key = dev_val_idx2key[clotho_idx]
            split = "FSD50K.dev_audio" if fsd50k_idx2key[fsd50k_idx][:-4] in dev_fnames else "FSD50K.eval_audio"
            for _, row in dev_val_df[dev_val_df["audio_key"].isin([clotho_key])].reset_index().iterrows():
                data.append({
                    "filename": os.path.join("/mnt/lustre/sjtu/shared/data/raa/SceneEventData/FSD50K", split, fsd50k_idx2key[fsd50k_idx]),
                    "caption": row["caption"],
                    "caption_key": fsd50k_idx2key[fsd50k_idx] + "_" + str(_ + 1),
                    "tokens": row["tokens"],
                    "audio_key": fsd50k_idx2key[fsd50k_idx]
                })
        pbar.update()
df = pd.DataFrame(data)
df.to_json("clotho_v2/fsd50k.json")

