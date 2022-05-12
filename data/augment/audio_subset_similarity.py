import json
import argparse
import random
from h5py import File
import numpy as np
import pandas as pd
from tqdm import tqdm
import sklearn.metrics


parser = argparse.ArgumentParser()
parser.add_argument("data", type=str, choices=["clotho", "audiocaps"])
parser.add_argument("output_label", type=str)
parser.add_argument("--thresholds", type=float, default=[0.0, 1.0], nargs=2)

args = parser.parse_args()

thresholds = args.thresholds
data = args.data

if data == "clotho":
    caption_data = json.load(open("clotho_v2/dev/text.json"))["audios"]
    audio_id_to_cap_items = {}
    audio_id_to_json_idx = {}
    for idx, item in enumerate(caption_data):
        audio_id = item["audio_id"]
        audio_id_to_cap_items[audio_id] = item["captions"]
        audio_id_to_json_idx[audio_id] = idx

    v1_df = pd.read_csv("clotho_v1/dev/lms.csv", sep="\t")
    v1_audio_ids = v1_df["audio_id"].unique()

    embs = []
    idx_to_audio_ids = []
    with File("clotho_v2/dev/panns_cnn14_fc.h5", "r") as store:
        for audio_id, embedding in tqdm(store.items(), ascii=True):
            embs.append(embedding[()])
            idx_to_audio_ids.append(audio_id)
    embs = np.stack(embs)

    v1_idxs = []
    v1_idx_to_audio_ids = []
    aug_idxs = []
    aug_idx_to_audio_ids = []
    for idx, audio_id in enumerate(idx_to_audio_ids):
        if audio_id in v1_audio_ids:
            v1_idxs.append(idx)
            v1_idx_to_audio_ids.append(audio_id)
        else:
            aug_idxs.append(idx)
            aug_idx_to_audio_ids.append(audio_id)

    v1_idxs = np.array(v1_idxs)
    aug_idxs = np.array(aug_idxs)

    v1_embs = embs[v1_idxs]
    aug_embs = embs[aug_idxs]
    similarity = sklearn.metrics.pairwise.cosine_similarity(v1_embs, aug_embs)

    aug_data = []
    with tqdm(total=similarity.shape[1], ascii=True) as pbar:
        for idx in range(similarity.shape[1]):
            audio_id = idx_to_audio_ids[aug_idxs[idx]]
            audio_id_to_similarity = {
                v1_idx_to_audio_ids[_]: similarity[:, idx][_] for _ in np.where(
                    (similarity[:, idx] > thresholds[0]) &
                    (similarity[:, idx] < thresholds[1])
                )[0]
            }
            matched_audio_ids = list(audio_id_to_similarity.keys())
            random.shuffle(matched_audio_ids)
            if len(matched_audio_ids) > 0:
                matched_audio_ids = list(zip(*sorted(
                    audio_id_to_similarity.items(),
                    key=lambda x: x[1],
                    reverse=True)))[0]
            
                aug_item = caption_data[audio_id_to_json_idx[audio_id]]
                matched_audio_id = matched_audio_ids[0]
                score = audio_id_to_similarity[matched_audio_id]
                aug_item["captions"] = audio_id_to_cap_items[matched_audio_id]
                aug_data.append(aug_item)
            pbar.update()
            
    json.dump({"audios": aug_data}, open(args.output_label, "w"), indent=4)

elif data == "audiocaps":
    caption_data = json.load(open("audiocaps/train/text.json"))["audios"]
    audio_id_to_cap_items = {}
    audio_id_to_json_idx = {}
    for idx, item in enumerate(caption_data):
        audio_id = item["audio_id"]
        audio_id_to_cap_items[audio_id] = item["captions"]
        audio_id_to_json_idx[audio_id] = idx

    subset_data = json.load(open("audiocaps/train_12ksubset/text.json"))["audios"]
    subset_audio_ids = [item["audio_id"] for item in subset_data]

    embs = []
    idx_to_audio_ids = []
    with File("audiocaps/train/panns_cnn14_fc.h5", "r") as store:
        for audio_id, embedding in tqdm(store.items(), ascii=True):
            embs.append(embedding[()])
            idx_to_audio_ids.append(audio_id)
    embs = np.stack(embs)

    subset_idxs = []
    subset_idx_to_audio_ids = []
    aug_idxs = []
    aug_idx_to_audio_ids = []
    for idx, audio_id in enumerate(idx_to_audio_ids):
        if audio_id in subset_audio_ids:
            subset_idxs.append(idx)
            subset_idx_to_audio_ids.append(audio_id)
        else:
            aug_idxs.append(idx)
            aug_idx_to_audio_ids.append(audio_id)

    subset_idxs = np.array(subset_idxs)
    aug_idxs = np.array(aug_idxs)

    subset_embs = embs[subset_idxs]
    aug_embs = embs[aug_idxs]
    similarity = sklearn.metrics.pairwise.cosine_similarity(subset_embs, aug_embs)

    aug_data = []
    with tqdm(total=similarity.shape[1], ascii=True) as pbar:
        for idx in range(similarity.shape[1]):
            audio_id = idx_to_audio_ids[aug_idxs[idx]]
            audio_id_to_similarity = {
                subset_idx_to_audio_ids[_]: similarity[:, idx][_] for _ in np.where(
                    (similarity[:, idx] > thresholds[0]) &
                    (similarity[:, idx] < thresholds[1])
                )[0]
            }
            matched_audio_ids = list(audio_id_to_similarity.keys())
            random.shuffle(matched_audio_ids)
            if len(matched_audio_ids) > 0:
                matched_audio_ids = list(zip(*sorted(
                    audio_id_to_similarity.items(),
                    key=lambda x: x[1],
                    reverse=True)))[0]
            
                aug_item = caption_data[audio_id_to_json_idx[audio_id]]
                matched_audio_id = matched_audio_ids[0]
                score = audio_id_to_similarity[matched_audio_id]
                aug_item["captions"] = audio_id_to_cap_items[matched_audio_id]
                aug_data.append(aug_item)
            pbar.update()
            
    json.dump({"audios": aug_data}, open(args.output_label, "w"), indent=4)
