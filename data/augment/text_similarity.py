import json
import random
from h5py import File
from tqdm import tqdm
import sklearn.metrics
import numpy as np
import argparse

random.seed(1)

parser = argparse.ArgumentParser()
parser.add_argument("input_label", type=str)
parser.add_argument("text_embedding", type=str)
parser.add_argument("output_label", type=str)
parser.add_argument("--thresholds", type=float, default=[0.9, 1.0], nargs=2)
parser.add_argument("--max_caption_aug", type=int, default=None)
parser.add_argument("--nearest", action="store_true",
                    help="whether to choose nearest caption in training data for augmentation, default: False")
parser.add_argument("--exclude_real", action="store_true",
                    help="whether to exclude original real data in output, default: False")

args = parser.parse_args()

thresholds = args.thresholds
max_caption_aug = args.max_caption_aug
nearest = args.nearest

embeds = []
embed_idx_to_key = []
with File(args.text_embedding, "r") as store:
    for key, embed in tqdm(store.items(), ascii=True):
        embeds.append(embed[()])
        embed_idx_to_key.append(key)
embeds = np.stack(embeds)

similarity = sklearn.metrics.pairwise.cosine_similarity(embeds)

data = json.load(open(args.input_label, "r"))["audios"]
audio_id_to_idx = {item["audio_id"]: idx for idx, item in enumerate(data)}
audio_id_length = len(next(iter(audio_id_to_idx.keys())))


with tqdm(total=similarity.shape[0], ascii=True) as pbar:
    for idx in range(similarity.shape[0]):
        audio_id = embed_idx_to_key[idx][:audio_id_length]
        cap_id = embed_idx_to_key[idx][audio_id_length+1:]
        key_to_similarity = {
            embed_idx_to_key[_]: similarity[idx][_] for _ in np.where(
                (similarity[idx] > thresholds[0]) & (similarity[idx] < thresholds[1]))[0]
        }
        matched_keys = list(key_to_similarity.keys())
        random.shuffle(matched_keys)
        if nearest:
            if len(matched_keys) > 0:
                matched_keys = list(zip(*sorted(key_to_similarity.items(), key=lambda x: x[1], reverse=True)))[0]

        ##########################
        # augmentation starts
        ##########################
        aug_num = 0
        for matched_key in matched_keys:
            matched_audio_id = matched_key[:audio_id_length]
            matched_cap_id = matched_key[audio_id_length+1:]
            if matched_audio_id != audio_id:
                aug_num += 1
                aug_item = data[audio_id_to_idx[matched_audio_id]]["captions"][int(matched_cap_id)-1].copy()
                aug_item["cap_id"] = f"{cap_id}_textaug_{aug_num}"
                data[audio_id_to_idx[audio_id]]["captions"].append(aug_item)
                if max_caption_aug and aug_num >= max_caption_aug:
                    break
        pbar.update()

if args.exclude_real:
    aug_data = []
    for item in data:
        aug_item = {}
        aug_item["audio_id"] = item["audio_id"]
        aug_item["captions"] = []
        for cap_item in item["captions"]:
            if "aug" in str(cap_item["cap_id"]):
                aug_item["captions"].append(cap_item)
        if len(aug_item["captions"]) > 0:
            aug_data.append(aug_item)
    json.dump({"audios": aug_data}, open(args.output_label, "w"), indent=4)
else:
    json.dump({"audios": data}, open(args.output_label, "w"), indent=4)
