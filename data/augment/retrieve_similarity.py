import json
import random
from h5py import File
from tqdm import tqdm
import numpy as np
import argparse

random.seed(1)

parser = argparse.ArgumentParser()
parser.add_argument("annotation", type=str)
parser.add_argument("sim_store", type=str)
parser.add_argument("output", type=str)
parser.add_argument("--thresholds", type=float, default=[0.0, 1.0], nargs=2)
parser.add_argument("--max_caption_aug", type=int, default=None)
parser.add_argument("--nearest", action="store_true",
                    help="whether to choose nearest caption in training data for augmentation, default: False")
parser.add_argument("--exclude_real", action="store_true",
                    help="whether to exclude original real data in output, default: False")

args = parser.parse_args()

thresholds = args.thresholds
max_caption_aug = args.max_caption_aug
nearest = args.nearest

with File(args.sim_store, "r") as store:
    similarity = store["sim"][()]
    audio_ids = store["audio_id"][()]
    if "text_id" in store:
        cap_ids = store["text_id"][()]
        cap_ids = [_.decode("UTF-8") + "_1" for _ in cap_ids]
    else:
        cap_ids = [_.decode("UTF-8") + "_1" for _ in audio_ids]

audio_ids = [_.decode("UTF-8") for _ in audio_ids]
if len(audio_ids[0]) == 16:
    audio_ids = [audio_id[1: 12] for audio_id in audio_ids]

data = json.load(open(args.annotation, "r"))["audios"]
cap_id_to_cap_item = {
    f"{item['audio_id']}_{cap_item['cap_id']}": cap_item
        for item in data for cap_item in item["captions"]
}
##########################################
# only part of the annotations
##########################################
if len(cap_id_to_cap_item) != len(cap_ids):
    avail_cap_ids = set(cap_id_to_cap_item.keys())
    avail_cap_idxs = [_ for _, cap_id in enumerate(cap_ids) if cap_id in avail_cap_ids]
    similarity = similarity[:, avail_cap_idxs]
    cap_ids = [cap_ids[avail_cap_idx] for avail_cap_idx in avail_cap_idxs]
##########################################
# audio_id_to_audio_idx = {item["audio_id"]: idx for idx, item in enumerate(data)}
audio_id_to_audio_idx = {audio_id: idx for idx, audio_id in enumerate(audio_ids)}
audio_id_length = len(audio_ids[0])

aug_data = []
with tqdm(total=similarity.shape[0], ascii=True, ncols=100) as pbar:
    for audio_idx in range(similarity.shape[0]):
        audio_id = audio_ids[audio_idx]
        cap_id_to_similarity = {
            cap_ids[_]: similarity[audio_idx][_] for _ in np.where(
                (similarity[audio_idx] >= thresholds[0]) & 
                (similarity[audio_idx] <= thresholds[1])
            )[0]
        }
        thresholded_cap_ids = list(cap_id_to_similarity.keys())
        random.shuffle(thresholded_cap_ids)
        if nearest:
            if len(thresholded_cap_ids) > 0:
                thresholded_cap_ids = list(zip(*sorted(cap_id_to_similarity.items(),
                                                       key=lambda x: x[1], reverse=True)))[0]
        ##########################
        # augmentation starts
        ##########################
        # if args.exclude_real:
            # aug_data.append({"audio_id": audio_id, "captions": []})
        # aug_num = 0
        # for cap_id in thresholded_cap_ids:
            # score = cap_id_to_similarity[cap_id]
            # if cap_id[:audio_id_length] != audio_id:
                # aug_num += 1
                # aug_item = cap_id_to_cap_item[cap_id].copy()
                # aug_item["cap_id"] = f"retriveaug_{aug_num}"
                # aug_item["similarity"] = f"{score:.3f}"
                # if args.exclude_real:
                    # aug_data[-1]["captions"].append(aug_item)
                # else:
                    # data[audio_id_to_audio_idx[audio_id]]["captions"].append(aug_item)
                # if max_caption_aug and aug_num >= max_caption_aug:
                    # break
        # if args.exclude_real and len(aug_data[-1]["captions"]) == 0:
            # aug_data.pop()
        aug_data.append({"audio_id": audio_id, "captions": []})
        aug_num = 0
        for cap_id in thresholded_cap_ids:
            score = cap_id_to_similarity[cap_id]
            aug_num += 1
            aug_item = cap_id_to_cap_item[cap_id].copy()
            aug_item["cap_id"] = f"retriveaug_{aug_num}"
            aug_item["similarity"] = f"{score:.3f}"
            aug_data[-1]["captions"].append(aug_item)
            if max_caption_aug and aug_num >= max_caption_aug:
                break
        if len(aug_data[-1]["captions"]) == 0:
            aug_data.pop()
        pbar.update()

if args.exclude_real:
    print(f"{len(aug_data)} augment audio after filtering")
    json.dump({"audios": aug_data}, open(args.output, "w"), indent=4)
else:
    json.dump({"audios": data}, open(args.output, "w"), indent=4)

