import json
from pathlib import Path
import argparse
import pickle
from collections import Counter

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


parser = argparse.ArgumentParser()
parser.add_argument("--split", choices=["dev", "val", "eval"],
                    required=True, type=str)
parser.add_argument("--keyword_encoder", required=True, type=str)
parser.add_argument("--output_name", required=True, type=str)
parser.add_argument("--threshold", type=int, default=30)

args = parser.parse_args()
split = args.split
encoder_path = args.keyword_encoder

normalization = {
    "bird": ["birds", "bird", "birdsong"],
    "footstep": ["footsteps", "steps"],
    "outdoor": ["outdoor", "outdoors"],
    "bell": ["bell", "bells"],
    "car": ["cars", "car"],
    "walk": ["walk", "walking"],
    "drop": ["drop", "drops"],
    "wood": ["woods", "wood", "wooden"],
    "drip": ["drip", "dripping"],
    "voice": ["voice", "voices"],
    "atmosphere": ["atmo", "atmos", "atmosphere"],
    "metal": ["metal", "metallic"]
}
dropped_words = ["field-recording", "ambience", "ambiance", "ambient", "weather", "summer", "background",
                 "soundscape", "sound", "recording", "general-noise", "spring", "morning", "interior",
                 "outside", "background-sound", "field", "winter", "night"]


dev_meta_df = pd.read_csv("/mnt/lustre/sjtu/shared/data/raa/DCASE2021/task6/"
                          "metadata/dev.csv", engine="python",
                          usecols=["file_name", "keywords"])

words = []
for _, row in dev_meta_df.iterrows():
    keywords = [x.lower() for x in row["keywords"].split(";")]
    words.extend(keywords)

cnt = Counter(words)
cnt_df = pd.DataFrame(cnt.items(), columns=["word", "count"])
filterd_df = cnt_df[cnt_df["count"] > args.threshold]

word_to_norm = {}
for norm, norm_words in normalization.items():
    for word in norm_words:
        word_to_norm[word] = norm

word_unique = set()
for word in filterd_df["word"].unique():
    if word in dropped_words:
        continue
    elif word in word_to_norm:
        word_unique.add(word_to_norm[word])
    else:
        word_unique.add(word)

encoder_path = f"data/clotho_v2/dev/keywords/{encoder_path}"
if not Path(encoder_path).exists():
    label_array = []
    for _, row in dev_meta_df.iterrows():
        label = []
        for word in row["keywords"].split(";"):
            word = word.lower()
            if word in word_unique and word not in label:
                label.append(word)
            elif word in word_to_norm and word_to_norm[word] not in label:
                label.append(word_to_norm[word])
        label_array.append(label)
    label_encoder = MultiLabelBinarizer()
    label_encoder.fit(label_array)
    pickle.dump(label_encoder, open(encoder_path, "wb"))

caption_data = json.load(open(f"data/clotho_v2/{split}/text.json"))["audios"]
meta_df = pd.read_csv("/mnt/lustre/sjtu/shared/data/raa/DCASE2021/task6/"
                      f"metadata/{split}.csv", engine="python",
                      usecols=["file_name", "keywords"])
fname_to_keywords = dict(zip(meta_df["file_name"], meta_df["keywords"]))

keywords = []
audio_ids = []
for item in caption_data:
    audio_id = item["audio_id"]
    audio_ids.append(audio_id)
    words = []
    for word in fname_to_keywords[item["raw_name"]].split(";"):
        word = word.lower()
        if word in word_unique and word not in words:
            words.append(word)
        elif word in word_to_norm and word_to_norm[word] not in words:
            words.append(word_to_norm[word])
    words = "; ".join(words)
    keywords.append(words)

output_df = pd.DataFrame({"audio_id": audio_ids, "keywords": keywords})
output_df.to_csv(f"data/clotho_v2/{split}/keywords/{args.output_name}",
                 sep="\t", index=False)
