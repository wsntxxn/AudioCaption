import argparse
import json
import math
from collections import Counter
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("input_json", type=str)
parser.add_argument("output_word_condition", type=str)
parser.add_argument("output_caption_condition", type=str)
parser.add_argument("--sentence_reduce", type=str, default="sum", choices=["mean", "sum"])

args = parser.parse_args()

sentence_reduce = args.sentence_reduce
data = json.load(open(args.input_json))["audios"]
counter = Counter()
total_words = 0
for item in data:
    for cap_item in item["captions"]:
        counter.update(cap_item["tokens"].split())
        total_words += len(cap_item["tokens"].split())

word_to_cond = {}
for word, cnt in counter.items():
    word_to_cond[word] = - math.log(cnt / total_words)
pd.DataFrame(
    {
        "word": word_to_cond.keys(), 
        "specificity": word_to_cond.values()
    }).to_csv(args.output_word_condition, sep="\t", index=False, float_format="%.3f")

cap_ids = []
specificities = []
for item in data:
    audio_id = item["audio_id"]
    for cap_item in item["captions"]:
        cap_id = cap_item["cap_id"]
        tokens = cap_item["tokens"].split()
        if sentence_reduce == "sum":
            specificity = sum(map(word_to_cond.__getitem__, tokens))
        else:
            specificity = sum(map(word_to_cond.__getitem__, tokens)) / len(tokens)
        cap_ids.append(f"{audio_id}_{cap_id}")
        specificities.append(specificity)
pd.DataFrame(
    {
        "cap_id": cap_ids,
        "specificity": specificities
    }).to_csv(args.output_caption_condition, sep="\t", index=False, float_format="%.3f")
