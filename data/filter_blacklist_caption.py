import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input", type=str)
parser.add_argument("blacklist", type=str)

args = parser.parse_args()
data = json.load(open(args.input))["audios"]

blacklist_samples = []
with open(args.blacklist) as reader:
    for line in reader.readlines():
        blacklist_samples.append(line.strip())

filtered = []
for item in data:
    if item["audio_id"] not in blacklist_samples:
        filtered.append(item)
json.dump({"audios": filtered}, open(args.input, "w"), indent=4)
