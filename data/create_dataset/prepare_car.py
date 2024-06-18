from pathlib import Path
import json
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("target", help="target output directory", type=str)

args = parser.parse_args()
root = args.target
root = Path(root)

df = pd.read_json("/mnt/lustre/sjtu/shared/data/raa/AudioCaption/car/label/zh_dev.json")
data = {}
for _, row in df.iterrows():
    filename = Path(row["filename"]).name
    audio_id = f"car_{filename}"
    caption = row["caption"]
    # tokens = " ".join(row["tokens"])
    caption_index = row["caption_index"]
    if audio_id not in data:
        data[audio_id] = {}
    if "raw_name" not in data[audio_id]:
        data[audio_id]["raw_name"] = filename
    if "captions" not in data[audio_id]:
        data[audio_id]["captions"] = []
    data[audio_id]["captions"].append({
        "caption": caption,
        # "tokens": tokens,
        "cap_id": f"{audio_id}_{caption_index}"
    })
   
dump_data = []
for audio_id, item in data.items():
    dump_item = {
        "audio_id": audio_id,
        "raw_name": item["raw_name"],
        "captions": item["captions"]
    }
    dump_data.append(dump_item)
(root / "dev").mkdir(parents=True, exist_ok=True)
json.dump({"audios": dump_data}, 
          open(Path(root) / "dev/text.json", "w"),
          indent=4,
          ensure_ascii=False)


df = pd.read_json("/mnt/lustre/sjtu/shared/data/raa/AudioCaption/car/label/zh_eval.json")
data = {}
for _, row in df.iterrows():
    filename = Path(row["filename"]).name
    audio_id = f"car_{filename}"
    caption = row["caption"]
    # tokens = " ".join(row["tokens"])
    caption_index = row["caption_index"]
    if audio_id not in data:
        data[audio_id] = {}
    if "raw_name" not in data[audio_id]:
        data[audio_id]["raw_name"] = filename
    if "captions" not in data[audio_id]:
        data[audio_id]["captions"] = []
    data[audio_id]["captions"].append({
        "caption": caption,
        # "tokens": tokens,
        "cap_id": str(caption_index)
    })
   
dump_data = []
for audio_id, item in data.items():
    dump_item = {
        "audio_id": audio_id,
        "raw_name": item["raw_name"],
        "captions": item["captions"]
    }
    dump_data.append(dump_item)
(root / "eval").mkdir(parents=True, exist_ok=True)
json.dump({"audios": dump_data}, 
          open(Path(root) / "eval/text.json", "w"),
          indent=4,
          ensure_ascii=False)
