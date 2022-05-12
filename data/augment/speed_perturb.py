from pathlib import Path
import json
import argparse
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('input_dir', type=str)
parser.add_argument('output_dir', type=str)
parser.add_argument('--speeds', type=float, nargs="+", default=[0.9, 1.1])
parser.add_argument('--sr', type=int, default=16000)

args = parser.parse_args()
argsdict = vars(args)

output_dir = Path(args.output_dir)
if not output_dir.exists():
    output_dir.mkdir(parents=True)

input_dir = Path(args.input_dir)
wav_df = pd.read_csv(input_dir / "wav.csv", sep="\t")
caption_info = json.load(open(input_dir / "text.json", "r"))["audios"]
audio_id_to_idx = {}
for idx, audio_info in enumerate(caption_info):
    audio_id_to_idx[audio_info["audio_id"]] = idx

sp_data = []
sp_caption_info = []
for _, row in wav_df.iterrows():
    audio_id, file_name = row["audio_id"], row["file_name"]
    for speed in args.speeds:
        sp_data.append({
            "audio_id": f"sp{speed}-{audio_id}",
            "file_name": f"ffmpeg -i {file_name} -f wav -ar {args.sr} -ab 16 - | sox -t wav - -t wav - speed {speed} |"
        })
        raw_caption = caption_info[audio_id_to_idx[audio_id]]
        sp_caption = raw_caption.copy()
        sp_caption["audio_id"] = f"sp{speed}-{audio_id}"
        sp_caption_info.append(sp_caption)

caption_info.extend(sp_caption_info)
json.dump({"audios": caption_info}, open(output_dir / "text.json", "w"), indent=4)
pd.DataFrame(sp_data).to_csv(output_dir / "wav.csv", sep="\t", index=False)

