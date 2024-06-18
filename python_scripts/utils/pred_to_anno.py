import argparse
import json

from tqdm import tqdm
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("--prediction_input", "-i", required=True, type=str)
parser.add_argument("--annotaion_output", "-o", required=True, type=str)
parser.add_argument("--wav_csv", "-w", type=str, required=False, default=None)

args = parser.parse_args()

if args.wav_csv is not None:
    wav_df = pd.read_csv(args.wav_csv, sep="\t")
    target_aids = set(wav_df["audio_id"].values)

predictions = json.load(open(args.prediction_input))
data = []

if "predictions" in predictions:
    predictions = predictions["predictions"]
    with tqdm(total=len(predictions), ascii=True) as pbar:
        for prediction in predictions:
            aid = prediction["filename"]
            if args.wav_csv is not None and aid not in target_aids:
                pbar.update()
                continue
            tokens = prediction["tokens"]
            data.append({
                "audio_id": aid,
                "captions": [
                    {
                        "cap_id": "1",
                        "tokens": tokens
                    }
                ]
            })
            pbar.update()
elif isinstance(predictions, dict):
    with tqdm(total=len(predictions), ascii=True) as pbar:
        for aid, tokens in predictions.items():
            aid = aid[1: 12]
            if args.wav_csv is not None and aid not in target_aids:
                pbar.update()
                continue
            data.append({
                "audio_id": aid,
                "captions": [
                    {
                        "cap_id": "1",
                        "tokens": tokens
                    }
                ]
            })
            pbar.update()
json.dump({"audios": data}, open(args.annotaion_output, "w"), indent=4)

