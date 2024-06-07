import torch
import json
import argparse

import numpy as np
from tqdm import trange
from transformers import AutoTokenizer, ClapModel


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(device)
    tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")
    
    key_to_pred = {}
    with open(args.prediction, "r") as f:
        all_pred = json.load(f)["predictions"]

    for item in all_pred:
        key_to_pred[item["filename"]] = item["tokens"]

    with open(args.reference, "r") as f:
        all_ref = json.load(f)["audios"]
    
    key_to_ref = {}
    for item in all_ref:
        caps = []
        for cap in item["captions"]:
            caps.append(cap["caption"])
        key_to_ref[item["audio_id"]] = caps

    keys = list(key_to_pred.keys())
    data_len = len(keys)
    cap_per_audio = len(key_to_ref[keys[0]])
    key_to_score = {}
    scores = []
    for i in trange(0, data_len, args.batch_size):
        batch_keys = keys[i: i + args.batch_size]
        batch_pred = [key_to_pred[key] for key in batch_keys]

        inputs = tokenizer(batch_pred, padding=True, return_tensors="pt").to(device)
        emb_pred = model.get_text_features(**inputs)

        batch_ref = sum([key_to_ref[key] for key in batch_keys], [])
        inputs = tokenizer(batch_ref, padding=True, return_tensors="pt").to(device)
        emb_ref = model.get_text_features(**inputs)
        emb_ref = emb_ref.view(len(batch_keys), cap_per_audio, -1)

        batch_score = torch.bmm(emb_pred.unsqueeze(1), emb_ref.transpose(-2, -1)).squeeze(1)
        batch_score = torch.mean(batch_score, dim=1)
        for key, score in zip(batch_keys, batch_score):
            key_to_score[key] = score.item()
            scores.append(score.item())
    
    dataset_score = np.mean(scores)
    with open(args.output, "w") as f:
        f.write(f"Clap score: {dataset_score:.3f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", "-p", type=str, required=True)
    parser.add_argument("--reference", "-r", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, required=True)
    parser.add_argument("--batch_size", "-b", type=int, default=32)

    args = parser.parse_args()
    main(args)

