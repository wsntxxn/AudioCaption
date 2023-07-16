import json
import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd


def process(file_name, audioset_df, output_path, audio_link_path):
    output_path.mkdir(exist_ok=True, parents=True)
    audio_link_path.mkdir(exist_ok=True, parents=True)
    df = pd.read_csv(file_name)
    data = {}
    wav_csv_df = []
    audio_id_to_cur_cap_id = {}
    # audio_id: Y[yid].wav
    audioset_df["youtube_id"] = audioset_df["audio_id"].apply(lambda x: x[1: 12])
    yid_to_filename = dict(zip(audioset_df["youtube_id"], audioset_df["file_name"]))
    with tqdm(total=df.shape[0]) as pbar:
        for _, row in df.iterrows():
            audio_id = row["youtube_id"]
            if audio_id not in yid_to_filename:
                pbar.update()
                continue
            if audio_id not in data:
                real_path = yid_to_filename[audio_id]
                audio_id_to_cur_cap_id[audio_id] = 0
                link_path = audio_link_path / Path(real_path).name
                link_path.symlink_to(real_path)
                data[audio_id] = {
                    "file_name": link_path.absolute().__str__()
                }
                wav_csv_df.append({
                    "audio_id": audio_id,
                    "file_name": link_path.absolute().__str__()
                })
            if "captions" not in data[audio_id]:
                data[audio_id]["captions"] = []
            audio_id_to_cur_cap_id[audio_id] += 1
            data[audio_id]["captions"].append({
                "caption": row["caption"],
                "audiocap_id": row["audiocap_id"],
                "cap_id": str(audio_id_to_cur_cap_id[audio_id])
            })
            pbar.update()
    pd.DataFrame(wav_csv_df).to_csv(output_path / "wav.csv", sep="\t", index=False)
    tmp = data.copy()
    data = { "audios": [] }
    for audio_id, captions in tmp.items():
        item = {"audio_id": audio_id}
        item.update(captions)
        data["audios"].append(item)
    json.dump(data, open(output_path / "text.json", "w"), indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audiocaps_annotation", type=str, help="audiocaps annotation directory, containing {train,val,eval}.csv")
    parser.add_argument("audioset_wav_csv", type=str, help="audioset raw data")
    parser.add_argument("--output_path", type=str, default="audiocaps", help="directory to store processed annotation files")
    parser.add_argument("--audio_link_path", type=str, default="audiocaps/audio_links", help="directory to audio links")
    args = parser.parse_args()

    annotation_path = Path(args.audiocaps_annotation)
    output_path = Path(args.output_path)
    audio_link_path = Path(args.audio_link_path)
    audioset_df = pd.read_csv(args.audioset_wav_csv, sep="\t")

    for split in ["train", "val", "test"]:
        process(annotation_path / f"{split}.csv", audioset_df, output_path / split, audio_link_path / split)

