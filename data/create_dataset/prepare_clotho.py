import json
import argparse
import shutil
import hashlib
from pathlib import Path
import pandas as pd


def process(split: str, 
            annotation_path: Path, 
            audio_path: Path, 
            encoded_audio_path: Path, 
            output_path: Path):
    annotation_file = annotation_path / f"{split}.csv"
    df = pd.read_csv(annotation_file)
    output_path = output_path / f"{split}"
    output_path.mkdir(parents=True, exist_ok=True)
    data = []
    wav_csv_df = []
    audio_path = audio_path / split
    for _, row in df.iterrows():
        raw_file_name = Path(row["file_name"]).stem
        audio_id = hashlib.md5(f"{split}_{raw_file_name}".encode()).hexdigest()
        encoded_file_name = f"{audio_id}.wav"
        (encoded_audio_path / encoded_file_name).symlink_to(audio_path.resolve() / row["file_name"])
        item = { "audio_id": audio_id, "captions": [], "raw_name": row["file_name"] }
        wav_csv_df.append({
            "audio_id": audio_id,
            "file_name": str((encoded_audio_path / encoded_file_name).absolute())
        })
        for cap_id in range(1, 6):
            item["captions"].append({
                "caption": row["caption_" + str(cap_id)],
                "cap_id": str(cap_id)
            })
        data.append(item)
    data = { "audios": data }
    pd.DataFrame(wav_csv_df).to_csv(output_path / "wav.csv", index=False, sep="\t")
    json.dump(data, open(output_path / "text.json", "w"), indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("clotho_root_dir", type=str, help="clotho dataset directory, containing at least `annotation/{dev,eval}.csv` and `audio/{dev/eval}`")
    parser.add_argument("output_path", type=str, help="directory to store processed data files")
    parser.add_argument("--version", type=int, default=2, help="clotho dataset version, 1 or 2", choices=[1, 2])
    args = parser.parse_args()

    data_root_dir = Path(args.clotho_root_dir)
    output_path = Path(args.output_path)
    audio_path = data_root_dir / "audio"
    annotation_path = data_root_dir / "annotation"
    encoded_audio_path = output_path / "hashed_audio"
    if encoded_audio_path.exists():
        shutil.rmtree(encoded_audio_path)
    encoded_audio_path.mkdir(parents=True, exist_ok=True)
    if args.version == 1:
        splits = ["dev", "eval"]
    else:
        splits = ["dev", "val", "eval"]
    for split in splits:
        process(split, annotation_path, audio_path, encoded_audio_path, output_path)




