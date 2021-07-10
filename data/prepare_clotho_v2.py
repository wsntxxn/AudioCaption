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
        # audio_id = row["audio_id"]
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
    # wav_scp_writer.close()
    pd.DataFrame(wav_csv_df).to_csv(output_path / "wav.csv", index=False, sep="\t")
    json.dump(data, open(output_path / "text.json", "w"), indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("clotho_annotation", type=str, help="clotho annotation directory, containing {dev,val,eval}.csv")
    parser.add_argument("clotho_audio", type=str, help="clotho audio directory, containing at least {dev,val,eval} subdirectory")
    parser.add_argument("encoded_audio", type=str, help="path used as the new audio directory, where filenames are encoded to remove special characters and soft linked to the original file")
    parser.add_argument("output_path", type=str, default="clotho_v2", help="directory to store processed data files")
    args = parser.parse_args()

    audio_path = args.clotho_audio
    encoded_audio_path = args.encoded_audio
    annotation_path = args.clotho_annotation
    output_path = args.output_path
    audio_path = Path(audio_path)
    annotation_path = Path(annotation_path)
    output_path = Path(output_path)
    encoded_audio_path = Path(encoded_audio_path)
    if encoded_audio_path.exists():
        shutil.rmtree(encoded_audio_path)
    encoded_audio_path.mkdir(parents=True, exist_ok=True)
    for split in ["dev", "val", "eval"]:
        process(split, annotation_path, audio_path, encoded_audio_path, output_path)




