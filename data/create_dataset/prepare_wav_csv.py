#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("audio_directory", type=str, help="directory containing input audio files")
parser.add_argument("output_csv", type=str, help="output wave csv filename")
parser.add_argument("--recursive", default=False, action="store_true")
parser.add_argument("--prefix", default="", type=str, help="prefix to add to audio_id")

args = parser.parse_args()

with open(args.output_csv, "w", newline="") as csv_file:
    writer = csv.writer(csv_file, delimiter="\t")
    writer.writerow(["audio_id", "file_name"])
    if args.recursive:
        fname_generator = Path(args.audio_directory).rglob("*")
    else:
        fname_generator = Path(args.audio_directory).iterdir()
    for file_name in fname_generator:
        ext = file_name.suffix
        if ext in (".wav", ".mp3", ".flac"):
            writer.writerow([args.prefix + file_name.name, str(file_name.absolute())])
    
