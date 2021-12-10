#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("audio_directory", type=str, help="directory containing input audio files")
parser.add_argument("output_csv", type=str, help="output wave csv filename")
parser.add_argument("--prefix", default=None, type=str, help="prefix of audio id")

args = parser.parse_args()
prefix = args.prefix

with open(args.output_csv, "w", newline="") as csv_file:
    writer = csv.writer(csv_file, delimiter="\t")
    writer.writerow(["audio_id", "file_name"])
    for file_name in Path(args.audio_directory).iterdir():
        if prefix:
            writer.writerow([f"{prefix}_{file_name.name}", str(file_name.absolute())])
        else:
            writer.writerow([file_name.name, str(file_name.absolute())])
    
