import argparse
import csv
import json


def write_csv(args):
    aid_to_h5 = {}
    with open(args.waveform_csv, "r", newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter="\t")
        for row in reader:
            aid_to_h5[row[0]] = row[1]

    data = json.load(open(args.annotation))["audios"]
    with open(args.output, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter="\t")
        writer.writerow(["audio_id", "hdf5_path"])
        for item in data:
            audio_id = item["audio_id"]
            h5_path = aid_to_h5[audio_id]
            writer.writerow([audio_id, h5_path])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", "-a", type=str, required=True)
    parser.add_argument("--waveform_csv", "-w", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, required=True)
    args = parser.parse_args()
    write_csv(args)