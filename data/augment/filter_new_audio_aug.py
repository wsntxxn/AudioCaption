import json
import argparse
import numpy as np
from tqdm import tqdm


def filter_prediction(args):
    blacklist_aids = []
    with open(args.blacklist, "r") as reader:
        for line in reader.readlines():
            blacklist_aids.append(line.strip())

    aids = []
    with open(args.wav_csv, "r") as reader:
        for line in reader.readlines()[1:]:
            aid, fname = line.strip().split()
            if aid not in blacklist_aids:
                aids.append(aid)
    gnrtr_data = json.load(open(args.generator_pred, "r"))["audios"]
    gnrtr_data = {data["audio_id"]: data["captions"] for data in gnrtr_data}
    evt_gnrtr_data = json.load(open(args.event_generator_pred, "r"))["audios"]
    evt_gnrtr_data = {data["audio_id"]: data["captions"] for data in evt_gnrtr_data}
    rtrv_data = json.load(open(args.retrieval_pred, "r"))["audios"]
    rtrv_data = {data["audio_id"]: data["captions"] for data in rtrv_data}

    if args.statistics:
        gnrtr_sims = []
        evt_gnrtr_sims = []
        for aid in tqdm(aids):
            gnrtr_sim = float(gnrtr_data[aid][0]["at_sim"])
            evt_gnrtr_sim = float(evt_gnrtr_data[aid][0]["at_sim"])
            gnrtr_sims.append(gnrtr_sim)
            evt_gnrtr_sims.append(evt_gnrtr_sim)
            if aid in rtrv_data:
                rtrv_sim = float(rtrv_data[aid][0]["similarity"])

        gnrtr_sims = np.array(gnrtr_sims)
        evt_gnrtr_sims = np.array(evt_gnrtr_sims)

        print("generator: mean: {:.3f}, 25p: {:.3f}, 50p: {:.3f}, 75p: {:.3f}".format(
            np.mean(gnrtr_sims), np.percentile(gnrtr_sims, 25),
            np.percentile(gnrtr_sims, 50), np.percentile(gnrtr_sims, 75)
        ))
        print("event generator: mean: {:.3f}, 25p: {:.3f}, 50p: {:.3f}, "
              "75p: {:.3f}".format(
            np.mean(evt_gnrtr_sims), np.percentile(evt_gnrtr_sims, 25),
            np.percentile(evt_gnrtr_sims, 50), np.percentile(evt_gnrtr_sims, 75)
        ))

    else:
        filtered_data = []
        for aid in tqdm(aids):
            gnrtr_sim = float(gnrtr_data[aid][0]["at_sim"])
            evt_gnrtr_sim = float(evt_gnrtr_data[aid][0]["at_sim"])
            if gnrtr_sim > evt_gnrtr_sim:
                sim = gnrtr_sim
                tokens = gnrtr_data[aid][0]["tokens"]
            else:
                sim = evt_gnrtr_sim
                tokens = evt_gnrtr_data[aid][0]["tokens"]

            if aid in rtrv_data:
                rtrv_sim = float(rtrv_data[aid][0]["similarity"])
                if rtrv_sim > sim:
                    sim = rtrv_sim
                    tokens = rtrv_data[aid][0]["tokens"]

            if sim >= 0.5:
                item = gnrtr_data[aid][0].copy()
                item["tokens"] = tokens
                item["at_sim"] = sim
                filtered_data.append({
                    "audio_id": aid,
                    "captions": [item]
                })
        filtered_data = {"audios": filtered_data}
        print("{} data left after filtering".format(len(filtered_data["audios"])))
        json.dump(filtered_data, open(args.output, "w"), indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_csv", type=str)
    parser.add_argument("--blacklist", type=str)
    parser.add_argument("--generator_pred", type=str)
    parser.add_argument("--event_generator_pred", type=str)
    parser.add_argument("--retrieval_pred", type=str)
    parser.add_argument("--statistics", default=False, action="store_true")
    parser.add_argument("--output", required=False, type=str)
    args = parser.parse_args()
    filter_prediction(args)
