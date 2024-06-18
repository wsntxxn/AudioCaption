#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import logging


def get_parser():
    parser = argparse.ArgumentParser(
        description="concatenate annotation json files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_jsons", type=str, nargs="+", help="input json files")
    parser.add_argument("output_json", type=str, help="output json file")
    parser.add_argument("--indent", type=int, default=None)
    return parser


def merge_data(data_total, data):
    data = data["audios"]
    for item in data:
        audio_id = item["audio_id"]
        if audio_id in data_total:
            data_total[audio_id]["captions"].extend(item["captions"])
        else:
            data_total[audio_id] = item


if __name__ == "__main__":
    args = get_parser().parse_args()

    # logging info
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=logfmt)

    data = {}
    for x in args.input_jsons:
        with open(x, encoding="utf-8") as f:
            j = json.load(f)
        logging.info(x + ": has " + str(len(j["audios"])) + " audio clips")
        merge_data(data, j)

    logging.info("new json has " + str(len(data)) + " audio clips")

    tmp = []
    for audio_id, value in data.items():
        tmp.append(value)
    data = {"audios": tmp}
    json.dump(
        data,
        open(args.output_json, "w"),
        indent=args.indent
    )