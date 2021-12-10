#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import json
import logging

def get_parser():
    parser = argparse.ArgumentParser(
        description="concatenate json files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_jsons", type=str, nargs="+", help="input json files")
    parser.add_argument("output_json", type=str, help="output json file")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()

    # logging info
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=logfmt)

    # make intersection set for utterance keys
    js = {"audios": []}
    for x in args.input_jsons:
        with open(x, encoding="utf-8") as f:
            j = json.load(f)
        logging.info(x + ": has " + str(len(j["audios"])) + " audio clips")
        js["audios"].extend(j["audios"])
    logging.info("new json has " + str(len(js["audios"])) + " audio clips")

    json.dump(
        js,
        open(args.output_json, "w"),
        indent=4
    )
    
