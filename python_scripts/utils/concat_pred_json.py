#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import json


def get_parser():
    parser = argparse.ArgumentParser(
        description="concatenate prediction json files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_jsons", type=str, nargs="+", help="input json files")
    parser.add_argument("output_json", type=str, help="output json file")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()

    predictions = {}
    for x in args.input_jsons:
        with open(x, encoding="utf-8") as f:
            j = json.load(f)
        for item in j["predictions"]:
            filename = item["filename"]
            tokens = item["tokens"]
            if isinstance(tokens, str):
                tokens = [tokens]
            if filename not in predictions:
                predictions[filename] = tokens
            else:
                predictions[filename].extend(tokens)

    data = []
    for filename, tokens in predictions.items():
        data.append({"filename": filename, "tokens": tokens})

    json.dump(
        { "predictions": data },
        open(args.output_json, "w"),
        indent=4
    )