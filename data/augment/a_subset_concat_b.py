import json
import random
import argparse
from pathlib import Path


def augment(args):
    random.seed(args.seed)
    a_data = json.load(open(args.set_a))["audios"]
    b_data = json.load(open(args.set_b))["audios"]

    a_number = int(len(a_data) * args.a_percent)
    if args.total_number is None:
        args.total_number = len(a_data)
    b_number = args.total_number - a_number
    print("a subset number: ", a_number)
    print("b subset number: ", b_number)

    if Path(args.subset_a_output).exists():
        a_subset = json.load(open(args.subset_a_output))["audios"]
        assert len(a_subset) == a_number
    else:
        a_subset = random.sample(a_data, a_number)
    if b_number > len(b_data):
        b_subset = b_data
    else:
        b_subset = random.sample(b_data, b_number)
    data = a_subset + b_subset
    print("a subset number: ", a_number)
    if not Path(args.subset_a_output).exists():
        json.dump({"audios": a_subset}, open(args.subset_a_output, "w"), indent=4)
    json.dump({"audios": data}, open(args.all_output, "w"), indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("set_a", type=str)
    parser.add_argument("set_b", type=str)
    parser.add_argument("subset_a_output", type=str)
    parser.add_argument("all_output", type=str)
    parser.add_argument("--a_percent", type=float, required=True)
    parser.add_argument("--seed", type=int, default=1, required=False)
    parser.add_argument("--total_number", type=int, default=None, required=False)
    args = parser.parse_args()
    augment(args)
