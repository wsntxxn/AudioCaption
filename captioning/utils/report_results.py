from pathlib import Path
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--exp_dir", help="experiment path", type=str, required=True)
parser.add_argument("--fname", help="result filename", type=str, required=True)
parser.add_argument("--output", help="output result file", default=None)

args = parser.parse_args()


scores = {}
for path in Path(args.exp_dir).iterdir():
    with open(path / "results" / args.fname, "r") as reader:
        for line in reader.readlines():
            metric, score = line.strip().split(": ")
            score = float(score)
            if metric not in scores:
                scores[metric] = []
            scores[metric].append(score)

if len(scores) == 0:
    print("No experiment directory found, wrong path?")
    exit(1)

with open(args.output, "w") as writer:
    print("Average results: ", file=writer)
    for metric, score in scores.items():
        score = np.array(score)
        mean = np.mean(score)
        std = np.std(score)
        print(f"{metric}: {mean:.3f} (±{std:.3f})", file=writer)
    print("", file=writer)
    print("Best results: ", file=writer)
    for metric, score in scores.items():
        score = np.max(score)
        print(f"{metric}: {score:.3f}", file=writer)
