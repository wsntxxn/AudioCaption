from pathlib import Path
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("exp_path", help="parent experiment path, with several child directory trained under different seeds")
parser.add_argument("--output", help="output result file", default=None)

args = parser.parse_args()

exp_path = Path(args.exp_path)
if args.output is None:
    args.output = exp_path / "scores.txt"

scores = {}
for path in exp_path.glob("seed_*"):
    with open(path / "scores_beam_3.txt", "r") as reader:
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
    for metric, score in scores.items():
        score = np.array(score)
        mean = np.mean(score)
        std = np.std(score)
        print(f"{metric}: {mean:.3f} (±{std:.3f})", file=writer)

