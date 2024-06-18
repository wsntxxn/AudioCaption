import json
import argparse
from tqdm import tqdm
from nltk.parse import CoreNLPParser

parser = argparse.ArgumentParser()
parser.add_argument("input_json", type=str)
parser.add_argument("server", type=str, help="Core NLP server, in the form of http://[host]:[port]")
parser.add_argument("parse_result", type=str, choices=["constituent", "pos"])
parser.add_argument("output", type=str)

args = parser.parse_args()

data = json.load(open(args.input_json))["audios"]

parse_result = args.parse_result

if parse_result == "constituent":
    nlp_parser = CoreNLPParser(url=args.server)
else:
    nlp_parser = CoreNLPParser(url=args.server, tagtype="pos")

structure_data = {}

clause_level_labels = ["S", "SBAR", "SBARQ", "SINV", "SQ"]
def get_children(tree):
    result = []
    for subtree in tree:
        if subtree.label() == "ROOT" or subtree.label() in clause_level_labels:
            result += get_children(subtree)
        elif subtree.label() not in ",.":
            result += [subtree.label(),]
    return result


for item in tqdm(data):
    audio_id = item["audio_id"]
    for cap_item in item["captions"]:
        caption = cap_item["caption"]
        tokens = cap_item["tokens"]
        cap_id = cap_item["cap_id"]
        if parse_result == "constituent":
            tree = list(nlp_parser.raw_parse(caption))[0]
            structure = get_children(tree)
        else:
            structure = [item[1] for item in nlp_parser.tag(tokens.split())]
        structure_data[f"{audio_id}_{cap_id}"] = " ".join(structure)

json.dump(structure_data, open(args.output, "w"), indent=4)

