import json
from tqdm import tqdm
import logging
import pickle
from collections import Counter
import argparse

from captioning.datasets.text_tokenizer import DictTokenizer


def ptb_tokenize(data):
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    captions = {}
    for audio_idx in range(len(data)):
        audio_id = data[audio_idx]["audio_id"]
        captions[audio_id] = []
        for cap_idx in range(len(data[audio_idx]["captions"])):
            caption = data[audio_idx]["captions"][cap_idx]["caption"]
            captions[audio_id].append({
                "caption": caption
            })
    tokenizer = PTBTokenizer()
    captions = tokenizer.tokenize(captions)
    for audio_idx in tqdm(range(len(data)), leave=False, ascii=True):
        audio_id = data[audio_idx]["audio_id"]
        for cap_idx in range(len(data[audio_idx]["captions"])):
            tokens = captions[audio_id][cap_idx]
            data[audio_idx]["captions"][cap_idx]["tokens"] = tokens
    return data


def spacy_tokenize(data):
    import spacy
    tokenizer = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    for audio_idx in tqdm(range(len(data)), leave=False, ascii=True):
        captions = data[audio_idx]["captions"]
        for cap_idx in range(len(captions)):
            caption = captions[cap_idx]["caption"]
            doc = tokenizer(caption)
            tokens = " ".join([token.text.lower() for token in doc])
            data[audio_idx]["captions"][cap_idx]["tokens"] = tokens
    return data


def build_tokenizer(args):
    """Build tokenizer from json file with a given threshold to drop all counts < threshold
        The structure of input json file: 
            {
              'audios': [
                {
                  'audio_id': 'xxx',
                  'captions': [
                    { 
                      'caption': 'xxx',
                      'cap_id': 'xxx'
                    }
                  ]
                },
                ...
              ]
            }
    """
    input_json = args.input_json
    data = json.load(open(input_json, "r"))["audios"]
    counter = Counter()
    threshold = args.threshold

    if args.tokenizer == "ptb":
        tokenize_fn = ptb_tokenize
    elif args.tokenizer == "spacy":
        tokenize_fn = spacy_tokenize
    else:
        raise ValueError("Unknown tokenizer: {}".format(args.tokenizer))

    data = tokenize_fn(data)
    json.dump({ "audios": data }, open(input_json, "w"), indent=4)

    for item in tqdm(data, leave=False, ascii=True):
        for cap_item in item["captions"]:
            tokens = cap_item["tokens"]
            counter.update(tokens.split(" "))

    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    tokenizer = DictTokenizer()
    for word in words:
        tokenizer.add_word(word)

    return tokenizer


def main(args):
    output_file = args.output_file
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=logfmt)
    logging.info("Build tokenizer...")
    tokenizer = build_tokenizer(args)
    pickle.dump(tokenizer.state_dict(), open(output_file, "wb"))
    logging.info("Total vocabulary size: {}".format(len(tokenizer)))
    logging.info("Saved vocab to '{}'".format(output_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, required=True,)
    parser.add_argument("--output_file", type=str, required=True,)
    parser.add_argument("--threshold", type=int, default=1)
    parser.add_argument("--tokenizer", type=str, default="ptb")

    # subparsers = parser.add_subparsers(dest="tokenizer")
    #
    # parser_ptb = subparsers.add_parser("ptb")
    #
    # parser_spacy = subparsers.add_parser("spacy")

    args = parser.parse_args()
    main(args)

