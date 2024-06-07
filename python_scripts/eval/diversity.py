import json
import random
random.seed(1)
import argparse
from functools import partial
from multiprocessing import Pool
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm


def calc_ngram(words, n=2):
    return zip(*[words[i:] for i in range(n)])

def calc_self_bleu(sentences, num_workers):
    pool = Pool(num_workers)
    result = []
    for idx in range(len(sentences)):
        hypothesis = sentences[idx]
        references = [sentences[_] for _ in range(len(sentences)) if _ != idx]
        result.append(pool.apply_async(
            partial(sentence_bleu, smoothing_function=SmoothingFunction().method1),
            args=(references, hypothesis))
        )
    score = 0.0
    cnt = 0
    for i in result:
        score += i.get()
        cnt += 1
    pool.close()
    pool.join()
    return score / cnt

parser = argparse.ArgumentParser()
parser.add_argument("system_output", type=str)
parser.add_argument("train_corpus", type=str)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--system_output_index", type=int, default=None)
parser.add_argument("--diversity_output", type=str, default=None)


args = parser.parse_args()
output = json.load(open(args.system_output))
train_data = json.load(open(args.train_corpus))["audios"]
train_captions = [cap_item["tokens"] for item in train_data
                    for cap_item in item["captions"]]
train_captions = set(train_captions)
system_output_index = args.system_output_index

vocabulary = set()
num_novel_captions = 0
ngrams = [set() for _ in range(2)]
num_total_words = 0
pred_captions = []

if "predictions" in output:
    for cap_item in tqdm(output["predictions"]):
        tokens = cap_item["tokens"]
        if isinstance(tokens, str):
            tokens_list = [tokens]
        elif isinstance(tokens, list) and len(tokens) > 1:
            if system_output_index is not None:
                if system_output_index >= 0:
                    tokens_list = [tokens[system_output_index]]
                else:
                    token_idx = random.randint(0, len(tokens) - 1)
                    tokens_list = [tokens[token_idx]]
            else:
                tokens_list = tokens
        else:
            raise Exception("tokens should be either list[str] or str")

        for tokens in tokens_list:
            if tokens not in train_captions:
                num_novel_captions += 1
            words = tokens.split()

            for word in words:
                vocabulary.add(word)

            for ngram in calc_ngram(words, 1):
                ngrams[0].add(ngram)
            for ngram in calc_ngram(words, 2):
                ngrams[1].add(ngram)
            num_total_words += len(words)

            pred_captions.append(words)

    self_bleu = calc_self_bleu(pred_captions, args.num_workers)

    print(f"Vocabulary size: {len(vocabulary)}")
    novel_percent = num_novel_captions / len(pred_captions)
    print(f"% novel sentences: {novel_percent:.2%}")
    div_1 = len(ngrams[0]) / num_total_words
    print(f"Distinct-1: {div_1:.2g}")
    div_2 = len(ngrams[1]) / num_total_words
    print(f"Distinct-2: {div_2:.2g}")
    print(f"Self-BLEU: {self_bleu:.2g}")

    if args.diversity_output:
        with open(args.diversity_output, "w") as writer:
            print(f"Vocabulary size: {len(vocabulary)}", file=writer)
            print(f"% novel sentences: {novel_percent:.2%}", file=writer)
            print(f"Distinct-1: {div_1:.2g}", file=writer)
            print(f"Distinct-2: {div_2:.2g}", file=writer)
            print(f"Self-BLEU: {self_bleu:.2g}", file=writer)

else:
    for item in tqdm(output["audios"]):
        cap_idx = random.randint(0, len(item["captions"]) - 1)
        cap_item = item["captions"][cap_idx]
        tokens = cap_item["tokens"]
        if tokens not in train_captions:
            num_novel_captions += 1
        words = tokens.split()

        for word in words:
            vocabulary.add(word)

        for ngram in calc_ngram(words, 1):
            ngrams[0].add(ngram)
        for ngram in calc_ngram(words, 2):
            ngrams[1].add(ngram)
        num_total_words += len(words)

        pred_captions.append(words)
    self_bleu = calc_self_bleu(pred_captions, args.num_workers)

    print(f"Vocabulary size: {len(vocabulary)}")
    novel_percent = num_novel_captions / len(pred_captions)
    print(f"% novel sentences: {novel_percent:.2%}")
    div_1 = len(ngrams[0]) / num_total_words
    print(f"Distinct-1: {div_1:.2g}")
    div_2 = len(ngrams[1]) / num_total_words
    print(f"Distinct-2: {div_2:.2g}")
    print(f"Self-BLEU: {self_bleu:.2g}")

    if args.diversity_output:
        with open(args.diversity_output, "w") as writer:
            print(f"Vocabulary size: {len(vocabulary)}", file=writer)
            print(f"% novel sentences: {novel_percent:.2%}", file=writer)
            print(f"Distinct-1: {div_1:.2g}", file=writer)
            print(f"Distinct-2: {div_2:.2g}", file=writer)
            print(f"Self-BLEU: {self_bleu:.2g}", file=writer)
