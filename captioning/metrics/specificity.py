import json
import argparse
import pandas as pd


def main(system_output,
         word_specificity):
    predictions = json.load(open(system_output))["predictions"]
    word_specificity_df = pd.read_csv(word_specificity, sep="\t")
    word_to_specificity = dict(zip(word_specificity_df["word"], 
                                   word_specificity_df["specificity"]))

    specificities = []
    for prediction in predictions:
        caption = prediction["tokens"]
        tokens = caption.split()
        specificity = 0
        for word in tokens:
            specificity += word_to_specificity[word]
        specificities.append(specificity)
    
    predition_specificity = sum(specificities) / len(specificities)
    print(f"Specificity: {predition_specificity:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("system_output")
    parser.add_argument("word_specificity")
    parser.add_argument("--cores", type=int, default=4)

    args = parser.parse_args()
    main(args.system_output, args.word_specificity)
