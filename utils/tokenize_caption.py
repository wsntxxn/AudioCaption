import json
from tqdm import tqdm
import re
import fire


def tokenize_caption(input_json: str,
                     keep_punctuation: bool = False,
                     host_address: str = None,
                     character_level: bool = False,
                     zh: bool = True ):
    """Build vocabulary from csv file with a given threshold to drop all counts < threshold

    Args:
        input_json(string): Preprossessed json file. Structure like this: 
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
        threshold (int): Threshold to drop all words with counts < threshold
        keep_punctuation (bool): Includes or excludes punctuation.

    Returns:
        vocab (Vocab): Object with the processed vocabulary
"""
    data = json.load(open(input_json, "r"))["audios"]
    
    if zh:
        from nltk.parse.corenlp import CoreNLPParser
        from zhon.hanzi import punctuation
        parser = CoreNLPParser(host_address)
        for audio_idx in tqdm(range(len(data)), leave=False, ascii=True):
            for cap_idx in range(len(data[audio_idx]["captions"])):
                caption = data[audio_idx]["captions"][cap_idx]["caption"]
                # Remove all punctuations
                if not keep_punctuation:
                    caption = re.sub("[{}]".format(punctuation), "", caption)
                if character_level:
                    tokens = list(caption)
                else:
                    tokens = list(parser.tokenize(caption))
                data[audio_idx]["captions"][cap_idx]["tokens"] = " ".join(tokens)
    else:
        punctuation = ',.():;?!"\''
        for audio_idx in tqdm(range(len(data)), leave=False, ascii=True):
            for cap_idx in range(len(data[audio_idx]["captions"])):
                caption = data[audio_idx]["captions"][cap_idx]["caption"].lower()
                # Remove all punctuations
                if not keep_punctuation:
                    caption = re.sub("[{}]".format(punctuation), " ", caption)
                caption = re.sub(" +", " ", caption)
                if character_level:
                    tokens = list(caption)
                else:
                    tokens = caption.split()
                data[audio_idx]["captions"][cap_idx]["tokens"] = " ".join(tokens)

    json.dump({ "audios": data }, open(input_json, "w"), indent=4)


if __name__ == "__main__":
    fire.Fire(tokenize_caption)
