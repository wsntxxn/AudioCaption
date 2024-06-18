import pickle
import json

import pandas as pd
import fire
from tqdm import tqdm

from extract_keyword import manual_rules, accepted_uposes

# 找出 audio keywords 中属于 AudioSet tags 的，保留
# 找出 noun 和 verb，在 vocabulary 中的
# manual rules

def get_kw_cands(sentence):
    kw_cands = []
    for word in sentence.words:
        upos = word.upos
        lemma = word.lemma.lower()
        if upos in accepted_uposes:
            kw_cands.append(lemma)
    return kw_cands


def main(annotation,
         audio_keyword_file,
         caption_stanza,
         tagging_label_encoder,
         keyword_encoder,
         output):
    caption_data = json.load(open(annotation))["audios"]
    key_to_stanza = pickle.load(open(caption_stanza, "rb"))
    tagging_label_encoder = pickle.load(open(tagging_label_encoder, "rb"))
    audio_keyword_df = pd.read_csv(audio_keyword_file, sep="\t")
    audio_keyword_df.fillna("", inplace=True)
    aid_to_kw = dict(zip(audio_keyword_df["audio_id"], audio_keyword_df["keywords"]))
    keyword_encoder = pickle.load(open(keyword_encoder, "rb"))
    tagging_set = set([x.lower() for x in tagging_label_encoder.classes_])
    keyword_set = set(keyword_encoder.classes_)

    key_to_kw = {}
    for item in tqdm(caption_data, ascii=True):
        aid = item["audio_id"]
        audio_kws = aid_to_kw[aid].split("; ")
        tagging_kws = []
        for keyword in audio_kws:
            if keyword in tagging_set:
                tagging_kws.append(keyword)

        for cap_item in item["captions"]:
            key = f"{aid}_{cap_item['cap_id']}"
            sentence = key_to_stanza[key]
            kw_cands = get_kw_cands(sentence)
            keywords = list(set(tagging_kws + kw_cands))
            manual_rules(keywords)
            keywords = [x for x in keywords if x in keyword_set]
            key_to_kw[key] = "; ".join(keywords)
    
    output_df = pd.DataFrame(key_to_kw.items(), columns=["cap_id", "keywords"])
    output_df.to_csv(output, sep="\t", index=False)


if __name__ == "__main__":
    fire.Fire(main)
