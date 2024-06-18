from collections import Counter
import json
from pathlib import Path
import pickle

from h5py import File
import fire
from tqdm import tqdm
import stanza
import numpy as np
import pandas as pd


stop_kws = ["sound", "distance", "background", "make", "piece"]
accepted_uposes = ["VERB", "NOUN"]
tagging_th = 0.1
stop_tagging = ["Sound effect", "Background noise"]
tagging_caption_th = 0.3

semantic_weight = 0.5


def cosine_similarity(x, y):
    return np.dot(x, y) / (
        np.linalg.norm(x) * np.linalg.norm(y))


def load_audio_tagging(tagging_label_encoder,
                       tagging_prediction,
                       tagging_embedding):
    label_encoder = pickle.load(open(tagging_label_encoder, "rb"))
    aid_to_tagging = {}
    with File(tagging_prediction, "r") as store:
        for aid, prob in tqdm(store.items(), ascii=True):
            aid_to_tagging[aid] = prob[()]

    tag_to_embed = {}

    with File(tagging_embedding, "r") as store:
        for tag, embedding in tqdm(store.items(), ascii=True):
            tag_to_embed[tag] = embedding[()]

    return {
        "label_encoder": label_encoder,
        "aid_to_tagging": aid_to_tagging,
        "tag_to_embed": tag_to_embed
    }


def load_caption_embedding(caption_embedding):
    key_to_embed = {}

    with File(caption_embedding, "r") as store:
        for key, embedding in tqdm(store.items(), ascii=True):
            key_to_embed[key] = embedding[()]

    return key_to_embed


def get_kw_cands(item, key_to_stanza):
    words = []
    audio_id = item["audio_id"]
    for cap_item in item["captions"]:
        cap_id = cap_item["cap_id"]
        key = f"{audio_id}_{cap_id}"
        sentence = key_to_stanza[key]
        cap_words = set()
        for word in sentence.words:
            upos = word.upos
            if upos in accepted_uposes:
                lemma = word.lemma
                cap_words.add(lemma.lower())
        words.extend(list(cap_words))
    word_to_cnt = Counter(words)
    kw_cands = []
    for word, cnt in word_to_cnt.items():
        if cnt >= len(item["captions"]) / 2 and word not in stop_kws:
            kw_cands.append(word)
    return kw_cands


def get_kw_tagging_cands(audio_id,
                         aid_to_tagging,
                         label_encoder,
                         key_to_embed,
                         tag_to_embed):
    prob = aid_to_tagging[audio_id]
    filtered_indexes = np.where(prob > tagging_th)[0]
    idx_to_score = {}
    for tag_idx in filtered_indexes:
        tag = label_encoder.classes_[tag_idx]
        sims = []
        for idx in range(1, 6):
            key = f"{audio_id}_{idx}"
            sims.append(cosine_similarity(key_to_embed[key], tag_to_embed[tag]))
        sim = np.mean(sims)
        idx_to_score[tag_idx] = sim * semantic_weight + prob[tag_idx] * (
            1 - semantic_weight)

    kw_tagging_cands = []
    for (tag_idx, score) in sorted(idx_to_score.items(),
                                   key=lambda x: x[1],
                                   reverse=True):
        tag = label_encoder.classes_[tag_idx]
        if tag in stop_tagging:
            continue
        if score > tagging_caption_th:
            kw_tagging_cands.append(tag.lower())
        if len(kw_tagging_cands) == 1:
            break
    
    return kw_tagging_cands


def manual_rules(keywords):
    # howl (wind)
    if "howl" in keywords and "wind" in keywords:
        keywords.remove("howl")
        keywords.remove("wind")
        if "howl (wind)" not in keywords:
            keywords.append("howl (wind)")
    # applause
    if "clap" in keywords or "clapping" in keywords:
        if "clap" in keywords:
            keywords.remove("clap")
        if "clapping" in keywords:
            keywords.remove("clapping")
        if "applause" not in keywords:
            keywords.append("applause")
    # plane
    to_remove = []
    has_plane = False
    for x in keywords:
        if "plane" in x:
            has_plane = True
            to_remove.append(x)
    for x in to_remove:
        keywords.remove(x)
    if "plane" not in keywords and has_plane:
        keywords.append("plane")
    
    if "aircraft" in keywords:
        keywords.remove("aircraft")
        if "plane" not in keywords:
            keywords.append("plane")

    # saw
    to_remove = []
    has_saw = False
    for x in keywords:
        if "saw" in x:
            has_saw = True
            to_remove.append(x)
    for x in to_remove:
        keywords.remove(x)
    if "saw" not in keywords and has_saw:
        keywords.append("saw")



def merge_kw_cands(kw_cands, kw_tagging_cands):
    kw_cands_no_dup = []
    for word in kw_cands:
        duplicated = False
        for kw_tagging in kw_tagging_cands:
            for single_word in kw_tagging.replace(",", "").split():
                if word in single_word:
                    duplicated = True
        if not duplicated:
            kw_cands_no_dup.append(word)
    keywords = kw_cands_no_dup + kw_tagging_cands
    manual_rules(keywords)
    return keywords


def main(annotation,
         caption_stanza,
         caption_embedding,
         tagging_prob,
         tagging_label_encoder,
         tagging_embedding,
         output,
         keyword_encoder=None
         ):

    caption_data = json.load(open(annotation))["audios"]

    if not Path(caption_stanza).exists():
        stanza_model = stanza.Pipeline('en', processors='tokenize,pos,lemma')
        key_to_stanza = {}
        for item in tqdm(caption_data, ascii=True):
            audio_id = item["audio_id"]
            for cap_item in item["captions"]:
                caption = cap_item["caption"]
                cap_id = cap_item["cap_id"]
                key = f"{audio_id}_{cap_id}"
                doc = stanza_model(caption)
                key_to_stanza[key] = doc.sentences[0]
        pickle.dump(key_to_stanza, open(caption_stanza, "wb"))
    else:
        key_to_stanza = pickle.load(open(caption_stanza, "rb"))

    tagging_data = load_audio_tagging(tagging_label_encoder,
                                      tagging_prob,
                                      tagging_embedding)
    aid_to_tagging = tagging_data["aid_to_tagging"]
    label_encoder = tagging_data["label_encoder"]
    tag_to_embed = tagging_data["tag_to_embed"]

    key_to_embed = load_caption_embedding(caption_embedding)

    aid_to_kw = {}

    if keyword_encoder is not None:
        keyword_encoder = pickle.load(open(keyword_encoder, "rb"))
        train_keywords = set(keyword_encoder.classes_)

    for item in tqdm(caption_data, ascii=True):
        audio_id = item["audio_id"]
        kw_cands = get_kw_cands(item, key_to_stanza)
        kw_tagging_cands = get_kw_tagging_cands(audio_id, aid_to_tagging,
            label_encoder, key_to_embed, tag_to_embed)
        keywords = merge_kw_cands(kw_cands, kw_tagging_cands)
        if keyword_encoder is not None:
            keywords = filter(lambda x: x in train_keywords, keywords)
        aid_to_kw[audio_id] = "; ".join(keywords)

    aids = []
    kws = []
    for aid, kw in aid_to_kw.items():
        aids.append(aid)
        kws.append(kw)

    output_df = pd.DataFrame({"audio_id": aids, "keywords": kws})
    data_length = output_df[~output_df["keywords"].isin([""])].shape[0]
    print(f"{data_length} data after filtering")
    output_df.to_csv(output, sep="\t", index=False)



if __name__ == "__main__":
    fire.Fire(main)



