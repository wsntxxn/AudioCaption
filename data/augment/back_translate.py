import json
import argparse
from tqdm import trange
from transformers import MarianMTModel, MarianTokenizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device)

en_to_med_model_name = 'Helsinki-NLP/opus-mt-en-ROMANCE'
# en_to_med_tokenizer = MarianTokenizer.from_pretrained(en_to_med_model_name)
en_to_med_tokenizer = MarianTokenizer.from_pretrained("/mnt/lustre/sjtu/home/xnx98/work/AudioTextPretrain/bert_cache/opus_mt_en_ROMANCE")
en_to_med_model = MarianMTModel.from_pretrained(en_to_med_model_name)
en_to_med_model = en_to_med_model.to(device)

med_to_en_model_name = 'Helsinki-NLP/opus-mt-ROMANCE-en'
# med_to_en_tokenizer = MarianTokenizer.from_pretrained(med_to_en_model_name)
med_to_en_tokenizer = MarianTokenizer.from_pretrained("/mnt/lustre/sjtu/home/xnx98/work/AudioTextPretrain/bert_cache/opus_mt_ROMANCE_en")
med_to_en_model = MarianMTModel.from_pretrained(med_to_en_model_name)
med_to_en_model = med_to_en_model.to(device)

def translate(texts, model, tokenizer, language="fr"):
    # Prepare the text data into appropriate format for the model
    template = lambda text: f"{text}" if language == "en" else f">>{language}<< {text}"
    src_texts = [template(text) for text in texts]
    # Tokenize the texts
    tokens = tokenizer(src_texts, return_tensors="pt", padding=True).to(device)
    # Generate translation using model
    translated = model.generate(**tokens)
    # Convert the generated tokens indices back into text
    translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return translated_texts

def back_translate(texts, source_lang="en", target_lang="fr"):
    # Translate from source to target language
    fr_texts = translate(texts, en_to_med_model, en_to_med_tokenizer, 
                         language=target_lang)
    # Translate from target language back to source language
    back_translated_texts = translate(fr_texts, med_to_en_model, med_to_en_tokenizer, 
                                      language=source_lang)
    return back_translated_texts


def process(data, batch_size):
    captions = list(set([caption_item["caption"] for item in data for caption_item in item["captions"]]))
    bt_mapping = {}
    for i in trange(0, len(captions), batch_size):
        texts = captions[i: i + batch_size]
        bt_texts = back_translate(texts)
        for text, bt_text in zip(texts, bt_texts):
            bt_mapping[text] = bt_text
    bt_data = []
    for item in data:
        bt_item = item.copy()
        for caption_item in bt_item["captions"]:
            caption_item["caption"] = bt_mapping[caption_item["caption"]]
            caption_item["cap_id"] = caption_item["cap_id"] + "_backtranslate"
        bt_data.append(bt_item)
    return bt_data


parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("output")
parser.add_argument("--batch_size", type=int, default=32)
args = parser.parse_args()

data = json.load(open(args.input))["audios"]
bt_data = process(data, args.batch_size)
json.dump({"audios": bt_data}, open(args.output, "w"), indent=4)
