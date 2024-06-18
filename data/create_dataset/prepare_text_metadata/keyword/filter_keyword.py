from collections import Counter
import pickle

import numpy as np
import pandas as pd
import fire
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer


def main(input_file,
         threshold,
         output_file,
         output_encoder,
         keyword_sep="; "):
    input_df = pd.read_csv(input_file, sep="\t")
    words = []
    for _, row in input_df.dropna().iterrows():
        kws = row["keywords"].split(keyword_sep)
        words.extend(kws)
    word_to_cnt = Counter(words)

    cnt_df = pd.DataFrame(word_to_cnt.items(), columns=["word", "count"])
    filtered_words = cnt_df[cnt_df["count"] >= threshold]["word"].values

    output_df = []
    filtered_words = set(filtered_words)
    pbar = tqdm(total=input_df.shape[0], ascii=True)
    for _, row in input_df.iterrows():
        if not isinstance(row["keywords"], str):
            output_df.append({
                "audio_id": row["audio_id"],
                "keywords": ""
            })
        else:
            kws = row["keywords"].split(keyword_sep)
            filtered_kws = [word for word in kws if word in filtered_words]
            filtered_kws = keyword_sep.join(filtered_kws)
            output_df.append({
                "audio_id": row["audio_id"],
                "keywords": filtered_kws
            })
        pbar.update()
    pbar.close()
    output_df = pd.DataFrame(output_df)
    data_length = output_df[~output_df["keywords"].isin([""])].shape[0]
    print(f"{len(filtered_words)} keyword after filtering")
    print(f"{data_length} data after filtering")
    output_df.to_csv(output_file, sep="\t", index=False)

    label_array = output_df[~output_df["keywords"].isin([""])][
        "keywords"].str.split(keyword_sep).values.tolist()
    label_encoder = MultiLabelBinarizer()
    label_encoder.fit(label_array)
    pickle.dump(label_encoder, open(output_encoder, "wb"))
    



if __name__ == "__main__":
    fire.Fire(main)



