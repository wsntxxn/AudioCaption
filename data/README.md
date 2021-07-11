# Prepare data

## Clotho

### Download raw data files and preprocess

- Download [clotho v1](https://zenodo.org/record/3490684#.YOmxohMzY6E) or [clotho v2](https://zenodo.org/record/4783391#.YOmxthMzY6E). Since clotho v1 itself is a subset of clotho v2, here take clotho v2 as an example.
- Unextract the files, put them in a directory, denoted as `$CLOTHO_ROOT`, make the structure like this:
  ```
  [CLOTHO_ROOT]
  ├── annotation
  │   ├── dev.csv
  │   ├── val.csv
  │   └── eval.csv
  ├── audio
  │   ├── dev
  │   ├── val
  │   ├── eval
  │   └── test
  └── metadata
      ├── dev.csv
      ├── val.csv
      └── eval.csv
  ```
- Run the preprocessing script, assuming the output directory is `$OUTPUT_PATH` (e.g. `data/clotho_v2`):
  ```bash
  OUTPUT_PATH=data/clotho_v2
  python data/prepare_clotho.py $CLOTHO_ROOT $OUTPUT_PATH
  ```

### Extract log mel spectrogram feature (40ms window length and 20ms window shift)
```bash
for SPLIT in dev val eval; 
  do python data/extract_feature.py $OUTPUT_PATH/$SPLIT/wav.csv $OUTPUT_PATH/$SPLIT/lms.h5 $OUTPUT_PATH/$SPLIT/lms.csv lms -win_length 1764 -hop_length 882 -n_mels 64; 
done
```

### Tokenize captions and build vocabulary
```bash
python utils/build_vocab.py $OUTPUT_PATH/dev/text.json $OUTPUT_PATH/dev/vocab.pkl --zh False
```

### (Optional, for DCASE2021) Merge development and validation subset
In DCASE2021 task 6 challenge, we merge the official development and validation sets and redo the train / validation split to get more data for training. 
```bash
mkdir $OUTPUT_PATH/dev_val
python utils/concat_csv.py $OUTPUT_PATH/dev/lms.csv $OUTPUT_PATH/val/lms.csv $OUTPUT_PATH/dev_val/lms.csv
python utils/concat_json.py $OUTPUT_PATH/dev/text.json $OUTPUT_PATH/val/text.json $OUTPUT_PATH/dev_val/text.json
python utils/build_vocab.py $OUTPUT_PATH/dev_val/text.json $OUTPUT_PATH/dev_val/vocab.pkl --zh False
```
