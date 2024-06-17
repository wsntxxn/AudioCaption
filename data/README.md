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
- Run the preprocessing script in this directory, assuming the output directory is `$CLOTHO_DIR` (e.g. `clotho_v2`):
  ```bash
  $ CLOTHO_DIR=clotho_v2
  $ python create_dataset/prepare_clotho.py $CLOTHO_ROOT $CLOTHO_DIR
  ```

### Pack waveform data to hdf5
We pack audio files to a single HDF5 to avoid reading too many files (which may yield low read performance on some machines):
```bash
$ python create_dataset/pack_waveform.py $CLOTHO_DIR/dev/wav.csv --output $CLOTHO_DIR/dev/waveform.h5
$ python create_dataset/pack_waveform.py $CLOTHO_DIR/val/wav.csv --output $CLOTHO_DIR/val/waveform.h5
$ python create_dataset/pack_waveform.py $CLOTHO_DIR/eval/wav.csv --output $CLOTHO_DIR/eval/waveform.h5
```

### Tokenize captions and build vocabulary
```bash
$ python ../python_scripts/utils/build_custom_tokenizer.py \
    --input_json $CLOTHO_DIR/dev/text.json \
    --output_file $CLOTHO_DIR/dev/vocab.pkl
```

## AudioCaps

AudioCaps is built on a subset of AudioSet so we prepare the dataset using `audioset_wav_csv` of the downloaded AudioSet. It is a tab separated table looks like the following:

|audio_id|file_name|
|----:|-----:|
|Y_f_i28HxMDA.wav|/path/to/audioset/Y_f_i28HxMDA.wav|
|Ya67ihfUaKUc.wav|/path/to/audioset/Ya67ihfUaKUc.wav|
|...  |...   |

We assume `audioset_wav_csv` is already prepared. It does not have to contain the full AudioSet samples but AudioCaps samples must be included. Also, `audio_id` must be in the form of `Y[youtube_id].wav`.

Then download the caption files from [AudioCaps](https://github.com/cdjkim/audiocaps). Assume the caption files are placed in `$AUDIOCAPS_ANNOTATION` and the processed data is in `AUDIOCAPS_DIR`.
```bash
$ python create_dataset/prepare_audiocaps.py $AUDIOCAPS_ANNOTATION $AUDIOSET_WAV_CSV --output_path $AUDIOCAPS_DIR --audio_link_path $AUDIOCAPS_DIR/audio_links
```
Similarly, pack waveform and build vocabulary:
```bash
$ python create_dataset/pack_waveform.py $AUDIOCAPS_DIR/train/wav.csv --output $AUDIOCAPS_DIR/train/waveform.h5
$ python create_dataset/pack_waveform.py $AUDIOCAPS_DIR/val/wav.csv --output $AUDIOCAPS_DIR/val/waveform.h5
$ python create_dataset/pack_waveform.py $AUDIOCAPS_DIR/test/wav.csv --output $AUDIOCAPS_DIR/test/waveform.h5
$ python ../python_scripts/utils/build_custom_tokenizer.py \
    --input_json $AUDIOCAPS_DIR/train/text.json \
    --output_file $AUDIOCAPS_DIR/train/vocab.pkl
```
