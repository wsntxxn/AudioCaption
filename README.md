# AudioCaption recipe

This repository provides a recipe for audio captioning with sequence to sequence models: data preprocessing, training, evaluation / prediction.

It supports:
* Models:
  * Audio encoder (most are from Sound Event Detection research): CRNN, CNN10
  * Text decoder: RNN, RNN with attention, Transformer
* Training methods:
  * Vanilla cross entropy (XE) training
  * [Scheduled sampling](https://arxiv.org/abs/1506.03099)
  * XE with [sequence loss training](http://arxiv.org/abs/1905.13448)
  * Self-critical sequence training ([scst](https://arxiv.org/abs/1612.00563))
* Evaluation:
  * Beam search
  * Test time ensemble

# Install

In order to successfully run the audio captioning recipe, you need to install the prerequisite packages and frameworks.

First checkout this repository:
```bash
git clone --recursive https://github.com/wsntxxn/AudioCaption
```
Then install the required packages:
```bash
pip install -r requirements.txt
```

# Data preprocessing

Current public audio captioning datasets include [clotho](https://arxiv.org/abs/1910.09387), [audiocaps](https://www.aclweb.org/anthology/N19-1011/) and [Audio Caption](https://arxiv.org/abs/1902.09254).
Since different datasets support data in different formats, we do not provide specific preprocessing recipe for each dataset.
However, you can use scripts in `utils` to do the preprocessing.
For audio input, `utils/featextract.py` extracts features from raw wave. 
For text output, `utils/build_vocab.py` builds the vocabulary from caption corpus.
See the documentation and comments in `utils` for details.

# Training

## Prepare data

For training, the following files are needed:
* feature_file: an HDF5 file storing audio features.
* caption_file: a json file providing caption labels for each audio, where two columns must be provided: `key` (corresponding to key in HDF5 file) and `tokens` (tokenized caption). By default, this file contains the whole development dataset. You can also provide `caption_file_train` and `caption_file_val` with your own train/val split.
* vocab_file: vocabulary built from the caption corpus.

## Configuration
The training configuration is written in a YAML file and passed to the training script.
All configurations are stored in `config/*.yaml`, where parameters like model type, learning rate, whether to use scheduled sampling can be specified.

## Start training
```bash
python runners/run.py train config/xe.yaml
```
The training script will use all configurations specified in `config/xe.yaml`.
The training logs and model checkpoints will be stored in `OUTPUTPATH/MODEL/TIMESTAMP`.
They can also be switched by passing `--ARG VALUE`, e.g., if you want to use scheduled sampling, you can run:
```bash
python runners/run.py train config/xe.yaml --ss True
```

# Evaluation

Evaluation is done by running function `evaluate` in `runners/run.py`. For example:
```bash
export EXP_PATH=experiments/***
python runners/run.py \
    evaluate \
    $EXP_PATH \
    data/clotho/logmel.hdf5 \
    data/clotho/logmel_eval.scp \
    data/clotho/eval.json
```
To use beam search (for example with a beam size of 3), use:
```bash
python runners/run.py \
    evaluate \
    $EXP_PATH \
    data/clotho/logmel.hdf5 \
    data/clotho/logmel_eval.scp \
    data/clotho/eval.json \
    --method beam \
    --beam-size 3
```

Standard captioning metrics (BLEU@1-4, ROUGE-L, CIDEr, METEOR and SPICE) will be calculated.



