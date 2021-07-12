# Audio Captioning recipe

This repository provides a recipe for audio captioning with sequence to sequence models: data preprocessing, training, evaluation / prediction.

It supports:
* Models:
  * Audio encoder (most from audio tagging and sound event detection): [CRNN5](https://arxiv.org/abs/2101.07687), [CNN10](https://arxiv.org/abs/1912.10211)
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

We now support [clotho](https://arxiv.org/abs/1910.09387) and [audiocaps](https://www.aclweb.org/anthology/N19-1011/). See details in [data/README.md](data/README.md).

# Training

## Configuration
The training configuration is written in a YAML file and passed to the training script.
All configurations are stored in `config/*.yaml`, where parameters like model type, learning rate, whether to use scheduled sampling can be specified.

## Start training
```bash
python runners/run.py train config/clotho_xe.yaml
```
The training script will use all configurations specified in `config/clotho_xe.yaml`.
They can also be switched by passing `--ARG VALUE`, e.g., if you want to use scheduled sampling, you can run:
```bash
python runners/run.py train config/clotho_xe.yaml --ss True
```

## DCASE2021 onfiguration and training
First download [pre-trained CNN10](https://zenodo.org/record/5090473/files/cnn10_unbalanced.pth) audio encoder:
```bash
mkdir experiments/pretrained_encoder
wget -O experiments/pretrained_encoder/cnn10_unbalanced.pth https://zenodo.org/record/5090473/files/cnn10_unbalanced.pth
```
Then run the training script:
```bash
python runners/run.py train config/dcase2021_xe.yaml
```
After the XE training finishes, continue training the model by scst fine-tuning:
```bash
python runners/run_scst.py train config/dcase2021_scst.yaml
```


# Evaluation

Evaluation is done by running function `evaluate` in `runners/run.py`. For example:
```bash
export EXP_PATH=experiments/***
python runners/run.py \
    evaluate \
    $EXP_PATH \
    data/clotho_v2/eval/lms.csv \
    data/clotho_v2/eval/text.json
```
To use beam search (for example with a beam size of 3), use:
```bash
python runners/run.py \
    evaluate \
    $EXP_PATH \
    data/clotho_v2/eval/lms.csv \
    data/clotho_v2/eval/text.json \
    --method beam \
    --beam-size 3
```

Standard captioning metrics (BLEU@1-4, ROUGE-L, CIDEr, METEOR and SPICE) will be calculated.



