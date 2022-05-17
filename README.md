# Audio Captioning recipe

This repository provides a recipe for audio captioning with sequence to sequence models: data preprocessing, training, evaluation and prediction.

# Install

Checkout this repository and install the required packages:
```bash
git clone https://github.com/wsntxxn/AudioCaption
pip install -r requirements.txt
```

# Data preprocessing

We now support [clotho](https://arxiv.org/abs/1910.09387) and [audiocaps](https://www.aclweb.org/anthology/N19-1011/). See details in [data/README.md](data/README.md).

# Training

## Configuration
The training configuration is written in a YAML file and passed to the training script. All configurations are stored in `config/*.yaml`.

## Start training
```bash
python captioning/runners/run.py train config/clotho_xe.yaml
```
It will use all configurations specified in `config/clotho_xe.yaml`.
The hyper-parameters can be covered by passing `--ARG VALUE`, e.g., if you want to use scheduled sampling, you can run:
```bash
python runners/run.py train config/clotho_xe.yaml --ss True
```


# Evaluation

```bash
export EXP_PATH=experiments/***
python captioning/runners/run.py \
    evaluate \
    $EXP_PATH \
    data/clotho_v2/eval/lms.csv \
    data/clotho_v2/eval/text.json \
    --method beam \
    --beam_size 3
```
It will calculates standard metrics.



