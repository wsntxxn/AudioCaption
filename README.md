# Audio Captioning recipe

This repository provides a recipe for audio captioning with sequence to sequence models: data preprocessing, training, evaluation and inference.

# Install

Checkout this repository and install the required packages:
```bash
$ git clone https://github.com/wsntxxn/AudioCaption
$ cd AudioCaption
$ pip install -r requirements.txt
```
Install the repository as a package:
```bash
$ pip install -e .
```

# Data preprocessing

We now support [Clotho](https://arxiv.org/abs/1910.09387) and [AudioCaps](https://www.aclweb.org/anthology/N19-1011/). See details in [data/README.md](data/README.md).

# Training

## Configuration
The training configuration is written in a YAML file and passed to the training script. Examples are in `configs`.

## Start training
For example, train a Cnn14_Rnn-Transformer model on Clohto:
```bash
$ python captioning/pytorch_runners/run.py train configs/clotho_v2/waveform/cnn14rnn_trm.yaml
```

# Evaluation
Assume the experiment directory is `$EXP_PATH`. Evaluation under the configuration in `configs/clotho_v2/waveform/test.yaml`:
```bash
$ python captioning/pytorch_runners/run.py evaluate $EXP_PATH configs/clotho_v2/waveform/test.yaml
```

# Inference
Using the trained model (checkpoint in `$CKPT`) to inference on new audio files:
```bash
$ python captioning/pytorch_runners/inference_waveform.py test.wav test.json $CKPT
```
