# Audio Captioning Recipe

This repository provides a recipe for audio captioning with sequence to sequence models: data preprocessing, training, evaluation and inference.
Specifically in this branch, we provide the main code of our DCASE2022 submission.  

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

## Contrastive Audio-text Pre-training
The first step is audio-text pre-training using contrastive learning. The code is in [another repo](https://github.com/wsntxxn/DCASE2022T6_CLAP). The pre-trained audio-text retrieval model [checkpoint](https://github.com/wsntxxn/AudioCaption/releases/download/v0.0.2/contrastive_pretrain_cnn14_bertm.pth) is provided for captioning training in the following step.

## Captioning Training
The training configuration is written in a YAML file and passed to the training script. Examples are in `eg_configs`.

For example, train a model with Cnn14_Rnn encoder and Transformer decoder on Clotho:
```bash
$ python python_scripts/train_eval/run.py train eg_configs/clotho_v2/waveform/cnn14rnn_trm.yaml
```

# Evaluation
Assume the experiment directory is `$EXP_PATH`. Evaluation under the configuration in `eg_configs/clotho_v2/waveform/test.yaml`:
```bash
$ python python_scripts/train_eval/run.py evaluate $EXP_PATH eg_configs/clotho_v2/waveform/test.yaml
```

# Inference
Inference using the checkpoint `$CKPT`:
```bash
$ python python_scripts/inference/inference.py \
    --input input.wav 
    --output output.json
    --checkpoint $CKPT
```

## Ensemble
To ensemble several models for inference, especially in challenges, the script with an sample configuration `eg_configs/dcase2022/ensemble/config.yaml` is:
```bash
$ python python_scripts/train_eval/ensemble.py evaluate eg_configs/dcase2022/ensemble.yaml
```

## Using off-the-shelf models
We release the models trained on Clotho and AudioCaps for easy use. They use contrastive pre-trained feature extractor:
```bash
$ mkdir pretrained_feature_extractors
$ wget https://github.com/wsntxxn/AudioCaption/releases/download/v0.0.2/contrastive_pretrain_cnn14_bertm.pth -O pretrained_feature_extractors/contrastive_pretrain_cnn14_bertm.pth
```
AudioCaps:
```bash
$ wget https://github.com/wsntxxn/AudioCaption/releases/download/v0.0.2/audiocaps_cntrstv_cnn14rnn_trm.zip
$ unzip audiocaps_cntrstv_cnn14rnn_trm.zip
$ python python_scripts/inference/inference.py \
    --input input.wav \
    --output output.json \
    --checkpoint audiocaps_cntrstv_cnn14rnn_trm/swa.pth
```
Clotho:
```bash
$ wget https://github.com/wsntxxn/AudioCaption/releases/download/v0.0.2/clotho_cntrstv_cnn14rnn_trm.zip
$ unzip clotho_cntrstv_cnn14rnn_trm.zip
$ python python_scripts/inference/inference.py \
    --input input.wav \
    --output output.json \
    --checkpoint clotho_cntrstv_cnn14rnn_trm/swa.pth
```

If you find these models useful, please cite our technical report:

```BibTeX
@techreport{xu2022sjtu,
    author={Xu, Xuenan and Xie, Zeyu and Wu, Mengyue and Yu, Kai},
    title={The SJTU System for DCASE2022 Challenge Task 6: Audio Captioning with Audio-Text Retrieval Pre-training},
    institution={DCASE2022 Challenge},
    year={2022}
}
```