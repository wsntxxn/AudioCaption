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
$ git clone https://github.com/wsntxxn/AudioCaption
```
Then install the required packages and the repo for convenience:
```bash
$ pip install -r requirements.txt
$ pip install -e .
```

# Data preprocessing

We now support [clotho](https://arxiv.org/abs/1910.09387) and [audiocaps](https://www.aclweb.org/anthology/N19-1011/). See details in [data/README.md](data/README.md).

# Training

## Configuration
The training configuration is written in a YAML file and passed to the training script.
All configurations are stored in `config/*.yaml`, where parameters like model type, learning rate, whether to use scheduled sampling can be specified.

## Start training
```bash
$ python captioning/ignite_runners/run.py train \
    --config config/clotho_xe.yaml
```
The training script will use all configurations specified in `config/clotho_xe.yaml`.
They can also be switched by passing `--ARG VALUE`, e.g., if you want to use scheduled sampling, you can run:
```bash
$ python captioning/ignite_runners/run.py train \
    --config config/clotho_xe.yaml \
    --ss True
```

## DCASE2021 onfiguration and training
First download [pre-trained CNN10](https://zenodo.org/record/5090473/files/cnn10_unbalanced.pth) audio encoder:
```bash
$ mkdir experiments/pretrained_encoder
$ wget -O experiments/pretrained_encoder/cnn10_unbalanced.pth https://zenodo.org/record/5090473/files/cnn10_unbalanced.pth
```
Then run the training script:
```bash
$ python captioning/ignite_runners/run.py train \
    --config config/dcase2021_xe.yaml
```
After the XE training finishes, continue training the model by scst fine-tuning:
```bash
$ python captioning/ignite_runners/run_scst.py train \
    --config config/dcase2021_scst.yaml
```


# Evaluation

Evaluation is done by running function `evaluate` in `runners/run.py`. For example:
```bash
$ export EXP_PATH=experiments/***
$ python captioning/ignite_runners/run.py evaluate \
    --experiment_path $EXP_PATH \
    --task clotho \
    --raw_feat_csv data/clotho_v2/eval/lms.csv \
    --fc_feat_csv data/clotho_v2/eval/lms.csv \
    --attn_feat_csv data/clotho_v2/eval/lms.csv \
    --caption_file data/clotho_v2/eval/text.json
```
To use beam search (for example with a beam size of 3), use:
```bash
$ python captioning/ignite_runners/run.py evaluate \
    --experiment_path $EXP_PATH \
    --task clotho \
    --raw_feat_csv data/clotho_v2/eval/lms.csv \
    --fc_feat_csv data/clotho_v2/eval/lms.csv \
    --attn_feat_csv data/clotho_v2/eval/lms.csv \    
    --caption_file data/clotho_v2/eval/text.json \
    --method beam \
    --beam-size 3
```

Standard captioning metrics (BLEU@1-4, ROUGE-L, CIDEr, METEOR and SPICE) will be calculated.

# Citation
If you find this repo useful, please consider citing
```BibTeX
@inproceedings{xu2020crnn,
    author={Xu, Xuenan and Dinkel, Heinrich and Wu, Mengyue and Yu, Kai},
    title={A CRNN-GRU Based Reinforcement Learning Approach to Audio Captioning},
    booktitle={Proceedings of the Detection and Classification of Acoustic Scenes and Events Workshop},
    year={2020},
    pages={225-229}
}
```
```BibTeX
@techreport{xu2021sjtu,
    author={Xu, Xuenan and Xie, Zeyu and Wu, Mengyue and Yu, Kai},
    title={The {SJTU} System for {DCASE2021} Challenge Task 6: Audio Captioning Based on Encoder Pre-training and Reinforcement Learning},
    institution={DCASE2021 Challenge},
    year={2021}
}
```

