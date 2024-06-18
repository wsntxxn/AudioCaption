# Audio Captioning Recipe

This repository provides a simple and easy-to-use recipe for audio captioning: data pre-processing, training, evaluation and inference.

Since we refactor our code several times, codebase of previous challenges are maintained in separate branches: [DCASE2021](https://github.com/wsntxxn/AudioCaption/tree/dcase2021) and [DCASE2022](https://github.com/wsntxxn/AudioCaption/tree/dcase2022). Please check these branches for reference to challenge submissions.

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

# Data pre-processing

We now support [Clotho](https://arxiv.org/abs/1910.09387) and [AudioCaps](https://www.aclweb.org/anthology/N19-1011/). See details in [data/README.md](data/README.md).

# Training

The training configuration is written in a YAML file and passed to the training script. Examples are in `eg_configs`.

For example, train a model with Cnn14_Rnn encoder and Transformer decoder on Clotho:
```bash
$ python python_scripts/train_eval/run.py train \
    --config eg_configs/clotho_v2/waveform/cnn14rnn_trm.yaml
```

# Evaluation
Assume the experiment directory is `$EXP_PATH`. Evaluation under the configuration in `eg_configs/clotho_v2/waveform/test.yaml`:
```bash
$ python python_scripts/train_eval/run.py evaluate \
    --experiment_path $EXP_PATH \
    --eval_config eg_configs/clotho_v2/waveform/test.yaml
```

# Inference
Inference using the checkpoint `$CKPT`:
```bash
$ python python_scripts/inference/inference.py \
    --input input.wav 
    --output output.json
    --checkpoint $CKPT
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
    title={The {SJTU} System for {DCASE2022} Challenge Task 6: Audio Captioning with Audio-Text Retrieval Pre-training},
    institution={DCASE2022 Challenge},
    year={2022}
}
```

# Demo
The code to create a demo interface using gradio is provided:
```bash
$ python demo.py --ckpt $EXP_PATH/swa.pth
```
Then the demo is running on http://localhost:7860.

# Related Papers
The following papers are related to this repository:
* [A CRNN-GRU Based Reinforcement Learning Approach to Audio Captioning](https://dcase.community/documents/workshop2020/proceedings/DCASE2020Workshop_Xu_83.pdf) for [SCST training](https://github.com/wsntxxn/AudioCaption/blob/dcase2022/eg_configs/dcase2022/cnn14rnn_2trm_scst.yaml).
* [Investigating Local and Global Information for Automated Audio Captioning with Transfer Learning](https://ieeexplore.ieee.org/abstract/document/9413982) for [DCASE2021 submission](https://github.com/wsntxxn/AudioCaption/tree/dcase2021).
* [Diversity-Controllable and Accurate Audio Captioning Based on Neural Condition](https://ieeexplore.ieee.org/document/9746834) for [diversity-controllable captioning](https://github.com/wsntxxn/AudioCaption/blob/dcase2021/captioning/ignite_runners/run_condition_adverse.py).
