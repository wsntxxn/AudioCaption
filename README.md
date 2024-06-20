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
PTB tokenizer and SPICE evaluation also requires java, which can be installed by conda
```bash
$ conda install bioconda::java-jdk
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
where the CNN14 is initialized by the [PANNs checkpoint](https://zenodo.org/records/3987831/files/Cnn14_mAP%3D0.431.pth), which should be manually downloaded to `experiments/pretrained_encoder/Cnn14_mAP=0.431.pth`.

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

## Hugging Face🤗 usage

For convenient usage with Hugging Face, we provide the corresponding [wrapper script](captioning/models/hf_wrapper.py). It contains pure PyTorch implementation of captioning models. You can use it without installing this repository:
```bash
wget https://github.com/wsntxxn/AudioCaption/raw/master/captioning/models/hf_wrapper.py -O hf_captioning.py
```

### Lightweight EffB2-Transformer model
Use the lightweight EffB2-Transformer model for fast inference:
```python
import torch
from transformers import PreTrainedTokenizerFast
import torchaudio
from hf_captioning import Effb2TrmCaptioningModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# use the model trained on AudioCaps
model = Effb2TrmCaptioningModel.from_pretrained(
    "wsntxxn/effb2-trm-audio-captioning",
).to(device)
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "wsntxxn/audiocaps-simple-tokenizer"
)

# inference on a single audio clip
wav, sr = torchaudio.load("/path/to/file.wav")
wav = torchaudio.functional.resample(wav, sr, model.config.sample_rate)
if wav.size(0) > 1:
    wav = wav.mean(0).unsqueeze(0)

with torch.no_grad():
    word_idxs = model(
        audio=wav,
        audio_length=[wav.size(1)],
    )

caption = tokenizer.decode(word_idxs[0], skip_special_tokens=True)
print(caption)

# inference on a batch
wav1, sr1 = torchaudio.load("/path/to/file1.wav")
wav1 = torchaudio.functional.resample(wav1, sr1, model.config.sample_rate)
wav1 = wav1.mean(0) if wav1.size(0) > 1 else wav1[0]

wav2, sr2 = torchaudio.load("/path/to/file2.wav")
wav2 = torchaudio.functional.resample(wav2, sr2, model.config.sample_rate)
wav2 = wav2.mean(0) if wav2.size(0) > 1 else wav2[0]

wav_batch = torch.nn.utils.rnn.pad_sequence([wav1, wav2], batch_first=True)

with torch.no_grad():
    word_idxs = model(
        audio=wav_batch,
        audio_length=[wav1.size(0), wav2.size(0)],
    )

captions = tokenizer.batch_decode(word_idxs, skip_special_tokens=True)
print(captions)
```

Alternatively, you can use the model trained on Clotho:
```python
model = Effb2TrmCaptioningModel.from_pretrained(
    "wsntxxn/effb2-trm-clotho-captioning",
).to(device)
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "wsntxxn/clotho-simple-tokenizer"
)
```

### Temporal-sensitive and controllable model
Use the temporal-enhanced captioning model for captioning with specific (simultaneous / sequential) temporal relationship description:
```python
from hf_captioning import Cnn14RnnTempAttnGruModel

model = Cnn14RnnTempAttnGruModel.from_pretrained(
    "wsntxxn/audiocaps-temporal-cnn14rnn-gru",
).to(device)
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "wsntxxn/audiocaps-simple-tokenizer"
)

wav, sr = torchaudio.load("/path/to/file.wav")
wav = torchaudio.functional.resample(wav, sr, model.config.sample_rate)
if wav.size(0) > 1:
    wav = wav.mean(0).unsqueeze(0)

with torch.no_grad():
    word_idxs = model(
        audio=wav,
        audio_length=[wav.size(1)],
        temporal_tag=[2], # desribe "sequential" if there are sequential events, otherwise use the most complex relationship
    )

caption = tokenizer.decode(word_idxs[0], skip_special_tokens=True)
print(caption)
```
The temporal tag is defined as:
|temporal tag|Definition|
|----:|-----:|
|0|Only 1 Event|
|1|Simultaneous Events|
|2|Sequential Events|
|3|More Complex Events|

From 0 to 3, the relationship indicated by the tag becomes more and more complex.
The model will infer the tag automatically.
If `temporal_tag` is not provided as the input, the model will use the inferred tag.
Otherwise, the model will try to follow the input tag if possible. 

If you find the temporal model useful, please cite this paper:
```BibTeX
@inproceedings{xie2023enhance,
    author = {Zeyu Xie and Xuenan Xu and Mengyue Wu and Kai Yu},
    title = {Enhance Temporal Relations in Audio Captioning with Sound Event Detection},
    year = 2023,
    booktitle = {Proc. INTERSPEECH},
    pages = {4179--4183},
}
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
