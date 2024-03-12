from pathlib import Path
import argparse
from functools import partial
import gradio as gr
import torch
from torchaudio.functional import resample

import captioning.utils.train_util as train_util


def load_model(cfg,
               ckpt_path,
               device):
    model = train_util.init_model_from_config(cfg["model"])
    ckpt = torch.load(ckpt_path, "cpu")
    train_util.load_pretrained_model(model, ckpt)
    model.eval()
    model = model.to(device)
    tokenizer = train_util.init_obj_from_dict(cfg["data"]["train"][
        "collate_fn"]["tokenizer"])
    if not tokenizer.loaded:
        tokenizer.load_state_dict(ckpt["tokenizer"])
    model.set_index(tokenizer.bos, tokenizer.eos, tokenizer.pad)
    return model, tokenizer


def infer(audio, device, model, tokenizer):
    sr, wav = audio
    if wav.ndim > 1:
        wav = wav.mean(1)
    wav = torch.as_tensor(wav) / 32768.0
    wav = resample(wav, sr, 16000)
    wav_len = len(wav)
    wav = wav.float().unsqueeze(0).to(device)
    input_dict = {
        "mode": "inference",
        "wav": wav,
        "wav_len": [wav_len],
        "specaug": False,
        "sample_method": "beam",
        "beam_size": 3,
    }
    with torch.no_grad():
        output_dict = model(input_dict)
        seq = output_dict["seq"].cpu().numpy()
        cap = tokenizer.decode(seq)[0]
    return cap


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--share", action="store_true", default=False)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = Path(args.exp_dir)
    cfg = train_util.load_config(exp_dir / "config.yaml")
    model, tokenizer = load_model(cfg, exp_dir / "swa.pth", device)

    interface = gr.Interface(fn=partial(infer, device=device, model=model, tokenizer=tokenizer),
                             inputs="audio",
                             outputs="text")
    interface.launch(share=args.share)

