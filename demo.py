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


def infer(file, device, model, tokenizer, target_sr):
    sr, wav = file
    wav = torch.as_tensor(wav)
    if wav.dtype == torch.short:
        wav = wav / 2 ** 15
    elif wav.dtype == torch.int:
        wav = wav / 2 ** 31
    if wav.ndim > 1:
        wav = wav.mean(1)
    wav = resample(wav, sr, target_sr)
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
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--share", action="store_true", default=False)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = Path(args.ckpt).parent
    cfg = train_util.load_config(exp_dir / "config.yaml")
    target_sr = cfg["data"]["train"]["dataset"]["args"]["target_sr"]
    model, tokenizer = load_model(cfg, args.ckpt, device)

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                file = gr.Audio(label="Input", visible=True)
                btn = gr.Button("Run")
            with gr.Column():
                output = gr.Textbox(label="Output")
            btn.click(
                fn=partial(infer,
                           device=device,
                           model=model,
                           tokenizer=tokenizer,
                           target_sr=target_sr),
                inputs=[file,],
                outputs=output
            )
        
        demo.launch(share=args.share)

