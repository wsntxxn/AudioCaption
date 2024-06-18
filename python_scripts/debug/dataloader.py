import argparse
from tqdm import tqdm
import torch

import captioning.utils.train_util as train_util


parser = argparse.ArgumentParser()

parser.add_argument("--config", "-c", type=str, required=True)

config_path = parser.parse_args().config
config = train_util.load_config(config_path)

print(config["data"]["train"])

dataset = train_util.init_obj_from_dict(config["data"]["train"]["dataset"])
collate_fn = train_util.init_obj_from_dict(config["data"]["train"]["collate_fn"])

dataloader = torch.utils.data.DataLoader(
    dataset=dataset, 
    collate_fn=collate_fn,
    **config["data"]["train"]["dataloader_args"])


with tqdm(total=len(dataloader), ascii=True, ncols=100) as pbar:
    for batch in dataloader:
        # output_dict = {
        #     key: data.shape for key, data in batch.items() if \
        #         isinstance(data, torch.Tensor)
        # }
        # pbar.set_postfix(output_dict)
        pbar.update()

for key, val in batch.items():
    print(f"{key}: {val.shape}")
