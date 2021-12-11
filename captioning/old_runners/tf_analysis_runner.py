import os
import sys

import fire
import pandas as pd
import torch

from ignite.engine.engine import Engine, Events
from ignite.metrics import Accuracy, Loss, RunningAverage, Average
from ignite.contrib.handlers import ProgressBar
from ignite.utils import convert_tensor

sys.path.append(os.getcwd())
import models
import utils.train_util as train_util
from utils.build_vocab import Vocabulary
from runners.base_runner import BaseRunner
from datasets.SJTUDataSet import SJTUDataset, SJTUDatasetEval, collate_fn


class TestRunner(BaseRunner):

    def _forward(self, model, batch, mode="train", **kwargs):
        assert mode in ("train", "sample")

        if mode == "sample":
            feats = batch[1]
            feat_lens = batch[-1]

            feats = convert_tensor(feats.float(),
                                   device=self.device,
                                   non_blocking=True)
            sampled = model(feats, feat_lens, mode="sample", **kwargs)
            return sampled

        # mode is "train"
        assert "tf" in kwargs, "need to know whether to use teacher forcing"

        feats = batch[0]
        caps = batch[1]
        feat_lens = batch[-2]
        cap_lens = batch[-1]
        feats = convert_tensor(feats.float(),
                               device=self.device,
                               non_blocking=True)
        caps = convert_tensor(caps.long(),
                              device=self.device,
                              non_blocking=True)
        # pack labels to remove padding from caption labels
        targets = torch.nn.utils.rnn.pack_padded_sequence(
            caps, cap_lens, batch_first=True).data


        if kwargs["tf"]:
            output = model(feats, feat_lens, caps, cap_lens, mode="forward", no_pack=True)
        else:
            output = model(feats, feat_lens, mode="sample", max_length=max(cap_lens))

        packed_probs = torch.nn.utils.rnn.pack_padded_sequence(
            output["probs"], cap_lens, batch_first=True).data
        packed_probs = convert_tensor(packed_probs, device=self.device, non_blocking=True)
        output["packed_probs"] = packed_probs

        output["targets"] = targets

        return output

    def test(self, config, **kwargs):
        config_parameters = train_util.parse_config_or_kwargs(config, **kwargs)
        caption_df = pd.read_json(config_parameters["caption_file"], dtype={"key": str})
        vocabulary = torch.load(config_parameters["vocab_file"])

        dump = torch.load(os.path.join(config_parameters["experiment_path"], "saved.pth"),
                          map_location="cpu")
        model = dump["model"]
        model = model.to(self.device)
        # Some scaler (sklearn standardscaler)
        scaler = dump["scaler"]
        dataset = SJTUDataset(
            kaldi_stream=config_parameters["feature_stream"],
            caption_df=caption_df,
            vocabulary=vocabulary,
            transform=scaler.transform)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=True,
            collate_fn=collate_fn([0, 1]),
            **config_parameters["dataloader_args"])

        def _inference(engine, batch):
            model.eval()
            caps = batch[1]
            caps = convert_tensor(caps.long(),
                                  device=self.device,
                                  non_blocking=True)
            with torch.no_grad():
                output_tf = self._forward(model, batch, tf=True)
                output_sample = self._forward(model, batch, tf=False)

                # print((output_tf["probs"][:, 0, :].argmax(dim=1) == caps[:, 0]).sum())
                # print((output_tf["probs"][:, 1, :].argmax(dim=1) == caps[:, 1]).sum())
                # print((output_tf["probs"][:, 2, :].argmax(dim=1) == caps[:, 2]).sum())
                return {"output_tf": output_tf, 
                        "output_sample": output_sample, 
                        "caps": caps}
            

        metrics = {
            "tf_acc": Accuracy(output_transform=lambda x: (x["output_tf"]["packed_probs"], x["output_tf"]["targets"])),
            "sample_acc": Accuracy(output_transform=lambda x: (x["output_sample"]["packed_probs"], x["output_sample"]["targets"])),
            "tf_acc_1": Accuracy(output_transform=lambda x: (x["output_tf"]["probs"][:, 0, :], x["caps"][:, 0])),
            "sample_acc_1": Accuracy(output_transform=lambda x: (x["output_sample"]["probs"][:, 0, :], x["caps"][:, 0])),
            "tf_acc_2": Accuracy(output_transform=lambda x: (x["output_tf"]["probs"][:, 1, :], x["caps"][:, 1])),
            "sample_acc_2": Accuracy(output_transform=lambda x: (x["output_sample"]["probs"][:, 1, :], x["caps"][:, 1])),
            "tf_acc_3": Accuracy(output_transform=lambda x: (x["output_tf"]["probs"][:, 2, :], x["caps"][:, 2])),
            "sample_acc_3": Accuracy(output_transform=lambda x: (x["output_sample"]["probs"][:, 2, :], x["caps"][:, 2])),
            "tf_acc_4": Accuracy(output_transform=lambda x: (x["output_tf"]["probs"][:, 3, :], x["caps"][:, 3])),
            "sample_acc_4": Accuracy(output_transform=lambda x: (x["output_sample"]["probs"][:, 3, :], x["caps"][:, 3])),
            "tf_acc_5": Accuracy(output_transform=lambda x: (x["output_tf"]["probs"][:, 4, :], x["caps"][:, 4])),
            "sample_acc_5": Accuracy(output_transform=lambda x: (x["output_sample"]["probs"][:, 4, :], x["caps"][:, 4])),
            "tf_acc_6": Accuracy(output_transform=lambda x: (x["output_tf"]["probs"][:, 5, :], x["caps"][:, 5])),
            "sample_acc_6": Accuracy(output_transform=lambda x: (x["output_sample"]["probs"][:, 5, :], x["caps"][:, 5])),
            "tf_acc_7": Accuracy(output_transform=lambda x: (x["output_tf"]["probs"][:, 6, :], x["caps"][:, 6])),
            "sample_acc_7": Accuracy(output_transform=lambda x: (x["output_sample"]["probs"][:, 6, :], x["caps"][:, 6])),
            "tf_acc_8": Accuracy(output_transform=lambda x: (x["output_tf"]["probs"][:, 7, :], x["caps"][:, 7])),
            "sample_acc_8": Accuracy(output_transform=lambda x: (x["output_sample"]["probs"][:, 7, :], x["caps"][:, 7])),
            "tf_acc_9": Accuracy(output_transform=lambda x: (x["output_tf"]["probs"][:, 8, :], x["caps"][:, 8])),
            "sample_acc_9": Accuracy(output_transform=lambda x: (x["output_sample"]["probs"][:, 8, :], x["caps"][:, 8])),
            "tf_acc_10": Accuracy(output_transform=lambda x: (x["output_tf"]["probs"][:, 9, :], x["caps"][:, 9])),
            "sample_acc_10": Accuracy(output_transform=lambda x: (x["output_sample"]["probs"][:, 9, :], x["caps"][:, 9])),
        }

        min_cap_len = 10

        # for i in range(min_cap_len):
            # metrics["tf_acc_{}".format(i+1)] = Accuracy(output_transform=lambda x: (x["output_tf"]["probs"][:, i, :], x["caps"][:, i]))
            # metrics["sample_acc_{}".format(i+1)] = Accuracy(output_transform=lambda x: (x["output_tf"]["probs"][:, i, :], x["caps"][:, i]))

        pbar = ProgressBar(persist=False, ascii=True)
        evaluator = Engine(_inference)
        pbar.attach(evaluator)

        for name, metric in metrics.items():
            metric.attach(evaluator, name)

        @evaluator.on(Events.COMPLETED)
        def log_results(engine):
            with open(os.path.join(config_parameters["experiment_path"], "position_wise_acc.txt"), "w") as f:
                f.write("{}: {:.2%}\t {}: {:.2%}\n".format(
                    "tf_acc", 
                    engine.state.metrics["tf_acc"],
                    "sample_acc",
                    engine.state.metrics["sample_acc"])
                )
                for i in range(min_cap_len):
                    f.write("{}: {:.2%}\t {}: {:.2%}\n".format(
                        "tf_acc_{}".format(i+1), 
                        engine.state.metrics["tf_acc_{}".format(i+1)], 
                        "sample_acc_{}".format(i+1), 
                        engine.state.metrics["sample_acc_{}".format(i+1)])
                    )

        evaluator.run(dataloader, max_epochs=1)

if __name__ == "__main__":
    fire.Fire(TestRunner)
