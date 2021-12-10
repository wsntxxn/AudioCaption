#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import argparse
import logging


parser = argparse.ArgumentParser()
parser.add_argument("input_csvs", type=str, nargs="+")
parser.add_argument("output_csv", type=str)
parser.add_argument("--sep", type=str, default="\t")

args = parser.parse_args()

logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=logfmt)

df = pd.concat([pd.read_csv(fname, sep=args.sep) for fname in args.input_csvs], join="inner")
df.reset_index(inplace=True, drop=True)
df.to_csv(args.output_csv, sep=args.sep, index=False)
logging.info("new csv has " + str(df.shape[0]) + " items")
