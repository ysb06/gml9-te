import argparse
import os

from .config.loader import load_config

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config.yaml")
args = parser.parse_args()

print(load_config(os.path.join("./metr_imc_trainer", args.config)))