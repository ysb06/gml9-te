import argparse
import os

from . import Trainer
from .config.loader import load_config

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="./metr_imc_trainer/config.yaml")
args = parser.parse_args()

trainer = Trainer(*load_config(args.config))
trainer.train()
