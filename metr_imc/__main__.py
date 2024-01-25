import argparse
from os.path import join

from .config import load_config
from .imcrts import generate_imcrts_data
from .metr_converter import execute_metr_converter

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config.yaml")
args = parser.parse_args()

config = load_config(join("./metr_imc", args.config))
generate_imcrts_data(config.imcrts)
execute_metr_converter(config.metr_imc)
