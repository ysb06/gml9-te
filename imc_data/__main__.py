import argparse
import importlib

parser = argparse.ArgumentParser()
parser.add_argument("--module", type=str, default="train")
args = parser.parse_args()

module = importlib.import_module(f'imc_data.{args.module}')
module.main()

print("Module Terminated")
