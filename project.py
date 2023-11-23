import argparse
import requests

parser = argparse.ArgumentParser()
parser.add_argument("--node-link-ver", dest="node_link_ver", default="2023-11-13")
args = parser.parse_args()

nl_ver = args.node_link_ver

print(nl_ver)