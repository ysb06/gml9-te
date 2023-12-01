from typing import List
import dgl
from dataclasses import dataclass
import torch


@dataclass
class Args:
    lr: float = 0.001
    disablecuda = True
    batch_size: int = 50
    epochs: int = 50
    num_layers: int = 9
    window: int = 144
    sensorsfilepath: str = "./data/sensor_graph/graph_sensor_ids.txt"
    disfilepath: str = "./data/sensor_graph/distances_la_2012.csv"
    tsfilepath: str = "./data/metr-la.h5"
    savemodelpath: str = "stgcnwavemodel.pt"
    pred_len: int = 5
    control_str: str = "TNTSTNTST"
    channels: List[int] = [1, 16, 32, 64, 32, 128]


args = Args()

device = (
    torch.device("cuda")
    if torch.cuda.is_available() and not args.disablecuda
    else torch.device("cpu")
)
