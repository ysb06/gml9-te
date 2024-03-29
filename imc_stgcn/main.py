from dataclasses import dataclass
from typing import List
import random

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import tqdm
import os

import dgl
from imc_stgcn.load_data import data_transform
from imc_stgcn.model import STGCN_WAVE
from imc_stgcn.sensors2graph import get_adjacency_matrix
from imc_stgcn.utils import evaluate_metric, evaluate_model


def seed(seed_value, deterministic: bool = True):
    random.seed = seed_value
    np.random.seed = seed_value
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seed(42)


@dataclass
class Args:
    channels: List[int]
    lr: float = 0.001
    disablecuda = True
    batch_size: int = 32
    epochs: int = 32
    num_layers: int = 6
    window: int = 144
    # sensorsfilepath: str =  "./resources/IMCRTS_Dataset/sensor_graph/graph_sensor_ids.txt"
    # disfilepath: str =      "./resources/IMCRTS_Dataset/sensor_graph/distance_imc.csv"
    # tsfilepath: str =       "./resources/IMCRTS_Dataset/imcrts_df.pickle"
    sensorsfilepath: str = "./resources/METRLA/metr_ids.txt"
    disfilepath: str = "./resources/METRLA/distances_la_2012.csv"
    tsfilepath: str = "./resources/METRLA/metr-la.h5"
    savemodelpath: str = "stgcnwavemodel.pt"
    pred_len: int = 5
    control_str: str = "TNTSTNTST"


args = Args(channels=[1, 16, 32, 64, 32, 128])

device = (
    torch.device("cuda")
    if torch.cuda.is_available() and not args.disablecuda
    else torch.device("cpu")
)
# device = torch.device("mps") # For the mac

with open(args.sensorsfilepath) as f:
    sensor_ids = f.read().strip().split(",")

distance_df = pd.read_csv(args.disfilepath, dtype={"from": "str", "to": "str"})

adj_mx = get_adjacency_matrix(distance_df, sensor_ids)
sp_mx = sp.coo_matrix(adj_mx)
G = dgl.from_scipy(sp_mx)


print("Reading Pickle...")
df_format = args.tsfilepath.split(".")[-1]
if df_format == "pickle":
    df = pd.read_pickle(args.tsfilepath)
elif df_format == "h5":
    df = pd.read_hdf(args.tsfilepath)
else:
    raise Exception("Unknown Format")
print("Reading Complete")
num_samples, num_nodes = df.shape

tsdata = df.to_numpy()


n_his = args.window

save_path = args.savemodelpath


n_pred = args.pred_len
n_route = num_nodes
blocks = args.channels
# blocks = [1, 16, 32, 64, 32, 128]
drop_prob = 0
num_layers = args.num_layers

batch_size = args.batch_size
epochs = args.epochs
lr = args.lr


W = adj_mx
len_val = round(num_samples * 0.1)
len_train = round(num_samples * 0.7)
train = df[:len_train]
val = df[len_train : len_train + len_val]
test = df[len_train + len_val :]

scaler = StandardScaler()
train = scaler.fit_transform(train)
val = scaler.transform(val)
test = scaler.transform(test)


x_train, y_train = data_transform(train, n_his, n_pred, device)
x_val, y_val = data_transform(val, n_his, n_pred, device)
x_test, y_test = data_transform(test, n_his, n_pred, device)

train_data = torch.utils.data.TensorDataset(x_train, y_train)
train_iter = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)
val_data = torch.utils.data.TensorDataset(x_val, y_val)
val_iter = torch.utils.data.DataLoader(val_data, batch_size)
test_data = torch.utils.data.TensorDataset(x_test, y_test)
test_iter = torch.utils.data.DataLoader(test_data, batch_size)


loss = nn.MSELoss()
G = G.to(device)
model = STGCN_WAVE(
    blocks, n_his, n_route, G, drop_prob, num_layers, device, args.control_str
).to(device)
optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

min_val_loss = np.inf
for epoch in tqdm.tqdm(range(1, epochs + 1)):
    l_sum, n = 0.0, 0
    model.train()
    for x, y in tqdm.tqdm(train_iter, total=len(train_iter)):
        y_pred = model(x).view(len(x), -1)
        l = loss(y_pred, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        l_sum += l.item() * y.shape[0]
        n += y.shape[0]
    scheduler.step()
    val_loss = evaluate_model(model, loss, val_iter)
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        torch.save(model.state_dict(), save_path)
    print(
        "epoch",
        epoch,
        ", train loss:",
        l_sum / n,
        ", validation loss:",
        val_loss,
    )


best_model = STGCN_WAVE(
    blocks, n_his, n_route, G, drop_prob, num_layers, device, args.control_str
).to(device)
best_model.load_state_dict(torch.load(save_path))


l = evaluate_model(best_model, loss, test_iter)
MAE, MAPE, RMSE = evaluate_metric(best_model, test_iter, scaler)
print("test loss:", l, "\nMAE:", MAE, ", MAPE:", MAPE, ", RMSE:", RMSE)
