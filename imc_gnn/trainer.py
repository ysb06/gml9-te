import argparse
import logging
import random
from dataclasses import dataclass, asdict
from typing import List
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import imc_gnn.utils
import imc_gnn.load_data
import imc_gnn.sensors2graph
import torch.utils.data
import imc_gnn.model
import tqdm

import dgl
import wandb

logger = logging.getLogger(__name__)

DATASET_ROOT = "./resources/metr_ic_sample/Dataset"
DATASET_GRAPH_ROOT = os.path.join(DATASET_ROOT, "sensor_graph")


@dataclass
class HyperParams:
    channels: List[int]
    lr: float = 0.0025
    disablecuda = True
    batch_size: int = 50
    epochs: int = 50
    window: int = 72  # 일단 3일 기준
    # 특정 시점에서 언제까지의 시간을 모델에 입력할 것인가 (Default: 144 = 60 / 5 * 12, 5분 간격 측정)
    # 1시간 간격 측정일 경우...24시간 or 12시간?
    num_layers: int = 9
    sensorsfilepath: str = os.path.join(DATASET_GRAPH_ROOT, "graph_sensor_ids.txt")
    disfilepath: str = os.path.join(DATASET_GRAPH_ROOT, "distances_imc_2023.csv")
    tsfilepath: str = os.path.join(DATASET_ROOT, "metr-imc.h5")
    savemodelpath: str = "./models/231213_imc_stgcn_wave_epoch50_best.pt"
    pred_len: int = 5
    control_str: str = "TNTSTNTST"
    info: str = "Device: Cuda, MV: fill with 0, Loss: MSELoss, Model: STGCN_WAVE, Optim: RMSprop, Scheduler: StepLR, Train-Val-Test Ratio: 7:1:2, Scaler: StandardScaler"


class Trainer:
    def __init__(
        self, args: HyperParams = HyperParams(channels=[1, 16, 32, 64, 32, 128])
    ) -> None:
        wandb.init(
            project=f"STGCN_WAVE Training with {DATASET_ROOT.split('/')[-1]}",
            config=asdict(args)
        )
        self.args = args
        self.device = torch.device("cuda")
        with open(args.sensorsfilepath) as f:
            sensor_ids = f.read().strip().split(",")
        self.distance_df = pd.read_csv(args.disfilepath, dtype={"from": str, "to": str})
        adj_matrix = imc_gnn.sensors2graph.get_adjacency_matrix(
            self.distance_df, sensor_ids
        )
        sparse_matrix = sp.coo_matrix(adj_matrix)  # Scipy에서 지원하는 Sparse Matrix
        self.G = dgl.from_scipy(sparse_matrix)

        self.ts_df = pd.read_hdf(args.tsfilepath)
        self.ts_df = self.ts_df.fillna(0)
        print(self.ts_df)

        self.num_samples, self.num_nodes = self.ts_df.shape
        self.tsdata = self.ts_df.to_numpy()
        self.drop_prob = 0

        self.W = adj_matrix

        self.train_iter, self.val_iter, self.test_iter = self.get_data_loader()

        self.loss = nn.MSELoss()
        self.G = self.G.to(self.device)
        self.model: imc_gnn.model.STGCN_WAVE = imc_gnn.model.STGCN_WAVE(
            args.channels,
            args.window,
            self.num_nodes,
            self.G,
            self.drop_prob,
            args.num_layers,
            self.device,
            args.control_str,
        ).to(self.device)
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=5, gamma=0.7
        )

        

    def get_data_loader(self):
        n_his = self.args.window
        n_pred = self.args.pred_len
        batch_size = self.args.batch_size

        len_val = round(self.num_samples * 0.1)
        len_train = round(self.num_samples * 0.7)
        train = self.ts_df[:len_train]
        val = self.ts_df[len_train : len_train + len_val]
        test = self.ts_df[len_train + len_val :]
        self.scaler = StandardScaler()
        train = self.scaler.fit_transform(train)
        val = self.scaler.transform(val)
        test = self.scaler.transform(test)
        x_train, y_train = imc_gnn.load_data.data_transform(
            train, n_his, n_pred, self.device
        )
        x_val, y_val = imc_gnn.load_data.data_transform(val, n_his, n_pred, self.device)
        x_test, y_test = imc_gnn.load_data.data_transform(
            test, n_his, n_pred, self.device
        )

        train_data = torch.utils.data.TensorDataset(x_train, y_train)
        train_iter = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)
        val_data = torch.utils.data.TensorDataset(x_val, y_val)
        val_iter = torch.utils.data.DataLoader(val_data, batch_size)
        test_data = torch.utils.data.TensorDataset(x_test, y_test)
        test_iter = torch.utils.data.DataLoader(test_data, batch_size)

        return train_iter, val_iter, test_iter

    def train(self):
        logger.info("Training...")

        min_validation_loss = np.inf
        for epoch in range(1, self.args.epochs + 1):
            l_sum = 0.0
            n = 0
            self.model.train()
            for x, y in tqdm.tqdm(self.train_iter, total=len(self.train_iter)):
                result: torch.Tensor = self.model(x)
                y_pred = result.view(len(x), -1)
                l: torch.Tensor = self.loss(y_pred, y)
                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()
                l_sum += l.item() * y.shape[0]
                n += y.shape[0]

                wandb.log({"train_loss": l_sum / n})
            self.scheduler.step()
            val_loss: float = imc_gnn.utils.evaluate_model(
                self.model, self.loss, self.val_iter
            )
            if val_loss < min_validation_loss:
                min_validation_loss = val_loss
                torch.save(self.model.state_dict(), self.args.savemodelpath)
                logger.info(f"Best Model Saved at {epoch}")
                wandb.log({"Best Model Epoch": epoch})
            print(
                "epoch",
                epoch,
                ", train loss:",
                l_sum / n,
                ", validation loss:",
                val_loss,
            )
            wandb.log({"val_loss": val_loss})

        best_model = imc_gnn.model.STGCN_WAVE(
            self.args.channels,
            self.args.window,
            self.num_nodes,
            self.G,
            self.drop_prob,
            self.args.num_layers,
            self.device,
            self.args.control_str,
        ).to(self.device)
        best_model.load_state_dict(torch.load(self.args.savemodelpath))

        l = imc_gnn.utils.evaluate_model(best_model, self.loss, self.test_iter)
        MAE, MAPE, RMSE = imc_gnn.utils.evaluate_metric(
            best_model, self.test_iter, self.scaler
        )
        print("test loss:", l, "\nMAE:", MAE, ", MAPE:", MAPE, ", RMSE:", RMSE)
        wandb.log({"test loss:": l, "MAE:": MAE, "MAPE": MAPE, "RMSE": RMSE})
        wandb.finish()