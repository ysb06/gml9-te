from typing import Dict
from torch.utils.data import Dataset, Subset
import pandas as pd
import os
import numpy as np
import networkx as nx
import torch
from functools import partial

from metr_imc_trainer.utils import parallel_dtw


class MetrImcDataset(Dataset):
    def __init__(
        self,
        data_folder_path: str = "",
        window_size: int = 12,
        worker_count: int = 1,
        r: float = 1.0,
        training_loader: Dict[str, int] = {"ratio": 7, "batch_size": 32},
        validation_loader: Dict[str, int] = {"ratio": 1, "batch_size": 128},
        test_loader: Dict[str, int] = {"ratio": 2, "batch_size": 128},
    ):
        self.name = "METR_IMC"
        self.r = r
        self.window_size = window_size
        self.time_index = self.time_to_idx(self.traffic_volume_data.index, freq="5min")

        self.traffic_volume_data = pd.read_hdf(
            os.path.join(data_folder_path, "metr-imc.h5")
        )

        self.adjacency_data = pd.read_csv(
            os.path.join(data_folder_path, "sensor_graph/distances_imc_2023.csv")
        )
        self.adjacency_data = self.normalize_adj(self.adjacency_data, self.r)
        self.graph = nx.from_numpy_array(self.adjacency_data)

        self.data_mean, self.data_std = (
            self.traffic_volume_data.mean(),
            self.traffic_volume_data.std(),
        )
        self.data_min, self.data_max = (
            self.traffic_volume_data.min(),
            self.traffic_volume_data.max(),
        )

    def __len__(self):
        return self.traffic_volume_data.shape[0] - 2 * self.window_size

    def __getitem__(self, idx: int):
        sim = self.normalize_sim(
            parallel_dtw(
                self.traffic_volume_data[idx : idx + self.window_size].T,
                self.traffic_volume_data[
                    idx + self.window_size : idx + 2 * self.window_size
                ].T,
            ),
            self.r,
        )
        
        return (
            torch.tensor(
                self.time_index[idx : idx + self.window_size],
                dtype=torch.int,
            ).T,
            torch.tensor(self.adjacency_data, dtype=torch.float),
            torch.tensor(sim, dtype=torch.float),
            torch.tensor(
                self.traffic_volume_data[
                    idx : idx + self.window_size,
                    :,
                ],
                dtype=torch.float,
            ).unsqueeze(-1),
            torch.tensor(
                self.traffic_volume_data[
                    idx + self.window_size : idx + 2 * self.window_size,
                    :,
                ],
                dtype=torch.float,
            ).unsqueeze(-1),
        )

    def dtw(self, idx: int):
        if self.sim is not None:
            if len(self.sim.shape) > 2:
                return self.sim[idx]
            return self.sim
        return self.normalize_sim(
            parallel_dtw(
                self.traffic_volume_data[idx : idx + self.window_size].T,
                self.traffic_volume_data[
                    idx + self.window_size : idx + 2 * self.window_size
                ].T,
            ),
            self.r,
        )

    def get_splited_loaders(self):
        return None, None, None

    @staticmethod
    def normalize_adj(adj: np.ndarray, r: float = 1.0):
        a = adj.copy()
        a[a != 0] = np.exp(-((a[a != 0] / (a.mean() * r)) ** 2))
        a[a <= 0.1] = 0
        return a

    @staticmethod
    def normalize_sim(sim: np.ndarray, r: float = 1.0):
        s = sim.copy()
        s[s != 0] = 1 - np.exp(-((s[s != 0] / (s.std() * r)) ** 2))
        return s

    @staticmethod
    def time_to_idx(time_indexes: pd.Index[pd.Timestamp], freq: str = "5min"):
        hashmap = {
            t: i for i, t in enumerate(pd.date_range("00:00", "23:55", freq=freq).time)
        }
        results = []
        for timestamp in time_indexes:
            results.append([hashmap[timestamp.time()], timestamp.weekday()])
        return np.array(results, dtype=np.int32)
