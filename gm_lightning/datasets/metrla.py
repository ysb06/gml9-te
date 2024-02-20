from typing import List, Tuple
import pandas as pd
import torch
import numpy as np

from torch.utils.data import Dataset, Subset
import scipy.sparse as sp
import dgl
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from torch import Tensor


class MetrGraphDataset(Dataset):
    def __init__(
        self,
        folder_path: str,
        traffic_data_path: str,
        distances_data_path: str,
        sensor_ids_path: str,
        k: float = 0.1,
        prediction_length: int = 5,
        window_size: int = 144,
    ) -> None:
        self.window_size = window_size
        self.prediction_length = prediction_length

        self.root_path = Path(folder_path)
        self.raw_data = pd.read_hdf(self.root_path / traffic_data_path)
        distances_data = pd.read_csv(
            self.root_path / distances_data_path, dtype={"from": "str", "to": "str"}
        )

        sensor_ids = []
        with open(self.root_path / sensor_ids_path) as f:
            sensor_ids = f.read().strip().split(",")
        adj_mx = self.get_adjacency_matrix(distances_data, sensor_ids, k)
        sp_mx = sp.coo_matrix(adj_mx)
        self.G = dgl.from_scipy(sp_mx)

        scaler = StandardScaler()
        self.traffic_data = scaler.fit_transform(self.raw_data)

        self.train_size = len(self)
        self.val_size = 0

    def split_data(
        self, training_ratio: float, validation_ratio: float
    ) -> Tuple[Subset, Subset, Subset]:
        self.train_size = int(training_ratio * len(self))
        self.val_size = int(validation_ratio * len(self))

        scaler = StandardScaler()
        scaler.fit(self.raw_data[: self.train_size])
        self.traffic_data = scaler.transform(self.raw_data)

        train_end = self.train_size
        val_end = self.train_size + self.val_size
        test_end = len(self.traffic_data)
        return (
            Subset(self, range(0, train_end)),
            Subset(self, range(train_end, val_end)),
            Subset(self, range(val_end, test_end)),
        )

    @property
    def test_size(self):
        return len(self.traffic_data) - self.train_size - self.val_size

    def __len__(self) -> int:
        return len(self.traffic_data) - self.window_size - self.prediction_length

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        num_route = self.traffic_data.shape[1]

        head = idx
        tail = idx + self.window_size

        x = self.traffic_data[head:tail].reshape(1, self.window_size, num_route)
        y = self.traffic_data[tail + self.prediction_length - 1]

        return torch.Tensor(x), torch.Tensor(y)

    @property
    def graph(self):
        return self.G

    @property
    def sensor_count(self):
        return self.G.num_nodes()

    def get_adjacency_matrix(
        self, distance_df: pd.DataFrame, sensor_ids: List[str], normalized_k=0.1
    ):
        """
        :param distance_df: data frame with three columns: [from, to, distance].
        :param sensor_ids: list of sensor ids.
        :param normalized_k: entries that become lower than normalized_k after normalization are set to zero for sparsity.
        :return: adjacency matrix
        """
        num_sensors = len(sensor_ids)
        dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
        dist_mx[:] = np.inf
        # Builds sensor id to index map.
        sensor_id_to_ind = {}
        for i, sensor_id in enumerate(sensor_ids):
            sensor_id_to_ind[sensor_id] = i
        # Fills cells in the matrix with distances.
        for row in distance_df.values:
            if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
                continue
            dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]

        # Calculates the standard deviation as theta.
        distances = dist_mx[~np.isinf(dist_mx)].flatten()
        std = distances.std()
        adj_mx = np.exp(-np.square(dist_mx / std))
        # Make the adjacent matrix symmetric by taking the max.
        # adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])

        # Sets entries that lower than a threshold, i.e., k, to zero for sparsity.
        adj_mx[adj_mx < normalized_k] = 0
        return adj_mx
