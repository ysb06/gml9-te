from typing import List, Tuple
import pandas as pd
import torch
import numpy as np

from torch.utils.data import Dataset
import scipy.sparse as sp
import dgl
from pathlib import Path

from torch import Tensor


class MeterGraphDataset(Dataset):
    def __init__(
        self,
        folder_path: str,
        traffic_data_path: str,
        distances_data_path: str,
        sensor_ids_path: str,
        k: float = 0.1,
    ) -> None:
        self.root_path = Path(folder_path)
        self.traffic_data = pd.read_hdf(self.root_path / traffic_data_path)
        distances_data = pd.read_csv(
            self.root_path / distances_data_path, dtype={"from": "str", "to": "str"}
        )
        sensor_ids = []
        with open(self.root_path / sensor_ids_path) as f:
            sensor_ids = f.read().strip().split(",")
        adj_mx = self.get_adjacency_matrix(distances_data, sensor_ids, k)
        sp_mx = sp.coo_matrix(adj_mx)
        self.G = dgl.from_scipy(sp_mx)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.x[idx], self.y[idx]

    @property
    def graph(self):
        return self.G
    
    @property
    def sensor_count(self):
        return len(self.G.nodes)

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

class METRLADataset:
    def __init__(self, file_path):
        self.data = None
        try:
            self.data = pd.read_csv(file_path)
        except FileNotFoundError:
            print("File not found. Please provide a valid file path.")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data.iloc[idx]
    
    def data_transform(self, data, n_his, n_pred, device):
        # produce data slices for training and testing
        n_route = data.shape[1]
        l = len(data)
        num = l - n_his - n_pred
        x = np.zeros([num, 1, n_his, n_route])
        y = np.zeros([num, n_route])

        cnt = 0
        for i in range(l - n_his - n_pred):
            head = i
            tail = i + n_his
            x[cnt, :, :, :] = data[head:tail].reshape(1, n_his, n_route)
            y[cnt] = data[tail + n_pred - 1]
            cnt += 1
        return torch.Tensor(x).to(device), torch.Tensor(y).to(device)

# Example usage
# dataset = METRLADataset("path/to/dataset.csv")
# dataset.load_data()
# num_samples = dataset.get_num_samples()
# num_features = dataset.get_num_features()
# sample = dataset.get_sample(0)