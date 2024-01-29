from torch.utils.data import Dataset, Subset
import pandas as pd
import os


class MetrImcDataset(Dataset):
    def __init__(
        self,
        data_folder_path: str = "",
        # mode: str = "test",
        # window_size: int = 12,
        # train_ratio: float = 0.8,
        # val_ratio: float = 0.1,
        # batch_size: int = 128,
        # nb_worker: int = 0,
        # r: float = 1.0,
    ):
        self.name = "METR_IMC"
        self.traffic_volume_data = pd.read_csv(
            os.path.join(data_folder_path, "metr-imc.h5")
        )
        self.adjacency_data = pd.read_csv(
            os.path.join(data_folder_path, "sensor_graph/distances_imc_2023.csv")
        )

        # self.traffic_volume_data.val
        # self.timestamps = data.index
        # self.timestamps.freq = self.timestamps.inferred_freq
        # self.time_index = self.time_to_idx(self.timestamps, freq="5min")
        # self.data = data.values
        # self.data_mean, self.data_std = self.data.mean(), self.data.std()
        # self.data_min, self.data_max = self.data.min(), self.data.max()
