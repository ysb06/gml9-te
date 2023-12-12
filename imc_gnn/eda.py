import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

DATASET_ROOT = "./resources/metr_ic_sample/Dataset"
DATASET_GRAPH_ROOT = os.path.join(DATASET_ROOT, "sensor_graph")


def execute():
    EDAnalyzer(pd.read_hdf(os.path.join(DATASET_ROOT, "metr-imc.h5")))


class EDAnalyzer:
    def __init__(self, raw: pd.DataFrame) -> None:
        self.raw = raw
        self.raw.index = pd.to_datetime(self.raw.index)

    def run(self):
        mean_data = self.raw.mean(axis=1)
        sample_data = self.raw[
            np.random.choice(self.raw.columns, size=9, replace=False)
        ]

        plt.plot(mean_data)
        plt.title("Average per Row")
        plt.xlabel("Row Index")
        plt.ylabel("Average Value")
        plt.show()

        fig, axes = plt.subplots(3, 3, figsize=(10, 8))  # 크기 조절 가능

        # 각 그리드에 선 그래프 그리기
        for i, ax in enumerate(axes.flatten()):
            sns.lineplot(
                data=sample_data, x=sample_data.index, y=sample_data.iloc[:, i], ax=ax
            )
            ax.set_title(sample_data.columns[i])

        plt.tight_layout()
        plt.show()
