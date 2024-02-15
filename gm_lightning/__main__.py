import random
from pathlib import Path
from typing import Any, Dict, List

import hydra
import lightning as L
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.optim import Optimizer
import dgl

from gm_lightning.datasets.metrla import MeterGraphDataset
from gm_lightning.models.stgcn_wave import STGCN_WAVE


def seed(seed_value, deterministic: bool = True):
    random.seed = seed_value
    np.random.seed = seed_value
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class STGCNWaveModel(L.LightningModule):
    def __init__(
        self,
        num_sensor: int,
        graph: dgl.DGLGraph,
        channels: List[int] = [1, 16, 32, 64, 32, 128],
        window: int = 12,
        dropout: float = 0,
        num_layers: int = 6,
        control_str: str = "TNTSTNTST",
        lr: float = 0.01,
        step_size: int = 5,
        gamma: float = 0.7,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.model = STGCN_WAVE(
            channels,
            window,
            num_sensor,
            graph,
            dropout,
            num_layers,
            device,
            control_str,
        )
        self.loss = nn.MSELoss()
        self.optimizer_args = {"lr": lr}
        self.scheduler_args = {"step_size": step_size, "gamma": gamma}

    def configure_optimizers(self) -> Optimizer:
        optimizer = torch.optim.RMSprop(self.model.parameters(), **self.optimizer_args)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **self.scheduler_args)

        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def training_step(self, batch, batch_idx) -> Tensor:
        x, y = batch
        y_pred = self.model(x).view(len(x), -1)
        loss = self.loss(y_pred, y)
        return loss


@hydra.main(
    version_base=None,
    config_path=(Path(__file__).parent / "config").as_posix(),
    config_name="config",
)
def run(config: DictConfig) -> None:
    dataset: MeterGraphDataset = instantiate(config.dataset)
    model: STGCNWaveModel = STGCNWaveModel(
        **config.model,
        num_sensor=dataset.sensor_count,
        graph=dataset.graph,
        device=torch.device(config.device)
    )


if __name__ == "__main__":
    run()
