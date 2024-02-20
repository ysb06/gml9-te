from typing import List, Tuple

import dgl
import lightning as L
import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from gm_lightning.models.torch.stgcn_wave import STGCN_WAVE


class STGCNWaveModel(L.LightningModule):
    def __init__(
        self,
        num_sensor: int,
        graph: dgl.DGLGraph,
        channels: List[int] = [1, 16, 32, 64, 32, 128],
        window: int = 144,
        dropout: float = 0,
        num_layers: int = 9,
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

        self.save_hyperparameters()

    def configure_optimizers(self) -> Optimizer:
        optimizer = torch.optim.RMSprop(self.model.parameters(), **self.optimizer_args)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **self.scheduler_args)

        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _):
        x, y = batch
        y_pred = self.forward(x).view(len(x), -1)
        loss: torch.Tensor = self.loss(y_pred, y)

        loss_sum = loss.item() * y.shape[0]
        num_sum = y.shape[0]

        self.log("training_step_loss", loss_sum / num_sum, on_step=True, on_epoch=False)
        self.log(
            "training_loss_sum",
            loss_sum,
            on_step=False,
            on_epoch=True,
            reduce_fx=torch.sum,
        )
        self.log(
            "training_num_sum",
            num_sum,
            on_step=False,
            on_epoch=True,
            reduce_fx=torch.sum,
        )

        return loss

    def on_train_epoch_end(self) -> None:
        outputs = self.trainer.callback_metrics
        self.log(
            "training_epoch_loss",
            outputs["training_loss_sum"] / outputs["training_num_sum"],
        )

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _):
        x, y = batch
        y_pred = self.forward(x).view(len(x), -1)
        loss: torch.Tensor = self.loss(y_pred, y)

        self.log(
            "validation_loss_sum",
            loss.item() * y.shape[0],
            on_step=False,
            on_epoch=True,
            reduce_fx=torch.sum,
        )
        self.log(
            "validation_num_sum",
            y.shape[0],
            on_step=False,
            on_epoch=True,
            reduce_fx=torch.sum,
        )

    def on_validation_epoch_end(self) -> None:
        outputs = self.trainer.callback_metrics
        self.log(
            "validation_loss",
            outputs["validation_loss_sum"] / outputs["validation_num_sum"],
        )
