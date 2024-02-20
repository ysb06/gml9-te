import random
from pathlib import Path

import hydra
import lightning as L
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import wandb

from gm_lightning.datasets.metrla import MetrGraphDataset
from gm_lightning.models.lightning.stgcn_wave import STGCNWaveModel


def get_torch_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        supported_devices = ["cuda", "mps", "cpu"]
        if device in supported_devices:
            return torch.device(device)
        else:
            raise ValueError(f"Unsupported device: {device}")


@hydra.main(
    version_base=None,
    config_path=(Path(__file__).parent / "config").as_posix(),
    config_name="config",
)
def run(config: DictConfig) -> None:
    device = get_torch_device(config.device)
    L.seed_everything(config.seed)

    dataset: MetrGraphDataset = instantiate(config.dataset)
    dataset.G.to(device)
    training_set, validation_set, test_set = dataset.split_data(**config.dataset_split)
    training_data_loader = DataLoader(training_set, **config.dataloader.training)
    validation_data_loader = DataLoader(validation_set, **config.dataloader.validation)

    model: STGCNWaveModel = STGCNWaveModel(
        **OmegaConf.to_container(config.model, resolve=True),
        num_sensor=dataset.sensor_count,
        graph=dataset.graph,
        device=device,
    )

    # wandb_logger = WandbLogger(name="METR-LA Lightning", project="STGCN_WAVE")
    # wandb_logger.experiment.config.update(config)
    # trainer = L.Trainer(logger=wandb_logger, max_epochs=config.epochs)
    trainer = L.Trainer(
        **config.trainer,
        accelerator=device.type,
    )
    trainer.fit(
        model,
        train_dataloaders=training_data_loader,
        val_dataloaders=validation_data_loader,
    )


if __name__ == "__main__":
    run()
