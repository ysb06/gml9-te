import logging
import random
from box import Box
from .config.loader import Config

import numpy as np
import torch

from .logger import WandbManager
from .datasets.metr import MetrImcDataset

logger = logging.getLogger(__name__)


def seed(seed_value, deterministic: bool = True):
    random.seed = seed_value
    np.random.seed = seed_value
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Trainer:
    def __init__(self, main_config: Box, module_config: Config) -> None:
        self.wandb_logger = WandbManager("STGM-METR_IMC", {})
        self.device = torch.device(main_config.device)
        self.dataset = MetrImcDataset(**module_config.datasets)
        (
            self.training_dataloader,
            self.valid_dataloader,
            self.test_dataloader,
        ) = self.dataset.get_splited_loaders(**module_config.datasets_split)

        print(main_config)
        print(module_config)

    def train(self) -> None:
        logger.info("Training...")

    def dispose(self) -> None:
        self.wandb_logger.dispose()
