import logging

import wandb
from box import Box

logging.basicConfig(
    format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO,
)


class WandbManager:
    def __init__(self, project_title: str, config: Box) -> None:
        # wandb.init(project=project_title, config=config.to_dict())
        pass

    def dispose(self) -> None:
        wandb.finish()
