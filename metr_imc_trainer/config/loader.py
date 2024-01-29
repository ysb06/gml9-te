from typing import Optional, Tuple
from box import Box
import torch

from ..utils import get_auto_device


class Config:
    def __init__(self, config: Box) -> None:
        self.config = Box(config, box_recast={"device": get_auto_device})

    def split_to_subs(self) -> Tuple[Box, Optional[Box]]:
        if "__subs__" in self.config:
            return (
                self.config - Box(__subs__=self.config.__subs__),
                self.config.__subs__,
            )
        else:
            return self.config, None

    def extract_config(self):
        conf, subs = self.split_to_subs()

        if subs is None:
            return conf, None
        else:
            return conf, Config(subs)


def load_config(path: str) -> Box:
    config_box = Box.from_yaml(filename=path)
    config = Config(config_box)

    return config.extract_config()


def extract_config(config: Config) -> Tuple[Box, Optional[Config]]:
    conf, subs = config.split_to_subs()

    if subs is None:
        return conf, None
    else:
        return conf, Config(subs)
