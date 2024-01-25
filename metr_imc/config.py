import yaml
from box import Box


def load_config(path: str) -> Box:
    config = {}
    with open(path, "r") as fr:
        config = yaml.load(fr, Loader=yaml.FullLoader)

    return Box(config)
