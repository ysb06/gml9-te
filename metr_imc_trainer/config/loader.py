from box import Box


def load_config(path: str) -> Box:
    return Box.from_yaml(filename=path)
