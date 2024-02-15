import torch
import logging

logger = logging.getLogger(__name__)

torch_supported_device = ["cpu", "cuda", "mps"]


def get_auto_device(value: str):
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        if value not in torch_supported_device:
            logger.warning(
                f"{value} looks not valid PyTorch supported device. It may cause an exception."
            )
        return torch.device(value)
