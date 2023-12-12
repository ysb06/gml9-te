import logging
import os

RESOURCE_PATH = "./resources/"

logging.basicConfig(
    format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO,
)
logging.getLogger(__name__).info(f"Running Module: {__name__}...")
if not os.path.exists(RESOURCE_PATH):
    os.mkdir(RESOURCE_PATH)


