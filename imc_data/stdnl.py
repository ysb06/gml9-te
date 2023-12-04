import logging
import os

import pandas as pd
import numpy as np
import geopandas as gpd

import imc_data

IMCRTS_DATA_PATH = os.path.join(imc_data.RESOURCE_PATH, "IMCRTS/imcrts.pickle")
NODELINK_ROOT_PATH = os.path.join(imc_data.RESOURCE_PATH, "[2023-11-13]NODELINKDATA")
OUTPUT_ROOT_PATH = os.path.join(imc_data.RESOURCE_PATH, "IMCRTS")

if not os.path.exists(OUTPUT_ROOT_PATH):
    os.makedirs(OUTPUT_ROOT_PATH)
logger = logging.getLogger(__name__)