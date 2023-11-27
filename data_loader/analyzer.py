import os
import data_loader
import geopandas as gpd
import logging

IMC_NODE_DATA_PATH = os.path.join(data_loader.RESOURCE_PATH, "imc_nodelink/imc_node.shp")

logger = logging.getLogger(__name__)

def analyze():
    get_node_region()

def get_node_region():
    data: gpd.GeoDataFrame = gpd.read_file(IMC_NODE_DATA_PATH)

    logger.info(f"Data Length: {len(data)}")
    print(data)
    print(data['NODE_ID'])
    print(data['NODE_ID'].str[:3].unique())