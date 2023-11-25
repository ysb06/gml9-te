import logging
import geopandas as gpd
import os

logging.basicConfig(
    format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logger.info("Running Program...")


DATA_PATH = "./resources/nodelink_data"

logger.info("Loading Nodes...")
node_shape: gpd.GeoDataFrame = gpd.read_file(
    os.path.join(DATA_PATH, "MOCT_NODE.shp"), encoding="cp949"
)
print(node_shape)
node_shape.head(10).to_excel(os.path.join(DATA_PATH, "sample.xlsx"))
# node_shapex = gpd.read_file(os.path.join(DATA_PATH, "MOCT_NODE.shx"), encoding="cp949")
# print(node_shapex)
