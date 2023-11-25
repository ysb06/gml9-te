import argparse
import requests
import logging
import geopandas as gpd
import os

DATA_PATH = "./resources/nodelink_data"
SAMPLE_DATA_PATH = "./resources/nodelink_data_sample"

logging.basicConfig(
    format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Arg 로드
parser = argparse.ArgumentParser()
parser.add_argument("--node-link-ver", dest="node_link_ver", default="DF_191/0")
args = parser.parse_args()


def generate_directories():
    """필수 폴더들이 없을 경우 생성"""

    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    if not os.path.exists(SAMPLE_DATA_PATH):
        os.makedirs(SAMPLE_DATA_PATH)


def generate_sample_data():
    """표준노드링크 데이터를 읽고 샘플을 생성"""
    file_list = [
        os.path.join(DATA_PATH, "MOCT_NODE.shp"),
        os.path.join(DATA_PATH, "MOCT_NODE.shx"),
        os.path.join(DATA_PATH, "MOCT_NODE.dbf"),
    ]
    logger.info("Loading...")
    for filename in file_list:
        logger.info(f"Reading {filename}...")
        data: gpd.GeoDataFrame = gpd.read_file(filename, encoding="cp949")
        logger.info("Original CRS:", data.crs)
        data = data.to_crs(epsg=4326)
        logger.info("Transformed CRS:", data.crs)
        print(data)
        data.head(10).to_excel(
            os.path.join(SAMPLE_DATA_PATH, f"node_{filename.split('.')[-1]}.xlsx")
        )


generate_directories()
generate_sample_data()
