# 표준노드링크가 데이터 크기가 너무 커서 로드 시간이 오래 걸려서 분석을 위한 샘플만 추출하는 코드
# imc_nodelink로 샘플을 대체 후 폐기

import logging
import geopandas as gpd
import os

import data_loader

DATA_ROOT_PATH = os.path.join(data_loader.RESOURCE_PATH, "[2023-11-13]NODELINKDATA")
NODE_DATA_PATH = os.path.join(DATA_ROOT_PATH, "MOCT_NODE.shp")
LINK_DATA_PATH = os.path.join(DATA_ROOT_PATH, "MOCT_LINK.shp")

logger = logging.getLogger(__name__)


class SampleGenerator:
    def __init__(self) -> None:
        self.root_path = os.path.join(data_loader.RESOURCE_PATH, "nodelink_sample")
        self.node_data_path = os.path.join(self.root_path, "moct_node.xlsx")
        self.link_data_path = os.path.join(self.root_path, "moct_link.xlsx")
        if not os.path.exists(self.root_path):
            os.mkdir(self.root_path)
        logger.info("Loading node data...")
        self.node_data: gpd.GeoDataFrame = gpd.read_file(
            NODE_DATA_PATH, encoding="cp949"
        )
        logger.info("Loading link data...")
        self.link_data: gpd.GeoDataFrame = gpd.read_file(
            LINK_DATA_PATH, encoding="cp949"
        )

    def generate(self) -> None:
        logger.info("Generating...")
        self.node_data.head(50).to_excel(self.node_data_path)
        self.link_data.head(50).to_excel(self.link_data_path)
        logger.info("Complete!")
