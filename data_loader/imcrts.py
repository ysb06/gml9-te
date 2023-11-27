import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import geopandas as gpd
import requests
import numpy as np

import data_loader

SERVICE_URL = "http://apis.data.go.kr/6280000/ICRoadVolStat/NodeLink_Trfc_DD"
PRIVATE_DECODED_KEY = "5HFe89gOZkcIZ/ZogD9zz18ZKcqBnu9nTIvf83zgORCxMx+SYz5RRGguMTi+zwrjolzlLWS/z363/7pyEVzUgw=="
MAX_ROW_COUNT = 5000

DATA_ROOT_PATH = os.path.join(data_loader.RESOURCE_PATH, "IMCRTS")

NODELINK_ROOT_PATH = os.path.join(data_loader.RESOURCE_PATH, "[2023-11-13]NODELINKDATA")
IMCRTS_PICKLE_DATA_PATH = os.path.join(DATA_ROOT_PATH, "imcrts_data.pickle")
IMCRTS_EXCEL_DATA_PATH = os.path.join(DATA_ROOT_PATH, "imcrts_data.xlsx")


if not os.path.exists(DATA_ROOT_PATH):
    os.makedirs(DATA_ROOT_PATH)

logger = logging.getLogger(__name__)


class IMCRTSCollector:
    def __init__(
        self, start_date: str = "20230101", end_date: str = "20231125"
    ) -> None:
        logger.info("Collecting...")
        self.params = {
            "serviceKey": PRIVATE_DECODED_KEY,
            "pageNo": 1,
            "numOfRows": MAX_ROW_COUNT,
            "YMD": "20240101",
        }
        self.start_date: datetime = datetime.strptime(start_date, "%Y%m%d")
        self.end_date: datetime = datetime.strptime(end_date, "%Y%m%d")

    def collect(self) -> None:
        data_list = []
        current_date: datetime = self.start_date

        logger.info(f"Collecting IMCRTS Data from {self.start_date} to {self.end_date}")

        day_count = 0
        while current_date <= self.end_date:
            current_date_string = current_date.strftime("%Y%m%d")
            self.params["YMD"] = current_date_string

            if day_count % 20 >= -1:
                logger.info(f"Requesting data at {current_date_string}...")

            code, data = self.get_data(self.params)
            if code == 200 and data is not None:
                data_list.extend(data)
            else:
                logger.error(f"Error Code: {code}")
                logger.error(f"Failed to Getting Data at [{current_date_string}]")
                break

            current_date += timedelta(days=1)
            time.sleep(0.1)

        df = pd.DataFrame(data_list)
        logger.info(f"Total Row Count: {len(df)}")
        logger.info("Creating Pickle...")
        df.to_pickle(IMCRTS_PICKLE_DATA_PATH)
        logger.info(f"{IMCRTS_PICKLE_DATA_PATH} is created")
        logger.info("Creating Excel...")
        df.to_excel(IMCRTS_EXCEL_DATA_PATH)
        logger.info(f"{IMCRTS_EXCEL_DATA_PATH} is created")

    def get_data(
        self, params: Dict[str, Any]
    ) -> Tuple[int, Optional[List[Dict[str, Any]]]]:
        """Request Data from Data Server
        SERVICE_URL로부터 GET 데이터 요청을 수행한다.

        Args:
            params (Dict[str, Any]): Parameters for Request

        Returns:
            Tuple[int, Optional[List[Dict[str, Any]]]]: Result of Data Request
        """
        res = requests.get(url=SERVICE_URL, params=params)
        data: Optional[List[Dict[str, Any]]] = None
        if res.status_code == 200:
            raw = res.json()
            if len(raw["response"]["body"]["items"]) > 0:
                data = raw["response"]["body"]["items"]

                if len(data) > MAX_ROW_COUNT:
                    message = f"Length of Data at {params['YMD']} is {data['response']['body']['items']} but sliced to {MAX_ROW_COUNT}"
                    logger.warning(message)
            else:
                logger.warning(f"No data at {params['YMD']}")
        else:
            print(res.text)

        return (res.status_code, data)


class IMCNodeLinkGenerator:
    def __init__(self) -> None:
        node_file_path = os.path.join(NODELINK_ROOT_PATH, "MOCT_NODE.shp")
        link_file_path = os.path.join(NODELINK_ROOT_PATH, "MOCT_LINK.shp")

        logger.info("Loading IMCRTS...")
        self.imcrts_data: pd.DataFrame = pd.read_pickle(IMCRTS_PICKLE_DATA_PATH)
        logger.info("Loading Node Data")
        self.node_data: gpd.GeoDataFrame = gpd.read_file(
            node_file_path, encoding="cp949"
        )
        logger.info("Loading Link Data")
        self.link_data: gpd.GeoDataFrame = gpd.read_file(
            link_file_path, encoding="cp949"
        )

    def generate(self):
        # 링크 데이터 생성
        logger.info("Collecting Incheon Link IDs...")
        imcrts_link_set = self.imcrts_data["linkID"].unique()

        logger.info("Collecting Incheon Links...")
        imc_link_data: gpd.GeoDataFrame = self.link_data[
            self.link_data["LINK_ID"].isin(imcrts_link_set)
        ]
        imc_link_data.to_crs(epsg=4326)  # 위도, 경도 방식으로 위치 변환. 정확도 낮아짐.

        link_output_pickle_path = os.path.join(DATA_ROOT_PATH, "imcrts_link.pickle")
        link_output_excel_path = os.path.join(DATA_ROOT_PATH, "imcrts_link.xlsx")
        link_output_shape_path = os.path.join(DATA_ROOT_PATH, "imcrts_link.shp")
        logger.info("Saving Link Data to Pickle...")
        imc_link_data.to_pickle(link_output_pickle_path)
        logger.info("Saving Link Data to Excel...")
        imc_link_data.to_excel(link_output_excel_path)
        logger.info("Saving Link Data to Shape...")
        imc_link_data.to_file(link_output_shape_path, encoding="utf-8")
        logger.info("Saving Link Data Complete")
        print(imc_link_data)

        # 노드 데이터 생성
        logger.info("Collecting Incheon Node IDs...")
        imcrts_node_set = np.unique(imc_link_data[["F_NODE", "T_NODE"]].values.ravel())

        logger.info("Collecting Incheon Nodes...")
        imc_node_data: gpd.GeoDataFrame = self.node_data[
            self.node_data["NODE_ID"].isin(imcrts_node_set)
        ]
        imc_node_data.to_crs(epsg=4326)  # 위도, 경도 방식으로 위치 변환. 정확도 낮아짐.

        node_output_pickle_path = os.path.join(DATA_ROOT_PATH, "imcrts_node.pickle")
        node_output_excel_path = os.path.join(DATA_ROOT_PATH, "imcrts_node.xlsx")
        node_output_shape_path = os.path.join(DATA_ROOT_PATH, "imcrts_node.shp")
        logger.info("Saving Node Data to Pickle...")
        imc_node_data.to_pickle(node_output_pickle_path)
        logger.info("Saving Node Data to Excel...")
        imc_node_data.to_excel(node_output_excel_path)
        logger.info("Saving Node Data to Shape...")
        imc_node_data.to_file(node_output_shape_path, encoding="utf-8")
        logger.info("Saving Node Data Complete")
        print(imc_node_data)
