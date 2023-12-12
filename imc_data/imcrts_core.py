from collections import defaultdict
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple, Union

import geopandas as gpd
import pandas as pd
import requests
import numpy as np
from tqdm import tqdm
import imc_data

from shapely.geometry import LineString

IMCRTS_Response = Tuple[int, Optional[List[Dict[str, Any]]]]

DATA_ROOT = os.path.join(imc_data.RESOURCE_PATH, "metr_ic_sample")
STD_NODE_PATH = os.path.join(DATA_ROOT, "moct_node.shp")
STD_LINK_PATH = os.path.join(DATA_ROOT, "moct_link.shp")
STD_TURNINFO_PATH = os.path.join(DATA_ROOT, "TURNINFO.dbf")

IMCRTS_ROOT = os.path.join(DATA_ROOT, "IMCRTS")
IMCRTS_PICKLE_PATH = os.path.join(IMCRTS_ROOT, "imcrts_data.pickle")
IMCRTS_EXCEL_PATH = os.path.join(IMCRTS_ROOT, "imcrts_data.xlsx")

DATASET_ROOT = os.path.join(DATA_ROOT, "Dataset")
TRAFFIC_DATA_PATH = os.path.join(DATASET_ROOT, "metr-imc.h5")

DATASET_GRAPH_ROOT = os.path.join(DATASET_ROOT, "sensor_graph")
SENSOR_LIST_PATH = os.path.join(DATASET_GRAPH_ROOT, "graph_sensor_ids.txt")
DISTANCE_LIST_PATH = os.path.join(DATASET_GRAPH_ROOT, "distances_imc_2023.csv")
LINK_INFO_PATH = os.path.join(DATA_ROOT, "Linkers.shp")

PRIVATE_DECODED_KEY = "5HFe89gOZkcIZ/ZogD9zz18ZKcqBnu9nTIvf83zgORCxMx+SYz5RRGguMTi+zwrjolzlLWS/z363/7pyEVzUgw=="

logger = logging.getLogger(__name__)
os.makedirs(IMCRTS_ROOT, exist_ok=True)
os.makedirs(DATASET_GRAPH_ROOT, exist_ok=True)


def main():
    processor = DataProcessor()
    processor.import_all()
    processor.export_all()


def generate_links():
    """링크 사이의 연결을 표시하기 위한 링크 생성"""
    std_link: gpd.GeoDataFrame = gpd.read_file(STD_LINK_PATH)
    std_link.set_index("LINK_ID", inplace=True)
    print(std_link)
    links: IcLinkList = IcLinkList()

    lines = []
    for _, row in tqdm(links.data.iterrows(), total=links.data.shape[0]):
        fn = str(round(row["from"]))
        tn = str(round(row["to"]))

        line1 = std_link.loc[fn].geometry
        line2 = std_link.loc[tn].geometry
        lines.append(LineString([line1.coords[-1], line2.coords[0]]))

    ll = gpd.GeoDataFrame(geometry=lines)
    ll.to_file(os.path.join(DATA_ROOT, "Linkers.shp"))


class DataProcessor:
    def __init__(self) -> None:
        logger.info("Loading Standard Node...")
        self.std_node: gpd.GeoDataFrame = gpd.read_file(STD_NODE_PATH)
        logger.info("Loading Standard Link...")
        self.std_link: gpd.GeoDataFrame = gpd.read_file(STD_LINK_PATH)
        logger.info("Loading Standard Turn Info...")
        self.turn_info: gpd.GeoDataFrame = gpd.read_file(STD_TURNINFO_PATH)
        logger.info("Loading IMCRTS Data...")
        self.imcrts_data: pd.DataFrame = pd.read_pickle(IMCRTS_PICKLE_PATH)

        logger.info("Loading Node List...")
        self.imc_node_list: IcNodeList = IcNodeList()
        logger.info("Loading Link...")
        self.imc_link_list: IcLinkList = IcLinkList()
        logger.info("Loading Traffic Data...")
        self.traffic_data: IcTrafficDataset = IcTrafficDataset()

    def import_all(self):
        logger.info("Importing Nodes...")
        self.imc_node_list.import_from_raw(self.std_link)
        logger.info("importing Traffic Data...")
        self.traffic_data.import_from_imcrts_data(self.imcrts_data, self.imc_node_list)
        logger.info("Importing Links...")
        self.imc_link_list.import_from_raw(
            self.std_link, self.turn_info, self.imc_node_list
        )

    def export_all(self):
        self.traffic_data.export()
        self.imc_node_list.export_to_txt()
        self.imc_link_list.export_to_csv()


class IcNodeList:
    def __init__(self, path: str = SENSOR_LIST_PATH) -> None:
        self.file_path = path
        self.data: Set[str] = set()
        try:
            with open(self.file_path, "r") as file:
                self.data = set([str(int(item)) for item in file.readline().split(",")])
        except FileNotFoundError:
            pass

    def import_from_raw(self, std_link: gpd.GeoDataFrame):
        self.data: Set[str] = set(std_link["LINK_ID"].unique())

    def export_to_txt(self):
        with open(self.file_path, "w") as file:
            file.write(",".join(sorted(self.data)))

    @property
    def count(self):
        return len(self.data)


class IcLinkList:
    def __init__(self, path: str = DISTANCE_LIST_PATH) -> None:
        self.path = path
        self.data: pd.DataFrame = pd.DataFrame()
        if os.path.exists(path):
            self.data: pd.DataFrame = pd.read_csv(path)

    def export_to_csv(self):
        self.data.to_csv(self.path)

    def import_from_raw(
        self,
        std_link: gpd.GeoDataFrame,
        turn_info: gpd.GeoDataFrame,
        node_list: IcNodeList,
    ):
        indexed_turn_info = turn_info.set_index(["NODE_ID", "ST_LINK", "ED_LINK"])

        def get_turn_code(intersection_id, st_link_id, ed_link_id):
            try:
                result = indexed_turn_info.loc[
                    (intersection_id, st_link_id, ed_link_id)
                ]
                if type(result) == str:
                    return [result]
                elif type(result) == list:
                    return result
                elif type(result) == pd.Series:
                    return result.tolist()
                else:
                    raise Exception("Error")
            except KeyError:
                return []

        road_sensor_df = std_link.set_index("LINK_ID")
        to_rds = std_link.groupby("F_NODE")["LINK_ID"].apply(list).to_dict()

        distance_output = {"from": [], "to": [], "cost": []}
        missing_link_ids = []

        for _, start_rd_id in tqdm(
            enumerate(node_list.data), total=node_list.count, position=0
        ):
            try:
                start_rd_length: float = road_sensor_df.at[start_rd_id, "LENGTH"] / 2
            except KeyError:
                missing_link_ids.append(start_rd_id)
                continue

            is_id = road_sensor_df.at[start_rd_id, "T_NODE"]
            if is_id not in to_rds:
                continue

            for end_rd_id in tqdm(
                to_rds[is_id], total=len(to_rds), leave=False, position=1
            ):
                turn_codes: List[str] = get_turn_code(is_id, start_rd_id, end_rd_id)
                s = int(start_rd_id[-3])
                e = int(end_rd_id[-3])

                if (s % 2 == 1 and s + 1 == e) or (s % 2 == 0 and s - 1 == e):
                    if "011" in turn_codes or "012" in turn_codes:
                        pass
                    else:
                        continue
                if (
                    "003" in turn_codes
                    or "101" in turn_codes
                    or "102" in turn_codes
                    or "103" in turn_codes
                ):
                    continue

                end_rd_length: float = road_sensor_df.at[end_rd_id, "LENGTH"] / 2

                distance_output["from"].append(start_rd_id)
                distance_output["to"].append(end_rd_id)
                distance_output["cost"].append(start_rd_length + end_rd_length)

        self.data = pd.DataFrame(distance_output)


class IcTrafficDataset:
    def __init__(self, path: str = TRAFFIC_DATA_PATH) -> None:
        self.path = path
        self.data: pd.DataFrame = pd.DataFrame()
        if os.path.exists(path):
            self.data: pd.DataFrame = pd.read_hdf(path, index_col="index")

    def export(self):
        self.data.to_hdf(self.path, key="data")

    def import_from_imcrts_data(
        self,
        imcrts_data: pd.DataFrame,
        node_list: IcNodeList,
        start_date: datetime = datetime(2023, 10, 1),
        end_date: datetime = datetime(2023, 11, 25),
    ):
        result: DefaultDict[str, List[int]] = defaultdict(list)
        result_index: List[str] = []
        traffic_date_group = imcrts_data.groupby("statDate")

        current_date = start_date
        with tqdm(total=((end_date - start_date).days + 1) * 24) as pbar:
            while current_date <= end_date:
                date_key = current_date.strftime("%Y-%m-%d")

                traffic_of_day: pd.DataFrame = traffic_date_group.get_group(date_key)
                traffic_of_day = traffic_of_day.set_index("linkID")
                for n in range(24):
                    col_key = "hour{:02d}".format(n)
                    date_index = current_date.strftime("%Y-%m-%d %H:%M:%S")

                    for link_ID in node_list.data:
                        value: int = np.nan
                        if link_ID in traffic_of_day.index:
                            traffic: str = traffic_of_day.loc[link_ID][col_key]
                            value = int(traffic)
                        result[link_ID].append(value)

                    result_index.append(date_index)
                    current_date += timedelta(hours=1)
                    pbar.update()

        self.data = pd.DataFrame(result, index=result_index)
        return self.data


class DataCollector:
    """
    Data.go.kr로부터 인천 도로 교통량 데이터를 특정 날짜 기간만큼 추출하고 저장
    """

    def __init__(
        self,
        url: str = "http://apis.data.go.kr/6280000/ICRoadVolStat/NodeLink_Trfc_DD",
        start_date: str = "20230101",
        end_date: str = "20231125",
        max_row_count: int = 5000,
    ) -> None:
        self.server_url = url
        self.max_row_count = max_row_count
        self.params = {
            "serviceKey": PRIVATE_DECODED_KEY,
            "pageNo": 1,
            "numOfRows": max_row_count,
            "YMD": "20240101",
        }
        self.start_date: datetime = datetime.strptime(start_date, "%Y%m%d")
        self.end_date: datetime = datetime.strptime(end_date, "%Y%m%d")

    def collect(self, need_file_save: bool = True) -> pd.DataFrame:
        """데이터를 수집하고 Pandas DataFrame형태로 변환

        Args:
            need_file_save (bool, optional): Wheter save result to file. Defaults to True.

        Returns:
            pd.DataFrame: Result of colloecting
        """
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
        if need_file_save:
            logger.info("Creating Pickle file...")
            df.to_pickle(IMCRTS_PICKLE_PATH)
            logger.info(f"{IMCRTS_PICKLE_PATH} is created")
            logger.info("Creating Excel file...")
            df.to_excel(IMCRTS_EXCEL_PATH)
            logger.info(f"{IMCRTS_EXCEL_PATH} is created")

        return df

    def get_data(self, params: Dict[str, Any]) -> IMCRTS_Response:
        """Request a data from server

        Args:
            params (Dict[str, Any]): parameters for web API request

        Returns:
            IMCRTS_Response: Response for request from server
        """
        res = requests.get(url=self.server_url, params=params)
        data: Optional[List[Dict[str, Any]]] = None
        if res.status_code == 200:
            raw = res.json()
            if len(raw["response"]["body"]["items"]) > 0:
                data = raw["response"]["body"]["items"]

                if len(data) > self.max_row_count:
                    message = f"Length of Data at {params['YMD']} is sliced to {self.max_row_count}"
                    logger.warning(message)
            else:
                logger.warning(f"No data at {params['YMD']}")
        else:
            print(res.text)

        return (res.status_code, data)
