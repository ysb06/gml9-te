from collections import defaultdict
import os
from typing import Any, DefaultDict, List, Set, Union
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
import numpy as np
import tqdm
import logging

RAW_ROOT = "./resources/IMCRTS_Mini"
TRAFFIC_RAW_PATH = os.path.join(RAW_ROOT, "imcrts_data.pickle")
INTERSECTION_RAW_PATH = os.path.join(RAW_ROOT, "imcrts_node_mini.shp")
SENSOR_RAW_PATH = os.path.join(RAW_ROOT, "imcrts_link_mini.shp")
TURN_INFO_RAW_PATH = os.path.join(RAW_ROOT, "TURNINFO.dbf")
IMCRTS_TURN_INFO_RAW_PATH = os.path.join(RAW_ROOT, "imcrts_turninfo.dbf")

DATASET_ROOT = "./resources/IMCRTS_Mini_Dataset"
DATASET_GRAPH_ROOT = os.path.join(DATASET_ROOT, "sensor_graph")
IMCRTS_DF_HDF_PATH = os.path.join(DATASET_ROOT, "imcrts_df.h5")
IMCRTS_DF_PKL_PATH = os.path.join(DATASET_ROOT, "imcrts_df.pickle")
IMCRTS_DF_XLS_PATH = os.path.join(DATASET_ROOT, "imcrts_df.xlsx")
ID_LIST_OUTPUT_TXT_PATH = os.path.join(DATASET_GRAPH_ROOT, "graph_sensor_ids.txt")
ID_LIST_OUTPUT_BIN_PATH = os.path.join(DATASET_GRAPH_ROOT, "graph_sensor_ids")
DISTANCE_CSV_PATH = os.path.join(DATASET_GRAPH_ROOT, "distance_imc.csv")
DISTANCE_NOSPACE_CSV_PATH = os.path.join(
    DATASET_GRAPH_ROOT, "distance_imc_no_space.csv"
)

os.makedirs(DATASET_GRAPH_ROOT, exist_ok=True)

logger = logging.getLogger(__name__)


class MiniGenerator:
    def __init__(self) -> None:
        self.traffic_raw: pd.DataFrame = pd.read_pickle(TRAFFIC_RAW_PATH)
        self.intersection_raw: gpd.GeoDataFrame = gpd.read_file(INTERSECTION_RAW_PATH)
        self.sensor_raw: gpd.GeoDataFrame = gpd.read_file(SENSOR_RAW_PATH)
        self.turn_info: gpd.GeoDataFrame = gpd.read_file(TURN_INFO_RAW_PATH)
        self.imcrts_turn_info: gpd.GeoDataFrame = self.refine_turn_info(self.turn_info, "16[1-8]")
        self.indexed_turn_info = self.imcrts_turn_info.groupby("NODE_ID")

        self.imcrts_turn_info.to_file(IMCRTS_TURN_INFO_RAW_PATH)

    def run(self):
        # graph_sensor_ids
        logger.info("Generating Sensor List...")
        target_sensor_set: Set[str] = set(self.sensor_raw["LINK_ID"].unique())
        target_sensor_list: List[str] = sorted(list(target_sensor_set))
        with open(ID_LIST_OUTPUT_TXT_PATH, "w") as file:
            file.write(", ".join(target_sensor_list))
        with open(ID_LIST_OUTPUT_BIN_PATH, "w") as file:
            file.write(",".join(target_sensor_list))  # 원래 데이터 형식

        # imcrts_df
        logger.info("Generating Traffic Data...")
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 11, 25)
        imcrts_df = self.generate_df(target_sensor_list, start_date, end_date)
        imcrts_df.interpolate()  # 보간 여부
        logger.info(f"Saving to hdf...")
        imcrts_df.to_hdf(IMCRTS_DF_HDF_PATH, "imc")  # 원래 데이터 형식에 맞추어
        logger.info(f"Saving to pickle...")
        imcrts_df.to_pickle(IMCRTS_DF_PKL_PATH)  # 이전 코드 호환용
        logger.info(f"Saving to excel...")
        imcrts_df.to_excel(IMCRTS_DF_XLS_PATH)  # 쉬운 데이터 확인을 위한 용도

        # distance_imc
        logger.info("Generating Graph and Distance...")
        distances_df = self.generate_distances(target_sensor_set)
        logger.info(f"Saving to csv...")
        distances_df.to_csv(DISTANCE_CSV_PATH, index=False)
        distances_df.to_csv(DISTANCE_NOSPACE_CSV_PATH, index=False)

    def refine_turn_info(self, df: pd.DataFrame, pattern: str) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
        return df[df["NODE_ID"].str.contains(pattern)]

    def generate_distances(self, sensor_set: Set[str]):
        def get_turn_code(node_id, st_link_id, ed_link_id):
            if node_id not in self.indexed_turn_info:
                return []
            else:
                df = self.indexed_turn_info[node_id]
                result: pd.DataFrame = df[
                    (df["ST_LINK"] == st_link_id) & (df["ED_LINK"] == ed_link_id)
                ]
                return result["TURN_TYPE"].tolist()

        to_rds = {}

        def add_out_links_to_node(data: pd.DataFrame):
            to_rds[data.iloc[0]["F_NODE"]] = list(data["LINK_ID"])

        road_sensor_df = self.sensor_raw.set_index("LINK_ID")
        from_intersection_group = self.sensor_raw.groupby("F_NODE")
        from_intersection_group.apply(add_out_links_to_node)

        distance_output = {"from": [], "to": [], "cost": []}

        missing_link_ids = []
        with tqdm.tqdm(total=len(sensor_set), desc="Init", leave=True) as pbar:
            for _, start_rd_id in enumerate(sensor_set):
                pbar.desc = start_rd_id
                pbar.update()
                try:
                    start_rd_length: float = (
                        road_sensor_df.at[start_rd_id, "LENGTH"] / 2
                    )
                except KeyError:
                    missing_link_ids.append(start_rd_id)
                    continue

                is_id = road_sensor_df.at[start_rd_id, "T_NODE"]
                if is_id not in to_rds:
                    continue
                for end_rd_id in to_rds[is_id]:
                    turn_codes: List[str] = get_turn_code(is_id, start_rd_id, end_rd_id)
                    s = int(start_rd_id[-3])
                    e = int(end_rd_id[-3])

                    if (s % 2 == 1 and s + 1 == e) or s % 2 == 0 and s - 1 == e:
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

        return pd.DataFrame(distance_output)

    def generate_df(
        self,
        targets: Set[str],
        start_date: datetime,
        end_date: datetime,
        missing_value: Union[int, Any] = np.nan,
    ):
        result: DefaultDict[str, List[int]] = defaultdict(list)
        result_index: List[str] = []
        traffic_date_group = self.traffic_raw.groupby("statDate")

        current_date = start_date
        dd = end_date - start_date
        with tqdm.tqdm(total=(dd.days + 1) * 24, desc="Init", leave=True) as pbar:
            while current_date <= end_date:
                date_key = current_date.strftime("%Y-%m-%d")
                pbar.desc = date_key

                traffic_of_day: pd.DataFrame = traffic_date_group.get_group(date_key)
                traffic_of_day = traffic_of_day.set_index("linkID")
                for n in range(24):
                    col_key = "hour{:02d}".format(n)
                    date_index = current_date.strftime("%Y-%m-%d %H:%M:%S")

                    initial_value = missing_value
                    for link_ID in targets:
                        value: int = initial_value
                        if link_ID in traffic_of_day.index:
                            traffic: str = traffic_of_day.loc[link_ID][col_key]
                            if type(traffic) != str:
                                raise Exception(
                                    f"One or more value for {link_ID} on {date_index}"
                                )
                            value = int(traffic)
                        result[link_ID].append(value)

                    result_index.append(date_index)
                    current_date += timedelta(hours=1)
                    pbar.update()

        result_df = pd.DataFrame(result, index=result_index)
        return result_df
