from box import Box
import logging
import os
import time
from collections import defaultdict
from datetime import datetime, timedelta, date
from typing import Any, DefaultDict, Dict, List, Optional, Set, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


def execute_metr_converter(config: Box):
    logger.info("Loading raw files...")
    std_link: gpd.GeoDataFrame = gpd.read_file(config.std_link_file_path)
    turn_info: gpd.GeoDataFrame = gpd.read_file(config.std_turn_info_file_path)
    imcrts_data: pd.DataFrame = pd.read_pickle(config.imcrts_data_file_path)

    logger.info("Preparing data...")
    imc_node_list: GraphSensorIDs = GraphSensorIDs(
        os.path.join(config.dataset_root_path, config.sensor_ids_file_path)
    )
    imc_link_list: DistanceIMC = DistanceIMC(
        os.path.join(config.dataset_root_path, config.distance_file_path)
    )
    traffic_data: MetrIMC = MetrIMC(
        os.path.join(config.dataset_root_path, config.traffic_volume_file_path)
    )

    logger.info("Converting node data...")
    imc_node_list.import_from_raw(std_link)
    logger.info("Converting traffic data...")
    traffic_data.import_from_imcrts_data(
        imcrts_data,
        imc_node_list,
        start_date=config.start_date,
        end_date=config.end_date,
    )
    logger.info("Converting distance data...")
    imc_link_list.import_from_raw(std_link, turn_info, imc_node_list)

    logger.info("Exporting traffic data...")
    traffic_data.export()
    logger.info("Exporting sensor ids data...")
    imc_node_list.export_to_txt()
    logger.info("Exporting distance data...")
    imc_link_list.export_to_csv()


class GraphSensorIDs:
    def __init__(self, path: str) -> None:
        self.file_path = path
        self.data: Set[str] = set()

    def export_to_txt(self):
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

        with open(self.file_path, "w") as file:
            file.write(",".join(sorted(self.data)))
        logger.info(f"Sensor list is exported to {self.file_path}")

    def import_from_raw(self, std_link: gpd.GeoDataFrame):
        self.data: Set[str] = set(std_link["LINK_ID"].unique())

    @property
    def count(self):
        return len(self.data)


class DistanceIMC:
    def __init__(self, path: str) -> None:
        self.path = path
        self.data: pd.DataFrame = pd.DataFrame()

    def export_to_csv(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        self.data.to_csv(self.path, index=False)
        logger.info(f"Distance list is exported to {self.path}")

    def import_from_raw(
        self,
        std_link: gpd.GeoDataFrame,
        turn_info: gpd.GeoDataFrame,
        node_list: GraphSensorIDs,
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


class MetrIMC:
    def __init__(self, path: str) -> None:
        self.path = path
        self.data: pd.DataFrame = pd.DataFrame()

    def export(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        self.data.to_hdf(self.path, key="data")
        logger.info(f"Traffic dataset is exported to {self.path}")

    def import_from_imcrts_data(
        self,
        imcrts_data: pd.DataFrame,
        node_list: GraphSensorIDs,
        start_date: date = date(2023, 10, 1),
        end_date: date = date(2023, 11, 25),
    ):
        start_datetime = datetime(start_date.year, start_date.month, start_date.day)
        end_datetime = datetime(end_date.year, end_date.month, end_date.day)

        result: DefaultDict[str, List[int]] = defaultdict(list)
        result_index: List[str] = []
        traffic_date_group = imcrts_data.groupby("statDate")

        current_datetime = start_datetime
        total_days = (end_datetime - start_datetime).days + 1
        with tqdm(total=total_days * 24) as pbar:
            while current_datetime <= end_datetime:
                date_key = current_datetime.strftime("%Y-%m-%d")

                traffic_of_day: pd.DataFrame = traffic_date_group.get_group(date_key)
                traffic_of_day = traffic_of_day.set_index("linkID")
                for n in range(24):
                    col_key = "hour{:02d}".format(n)
                    date_index = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
                    pbar.desc = date_index

                    for link_ID in node_list.data:
                        value: int = np.nan
                        if link_ID in traffic_of_day.index:
                            traffic: str = traffic_of_day.loc[link_ID][col_key]
                            value = int(traffic)
                        result[link_ID].append(value)

                    result_index.append(date_index)
                    current_datetime += timedelta(hours=1)
                    
                    pbar.update()

        self.data = pd.DataFrame(result, index=result_index)
        return self.data
