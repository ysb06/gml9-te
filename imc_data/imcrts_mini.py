from collections import defaultdict
import os
from typing import Any, DefaultDict, Dict, List, Optional, Set, Union
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
import numpy as np
import tqdm

SAMPLE_PATH = "./resources/IMCRTS_Dataset/imcrts_df.pickle"

RAW_ROOT = "./resources/IMCRTS_Mini"
TRAFFIC_RAW_PATH = os.path.join(RAW_ROOT, "imcrts_data.pickle")
INTERSECTION_RAW_PATH = os.path.join(RAW_ROOT, "imcrts_node_mini.shp")
SENSOR_RAW_PATH = os.path.join(RAW_ROOT, "imcrts_link_mini.shp")

DATASET_ROOT = "/resources/IMCRTS_Mini_Dataset"
DATASET_GRAPH_ROOT = os.path.join(DATASET_ROOT, "sensor_graph")
IMCRTS_DF_PATH = os.path.join(DATASET_ROOT, "imcrts_df.h5")


class MiniGenerator:
    def __init__(self) -> None:
        print(pd.read_pickle(SAMPLE_PATH).head(3))
        self.traffic_raw: pd.DataFrame = pd.read_pickle(TRAFFIC_RAW_PATH)
        self.intersection_raw: gpd.GeoDataFrame = gpd.read_file(INTERSECTION_RAW_PATH)
        self.sensor_raw: gpd.GeoDataFrame = gpd.read_file(SENSOR_RAW_PATH)

    def run(self):
        target_sensor_list = sorted(list(set(self.sensor_raw["LINK_ID"].unique())))
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 11, 25)
        imcrts_df = self.generate_df(target_sensor_list, start_date, end_date)

        print(imcrts_df)

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
