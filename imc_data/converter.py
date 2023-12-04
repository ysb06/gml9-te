import os
from typing import List
import imc_data
import pandas as pd
import geopandas as gpd
import tqdm

IMCRTS_DATA_ROOT = os.path.join(imc_data.RESOURCE_PATH, "IMCRTS")
IMCRTS_TRAFFIC_DATA_PATH = os.path.join(IMCRTS_DATA_ROOT, "imcrts_data.pickle")
IMCRTS_NODE_PATH = os.path.join(IMCRTS_DATA_ROOT, "imcrts_node.shp")
IMCRTS_LINK_PATH = os.path.join(IMCRTS_DATA_ROOT, "imcrts_link.shp")
TURN_INFO_PATH = os.path.join(
    imc_data.RESOURCE_PATH, "[2023-11-13]NODELINKDATA/TURNINFO.dbf"
)

OUTPUT_ROOT = os.path.join(imc_data.RESOURCE_PATH, "IMCRTS_Dataset")
GRAPH_OUPUT_ROOT = os.path.join(OUTPUT_ROOT, "sensor_graph")
# METR-LA, PEMS-Bay와 같은 구조
TS_OUTPUT_PATH = os.path.join(OUTPUT_ROOT, "imcrts_df.pickle")
DISTANCE_OUTPUT_PATH = os.path.join(GRAPH_OUPUT_ROOT, "distance_imc.csv")
ID_LIST_OUTPUT_PATH = os.path.join(GRAPH_OUPUT_ROOT, "graph_sensor_ids.txt")
os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(GRAPH_OUPUT_ROOT, exist_ok=True)


class Converter:
    def __init__(self) -> None:
        self.traffic_data: pd.DataFrame = pd.read_pickle(IMCRTS_TRAFFIC_DATA_PATH)
        self.node_data: gpd.GeoDataFrame = gpd.read_file(IMCRTS_NODE_PATH)
        self.link_data: gpd.GeoDataFrame = gpd.read_file(IMCRTS_LINK_PATH)
        self.turn_info: gpd.GeoDataFrame = gpd.read_file(TURN_INFO_PATH)
        self.turn_info_by_is = {
            is_id: group for is_id, group in self.turn_info.groupby("NODE_ID")
        }

    def run(self):
        # 명원님 작업 그대로
        ts_output: pd.DataFrame = pd.read_pickle(
            os.path.join(IMCRTS_DATA_ROOT, "imcrts_df.pickle")
        )
        rd_id_output = list(ts_output.columns)

        # 센서는 도로 중앙에 있다고 가정
        # 도로 중앙에서 시작해 교차로에 연결된 다른 도로의 중앙까지 거리
        # 도로 길이의 반과 연결된 도로 길이의 반을 합침
        distance_output = {"from": [], "to": [], "cost": []}

        # is_data = self.node_data.set_index("NODE_ID")
        rd_data = self.link_data.set_index("LINK_ID")

        f_is_group = self.link_data.groupby("F_NODE")
        to_rds = {}
        def add_out_links_to_node(data: pd.DataFrame):
            to_rds[data.iloc[0]["F_NODE"]] = list(data["LINK_ID"])
        f_is_group.apply(add_out_links_to_node)

        missing_link_ids = []
        for _, start_rd_id in tqdm.tqdm(
            enumerate(rd_id_output), total=len(rd_id_output)
        ):
            try:
                start_rd_length: float = rd_data.at[start_rd_id, "LENGTH"] / 2
            except KeyError:
                missing_link_ids.append(start_rd_id)
                continue

            is_id = rd_data.at[start_rd_id, "T_NODE"]
            if is_id not in to_rds:
                continue
            for end_rd_id in to_rds[is_id]:
                turn_codes: List[str] = self.get_turn_code(is_id, start_rd_id, end_rd_id)
                # is_id, start_rd_id, end_rd_id
                s = int(start_rd_id[-3])
                e = int(end_rd_id[-3])
                
                if (s % 2 == 1 and s + 1 == e) or s % 2 == 0 and s - 1 == e:
                    if "011" in turn_codes or "012" in turn_codes:
                        pass
                    else:
                        continue
                if "003" in turn_codes or "101" in turn_codes or "102" in turn_codes or "103" in turn_codes:
                    continue

                end_rd_length: float = rd_data.at[end_rd_id, "LENGTH"] / 2

                distance_output["from"].append(start_rd_id)
                distance_output["to"].append(end_rd_id)
                distance_output["cost"].append(start_rd_length + end_rd_length)

        ts_output.to_pickle(TS_OUTPUT_PATH)
        text = ", ".join(rd_id_output)
        with open(ID_LIST_OUTPUT_PATH, "w") as file:
            file.write(text)
        distance_output = pd.DataFrame(distance_output)
        distance_output.to_csv(DISTANCE_OUTPUT_PATH, index=False)
        print(distance_output)
        print(f"Unknown Links: {missing_link_ids}")
        print("Those links are net included in results")
    
    def get_turn_code(self, node_id, st_link_id, ed_link_id):
        if node_id not in self.turn_info_by_is:
            return []
        else:
            df = self.turn_info_by_is[node_id]
            result: pd.DataFrame = df[(df['ST_LINK'] == st_link_id) & (df['ED_LINK'] == ed_link_id)]

            return result['TURN_TYPE'].tolist()
