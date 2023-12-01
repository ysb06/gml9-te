import os
import imc_data
import pandas as pd

IMCRTS_DATA_ROOT = os.path.join(imc_data.RESOURCE_PATH, "IMCRTS")
IMCRTS_TRAFFIC_DATA_PATH = os.path.join(IMCRTS_DATA_ROOT, "imcrts_data.pickle")
IMCRTS_NODE_PATH = os.path.join(IMCRTS_DATA_ROOT, "imcrts_node.pickle")
IMCRTS_LINK_PATH = os.path.join(IMCRTS_DATA_ROOT, "imcrts_link.pickle")

OUTPUT_ROOT = os.path.join(imc_data.RESOURCE_PATH, "IMCRTS_Dataset")
GRAPH_OUPUT_ROOT = os.path.join(OUTPUT_ROOT, "sensor_graph")
# METR-LA, PEMS-Bay와 같은 구조
TS_OUTPUT_PATH = os.path.join(OUTPUT_ROOT, "imcrts_df.pickle")
DISTANCE_OUTPUT_PATH = os.path.join(GRAPH_OUPUT_ROOT, "distance_imc.csv")
DISTANCE_OUTPUT_PATH = os.path.join(GRAPH_OUPUT_ROOT, "graph_sensor_ids.txt")

class Converter:
    def __init__(self) -> None:
        self.traffic_data = pd.read_pickle(IMCRTS_DATA_ROOT)
        self.node_data = pd.read_pickle(IMCRTS_NODE_PATH)
        self.link_data = pd.read_pickle(IMCRTS_LINK_PATH)

    def run(self):
        pass

