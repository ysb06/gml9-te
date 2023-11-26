import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import requests
from datetime import datetime, timedelta
import data_loader
import os
import pandas as pd
import time

logger = logging.getLogger(__name__)

SERVICE_URL = "http://apis.data.go.kr/6280000/ICRoadVolStat/NodeLink_Trfc_DD"
PRIVATE_ENCODED_KEY = r"5HFe89gOZkcIZ%2FZogD9zz18ZKcqBnu9nTIvf83zgORCxMx%2BSYz5RRGguMTi%2BzwrjolzlLWS%2Fz363%2F7pyEVzUgw%3D%3D"
PRIVATE_DECODED_KEY = "5HFe89gOZkcIZ/ZogD9zz18ZKcqBnu9nTIvf83zgORCxMx+SYz5RRGguMTi+zwrjolzlLWS/z363/7pyEVzUgw=="
MAX_ROW_COUNT = 5000

OUTPUT_ROOT_PATH = os.path.join(data_loader.RESOURCE_PATH, "IMCRTS")
if not os.path.exists(OUTPUT_ROOT_PATH):
    os.makedirs(OUTPUT_ROOT_PATH)


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
        data_excel_path = os.path.join(OUTPUT_ROOT_PATH, "imcrts.xlsx")
        data_pickle_path = os.path.join(OUTPUT_ROOT_PATH, "imcrts.pickle")
        logger.info("Creating Pickle...")
        df.to_pickle(data_pickle_path)
        logger.info(f"{data_pickle_path} is created")
        logger.info("Creating Excel...")
        df.to_excel(data_excel_path)
        logger.info(f"{data_excel_path} is created")

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
