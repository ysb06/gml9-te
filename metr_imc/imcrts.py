from box import Box
import time
from datetime import datetime, timedelta, date
from typing import Any, Dict, List, Optional, Tuple, Union
from os.path import join

import pandas as pd
import requests
import logging

logger = logging.getLogger(__name__)


def generate_imcrts_data(config: Box):
    if config.need_update:
        key = ""
        with open(config.private_key_path, mode="r") as file:
            key = file.readline()

        collector = IMCRTSCollector(
            url=config.service_url,
            key=key,
            start_date=config.start_date,
            end_date=config.end_date,
            row_numbers=config.row_numbers,
            save_path=config.save_path,
        )
        collector.collect()
    else:
        logger.warning("Using previous updated data")


class IMCRTSCollector:
    def __init__(
        self,
        url: str,
        key: str,
        start_date: Union[datetime, date],
        end_date: Union[datetime, date],
        row_numbers: int,
        save_path: str,
    ) -> None:
        logger.info("Collecting...")
        self.url = url
        self.start_date: Union[datetime, date] = start_date
        self.end_date: Union[datetime, date] = end_date
        self.save_path = save_path

        self.params = {
            "serviceKey": key,
            "pageNo": 1,
            "numOfRows": row_numbers,
            "YMD": "20240101",
        }

    def collect(self) -> None:
        """
        데이터를 수집하고 Pandas DataFrame형태로 변환 후 Pickle 및 Excel형태로 저장
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
        logger.info("Creating Pickle...")
        path = join(self.save_path, "imcrts_data.pickle")
        df.to_pickle(path)
        logger.info(f"{path} is created")
        logger.info("Creating Excel...")
        path = join(self.save_path, "imcrts_data.xlsx")
        df.to_excel(path)
        logger.info(f"{path} is created")

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
        res = requests.get(url=self.url, params=params)
        data: Optional[List[Dict[str, Any]]] = None
        if res.status_code == 200:
            raw = res.json()
            if len(raw["response"]["body"]["items"]) > 0:
                data = raw["response"]["body"]["items"]

                if len(data) > self.params["numOfRows"]:
                    message = f"Length of Data at {params['YMD']} is {data['response']['body']['items']} but sliced to {self.params['numOfRows']}"
                    logger.warning(message)
            else:
                logger.warning(f"No data at {params['YMD']}")
        else:
            print(res.text)

        return (res.status_code, data)
