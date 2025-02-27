from pathlib import Path

import json
from typing import Any
import pandas as pd
from datetime import datetime

SAMPLE_JSON_PATH = Path(__file__).parent / "sample_precipitation.json"


class PrecipitationGenerator:

    df: pd.DataFrame
    metadata: dict[str, Any]
    climate_indices: dict[int, dict[str, float]]

    def __init__(self):
        with open(SAMPLE_JSON_PATH) as file:
            content = json.load(file)
        self.df = pd.DataFrame(
            content["data"],
            columns=content["schema"]
            )
        self.df["date"] = pd.to_datetime(self.df["date"], format=content["date_format"])
        self.metadata = {
            "city": content["city"],
            "model": content["model"],
            "scenario": content["scenario"],
            "start_date": datetime.strptime(content["start_date"], content["date_format"]),
            "end_date": datetime.strptime(content["end_date"], content["date_format"]),
            "frequency": content["frequency"]
        }
        self.climate_indices = {
            int(key): value for key, value in content["climate_indices"].items()
        }
        self._set_auxiliary_columns()

    def _set_auxiliary_columns(self) -> None:
        self.df["year"] = self.df["date"].dt.year
        self.df["month"] = self.df["date"].dt.month
        self.df["dry_days"] = self.df["precipitation"] < 1
        self.df["wet_days"] = self.df["precipitation"] >= 1

    def all_data(self) -> pd.DataFrame:
        return self.df.copy()

    def by_year(self, year: int) -> pd.DataFrame:
        result = self.df[self.df["year"] == year].copy()
        result.reset_index(drop=True, inplace=True)
        return result
