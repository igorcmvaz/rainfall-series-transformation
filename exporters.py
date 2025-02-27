import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from constants import PARQUET_CONF

logger = logging.getLogger("rainfall_transformation")


class BaseExporter:

    parent_output_dir: Path

    def __init__(self, parent_output_dir: Path) -> None:
        self.parent_output_dir = parent_output_dir

    def _get_base_path(self) -> str:
        return f"{datetime.now().strftime('%Y-%m-%dT%H-%M')}-"


class ParquetExporter(BaseExporter):

    def _get_base_file_name(self) -> str:
        return self._get_base_path() + "consolidated.parquet"

    def generate_parquet(self, dataframe: pd.DataFrame) -> None:
        output_path = Path(self.parent_output_dir, self._get_base_file_name())
        dataframe.to_parquet(output_path, index=False, **PARQUET_CONF)
        logger.info(f"Successfully exported dataframe to file at '{output_path.resolve()}'")


class CSVExporter(BaseExporter):

    def __init__(self, parent_output_dir: Path) -> None:
        super().__init__(parent_output_dir)
        self.output_dir = Path(self.parent_output_dir, self._get_base_directory_name())
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_base_directory_name(self) -> str:
        return self._get_base_path() + "output"

    def _get_file_path(self, city_name: str, model: str, period_label: str) -> Path:
        return Path(
            self.output_dir, f"{city_name}_{model}_{period_label}").with_suffix(".csv")

    def generate_csv(
            self,
            data_series: np.ndarray[tuple[datetime, float]],
            city_name: str,
            model: str,
            period_label: str,
            schema: list[str] = ["date", "precipitation"]) -> None:
        df = pd.DataFrame(data_series, columns=schema)
        output_path = self._get_file_path(city_name, model, period_label)
        df.to_csv(
            output_path, sep=",", index=False, encoding="utf-8", date_format="%Y-%m-%d")
        logger.info(
            f"Successfully exported precipitation data series to file at "
            f"'{output_path.resolve()}'")


class NetunoExporter(CSVExporter):

    def _get_file_path(self, city_name, model, period_label) -> Path:
        return Path(
            self.output_dir,
            f"(Netuno){city_name}_{model}_{period_label}").with_suffix(".csv")

    def generate_csv(
            self,
            data_series: np.ndarray[tuple[datetime, float]],
            city_name: str,
            model: str,
            period_label: str) -> None:
        return super().generate_csv(
            [precipitation for _, precipitation in data_series],
            city_name,
            model,
            period_label,
            schema=["precipitation"])
