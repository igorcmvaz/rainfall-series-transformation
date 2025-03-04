import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from globals.constants import PARQUET_CONF

logger = logging.getLogger("rainfall_transformation")


class BasePrecipitationExporter:

    parent_output_dir: Path

    def __init__(self, parent_output_dir: Path) -> None:
        self.parent_output_dir = parent_output_dir

    def _get_base_path(self) -> str:
        """
        Retrieves the base path new items, with formatted datetime.

        Returns:
            str: Base name for new items.
        """
        return f"{datetime.now().strftime('%Y-%m-%dT%H-%M')}-"


class ParquetExporter(BasePrecipitationExporter):

    def _get_base_file_name(self) -> str:
        """
        Retrieves the base file name for new Parquet files.

        Returns:
            str: Base name for new Parquet files.
        """
        return self._get_base_path() + "consolidated.parquet"

    def generate_parquet(self, dataframe: pd.DataFrame) -> None:
        """
        Generates a new Parquet file from the contents in a given DataFrame.

        Note: Any index columns are not written to the output Parquet file. Advanced Parquet
        configuration arguments are defined in `globals.constants.PARQUET_CONF`.

        Args:
            dataframe (pd.DataFrame): Data frame to be exported to the Parquet file.
        """
        output_path = Path(self.parent_output_dir, self._get_base_file_name())
        dataframe.to_parquet(output_path, index=False, **PARQUET_CONF)
        logger.info(f"Successfully exported dataframe to '{output_path.resolve()}'")


class CSVExporter(BasePrecipitationExporter):

    def __init__(self, parent_output_dir: Path) -> None:
        super().__init__(parent_output_dir)
        self.output_dir = Path(self.parent_output_dir, self._get_base_directory_name())
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_base_directory_name(self) -> str:
        """
        Retrieves the base directory name where the output CSV files will be saved.

        Returns:
            str: Base name for the new directory.
        """
        return self._get_base_path() + "output"

    def _get_file_path(self, city_name: str, model: str, scenario: str) -> Path:
        """
        Generates a file name for a new CSV file, given city name, climate model and
        scenario.

        Args:
            city_name (str): Name of the city related to the data.
            model (str): Climate model related to the data.
            scenario (str): Climate scenario related to the data.

        Returns:
            Path: Path to the new CSV file to be created.
        """
        return Path(
            self.output_dir, f"{city_name}_{model}_{scenario}").with_suffix(".csv")

    def generate_csv(
            self,
            data_series: np.ndarray[tuple[datetime, float]],
            city_name: str,
            model: str,
            scenario: str,
            schema: list[str] = ["date", "precipitation"]) -> None:
        """
        Generates a CSV file from a given data series.

        Args:
            data_series (np.ndarray[tuple[datetime, float]]): Precipitation data series to
                be exported.
            city_name (str): Name of the city related to the data.
            model (str): Climate model related to the data.
            scenario (str): Climate scenario related to the data.
            schema (list[str], optional): Output schema (columns to be exported). Must have
            the same dimensions as the data series. Defaults to ["date", "precipitation"].
        """
        df = pd.DataFrame(data_series, columns=schema)
        output_path = self._get_file_path(city_name, model, scenario)
        df.to_csv(
            output_path, sep=",", index=False, encoding="utf-8", date_format="%Y-%m-%d")
        logger.info(
            f"Successfully exported precipitation data series to '{output_path.resolve()}'")


class NetunoExporter(CSVExporter):

    def _get_file_path(self, city_name: str, model: str, scenario: str) -> Path:
        """
        Generates a file name for a new CSV file, given city name, climate model and
        scenario, labeled with "(Netuno)".

        Args:
            city_name (str): Name of the city related to the data.
            model (str): Climate model related to the data.
            scenario (str): Climate scenario related to the data.

        Returns:
            Path: Path to the new CSV file to be created.
        """
        return Path(
            self.output_dir,
            f"(Netuno){city_name}_{model}_{scenario}").with_suffix(".csv")

    def generate_csv(
            self,
            data_series: np.ndarray[tuple[datetime, float]],
            city_name: str,
            model: str,
            scenario: str) -> None:
        """
        Generates a CSV file from a given data series, including only precipitation data,
        no other columns and no headers.

        Args:
            data_series (np.ndarray[tuple[datetime, float]]): Precipitation data series to
                be exported.
            city_name (str): Name of the city related to the data.
            model (str): Climate model related to the data.
            scenario (str): Climate scenario related to the data.
        """
        super().generate_csv(
            [precipitation for _, precipitation in data_series],
            city_name,
            model,
            scenario,
            schema=["precipitation"])


class JSONCoordinatesExporter:

    @staticmethod
    def generate_json(coordinates: dict, output_path: Path) -> None:
        """
        Generates a JSON file in a given output path, with the given coordinates content.

        Args:
            coordinates (dict): Coordinates data and city details to be exported.
            output_path (Path): Path where the new JSON file will be saved.
        """
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(coordinates, file, indent=2, ensure_ascii=False, sort_keys=True)
        logger.info(f"Exported new coordinates file to '{output_path.resolve()}'")
