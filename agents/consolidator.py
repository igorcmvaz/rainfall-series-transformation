import logging
import pickle
import shutil
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pandas as pd

from agents.calculator import IndicesCalculator, estimate_combinations
from agents.exporters import CSVExporter
from agents.extractors import NetCDFExtractor
from agents.validators import CoordinatesValidator, PathValidator
from globals.constants import RECOVERY_FILENAME_FORMAT, TEMP_FILE_NAME
from globals.errors import InvalidSourceFileError
from globals.types import MetaData, PrecipitationSeries, RecoveryData

logger = logging.getLogger("rainfall_transformation")


class Consolidator:

    cities: dict[str, dict[str, dict[str, float]]]
    scenarios: dict[str, dict[str, str]]
    models: list[str]
    source_dir: Path
    state: dict[str, int]

    def __init__(
            self,
            cities: dict[str, dict[str, dict[str, float]]],
            scenarios: dict[str, dict[str, str]],
            models: list[str],
            source_dir: Path,
            csv_generator: CSVExporter | None = None) -> None:
        self.models = models
        self.scenarios = scenarios
        self.cities = cities
        self.source_dir = source_dir
        self.state = {
            "total": estimate_combinations(models, scenarios, cities),
            "processed": 0,
            "error": 0,
            "skipped": 0,
            "success": 0,
            "success_rate": 0,
            "process_rate": 0
        }
        self.csv_generator = csv_generator
        self.temp_dir = self._create_temp_dir()

    def _create_temp_dir(self) -> Path:
        """
        Creates a temporary directory for recovery files, if it does not already exist.

        Returns:
            Path: Path to the temporary directory.
        """
        temp_path = Path(self.source_dir.parent, "temp")
        temp_path.mkdir(exist_ok=True)
        logger.debug(f"Created temporary recovery directory at '{temp_path.resolve()}'")
        return temp_path

    def _count_error(self, **kwargs: dict[str, Any]) -> None:
        """
        Updates the internal state of the class registering an error in processing the
        current operation.

        Args:
            **kwargs (dict[str, Any]): Parameters to be formatted as 'key=value' and
            included in the logged message.
        """
        self.state["processed"] += 1
        self.state["error"] += 1
        details = [f"{key}={value}" for key, value in kwargs.items()]
        logger.error(
            f"[{self.state['processed']}/{self.state['total']}] Error during processing for"
            f" {', '.join(details)}")

    def _count_processed(self, **kwargs: dict[str, Any]) -> None:
        """
        Updates the internal state of the class registering a success in processing the
        current operation.

        Args:
            **kwargs (dict[str, Any]): Parameters to be formatted as 'key=value' and
            included in the logged message.
        """
        self.state["processed"] += 1
        details = [f"{key}={value}" for key, value in kwargs.items()]
        logger.info(
            f"[{self.state['processed']}/{self.state['total']}] Completed processing for "
            f"{', '.join(details)}")

    def _set_final_state(self) -> None:
        """
        Updates the elements from internal state of the class that are derived from others,
        and computes metrics related to them, used at the end of operations.
        """
        self.state["skipped"] = self.state["total"] - self.state["processed"]
        self.state["success"] = self.state["processed"] - self.state["error"]
        self.state["process_rate"] = 100*self.state['processed']/self.state['total']
        self.state["success_rate"] = 100*(
            self.state['success'])/(self.state['total'] - self.state['skipped'])
        logger.info(
            f"Processed {self.state['processed']} combinations, from a total of "
            f"{self.state['total']}, skipped {self.state['skipped']} item(s), encountered "
            f"{self.state['error']} error(s). "
            f"Success rate: {self.state['success_rate']:.2f}%. "
            f"Effective processed rate: {self.state['process_rate']:.2f}%")

    def _insert_metadata(self, dataframe: pd.DataFrame, **kwargs: dict[str, Any]) -> None:
        """
        Inserts the provided metadata fields into the given dataframe.

        Args:
            dataframe (pd.DataFrame): Dataframe to be updated with the provided fields.
            **kwargs (dict[str, Any]): Fields and corresponding values to be added to the
            dataframe. Field names are capitalized, values are kept as is.
        """
        for key, value in kwargs.items():
            dataframe[key.capitalize()] = value

    def _dump_recovery_data(
            self, model: str, scenario: str, recovery_data: RecoveryData) -> None:
        """
        Dumps precipitation related to a climate model and scenario into a binary recovery
        file.

        Args:
            model (str): Climate model related to the data.
            scenario (str): Climate scenario related to the data.
            recovery_data (RecoveryData): Recovery data (precipitation series and metadata)
            to be stored in the file.
        """
        if not recovery_data:
            return
        output_file_path = Path(
            self.temp_dir, RECOVERY_FILENAME_FORMAT.format(model=model, scenario=scenario))
        dirty_path = Path(self.temp_dir, TEMP_FILE_NAME)
        start_time = time.perf_counter()
        with open(dirty_path, "wb") as temp_file:
            pickle.dump(recovery_data, temp_file, protocol=pickle.HIGHEST_PROTOCOL)
        dirty_path.replace(output_file_path)
        logger.info(
            f"Successfully saved recovery file at '{output_file_path.name}' in "
            f"{time.perf_counter() - start_time:.2f}s")

    def _validate_recovery_path(self, model: str, scenario: str) -> Path | None:
        """
        Validates the path to a recovery file given a climate model and scenario.

        Args:
            model (str): Climate model related to the data.
            scenario (str): Climate scenario related to the data.

        Returns:
            Path | None: Path to the recovery file, if it exists, else None.
        """
        recovery_file = Path(
            self.temp_dir, RECOVERY_FILENAME_FORMAT.format(model=model, scenario=scenario))
        if recovery_file.is_file():
            return recovery_file
        logger.debug(
            f"No recovery file found for model '{model}', scenario '{scenario}', "
            f"proceeding with normal extraction")
        return None

    def _recover_data_from_file(self, path_to_file: Path) -> RecoveryData:
        """
        Recovers data from a binary file in a given path.

        Args:
            path_to_file (Path): Path to the recovery file.

        Returns:
            RecoveryData: Data (precipitation series and metadata) recovered from the file.
        """
        start_time = time.perf_counter()
        with open(path_to_file, "rb") as file:
            recovered_data = pickle.load(file)
        logger.info(
            f"Retrieved data from recovery file '{path_to_file.name}' in "
            f"{time.perf_counter() - start_time:.2f}s")
        return recovered_data

    def clear_temp_files(self) -> None:
        """Deletes the temporary directory along with all temporary files."""
        shutil.rmtree(self.temp_dir)

    def generate_precipitation_dataset(
            self) -> Generator[tuple[PrecipitationSeries, MetaData]]:
        """
        Returns a generator that yields precipitation data series.

        The generator yields arrays of tuples containing datetime and precipitation values
        for each city, climate model and scenario, as well as metadata. If a CSV exporter is
        available, the precipitation data series is exported before being yielded.

        Yields:
            Generator[tuple[PrecipitationSeries, MetaData]]: Tuple where the first element
            is an array of (datetime, precipitation) tuples, and the second element is a
            dictionary with metadata about the dataset.
        """
        logger.info(
            f"Starting extraction of precipitation data for {len(self.cities)} cities, "
            f"{len(self.models)} climate model(s) and {len(self.scenarios)} scenario(s)"
            )
        for model in self.models:
            for scenario in self.scenarios:
                cities_to_process = self.cities.copy()
                for_recovery: RecoveryData = {}
                if (recovery_path := self._validate_recovery_path(
                        model, scenario)) is not None:
                    recovered_data = self._recover_data_from_file(recovery_path)
                    for city, content in recovered_data.items():
                        if city not in cities_to_process:
                            continue
                        self._count_processed(**content["metadata"])
                        cities_to_process.pop(city, None)
                        yield content["data"], content["metadata"]
                if not cities_to_process:
                    continue
                try:
                    extractor = NetCDFExtractor(
                        PathValidator.validate_precipitation_source_path(
                            model, scenario, self.source_dir))
                except InvalidSourceFileError:
                    logger.warning(
                        f"File corresponding to model '{model}' and scenario '{scenario}' "
                        f"was not found, skipping")
                    continue
                for city_name, details in cities_to_process.items():
                    latitude, longitude = CoordinatesValidator.get_coordinates(details)
                    data_series = extractor.extract_precipitation(latitude, longitude)
                    if not data_series.any():
                        self._count_error(
                            city=city_name,
                            model=model,
                            scenario=scenario,
                            latitude=latitude,
                            longitude=longitude)
                        continue
                    metadata: dict[str, str | float] = {
                        "city": city_name,
                        "model": model,
                        "scenario": scenario,
                        "latitude": latitude,
                        "longitude": longitude
                    }
                    if self.csv_generator is not None:
                        self.csv_generator.generate_csv(
                            data_series,
                            metadata["city"],
                            metadata["model"],
                            metadata["scenario"])
                    self._count_processed(**metadata)
                    for_recovery[city_name] = {"data": data_series, "metadata": metadata}
                    yield data_series, metadata
                self._dump_recovery_data(model, scenario, for_recovery)
            logger.info(f"Completed processing of model '{model}'")
        self._set_final_state()

    def generate_all_precipitation_series(self) -> None:
        """
        Generates all precipitation data series by exhausting the generator from
        `generate_precipitation_dataset()`.
        """
        start_time = time.perf_counter()
        for _ in self.generate_precipitation_dataset():
            pass
        logger.info(f"Completed process in {time.perf_counter() - start_time:.2f}s")

    def generate_precipitation_indices(self) -> Generator[pd.DataFrame]:
        """
        Returns a generator that yields dataframes containing precipitation indices for each
        city, climate model and scenario.

        Yields:
            Generator[pd.DataFrame]: Dataframe containing precipitation indices computed
            from data available in NetCDF4 file, for each specified city, climate model and
            scenario.
        """
        for data_series, metadata in self.generate_precipitation_dataset():
            indices = IndicesCalculator(data_series).compute_climate_indices()
            self._insert_metadata(indices, **metadata)
            yield indices

    def consolidate_indices_dataset(self) -> pd.DataFrame:
        """
        Concatenates the results of all precipitation indices datasets into a single
        dataframe.

        Returns:
            pd.DataFrame: Dataframe containing the data relative to all cities, models and
            scenarios that are available, generated by `generate_precipitation_dataset()`.
        """
        start_time = time.perf_counter()
        result = pd.concat(
            self.generate_precipitation_indices(),
            ignore_index=True,
            copy=False)
        logger.info(f"Completed process in {time.perf_counter() - start_time:.2f}s")
        return result
