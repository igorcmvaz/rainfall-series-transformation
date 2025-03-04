import logging
from collections.abc import Generator
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agents.calculator import IndicesCalculator, estimate_combinations
from agents.exporters import CSVExporter
from agents.extractors import NetCDFExtractor
from agents.validators import CoordinatesValidator, PathValidator

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

    def generate_precipitation_dataset(
            self) -> Generator[tuple[np.ndarray[tuple[datetime, float]], dict[str, Any]]]:
        """
        Returns a generator that yields arrays of tuples containing datetime and
        precipitation values for each city, climate model and scenario, as well as metadata
        about the dataset.

        Yields:
            Generator[tuple[np.ndarray[tuple[datetime, float]], dict[str, Any]]]: Tuple
            where the first element is an array of (datetime, precipitation) tuples, and the
            second element is a dictionary with metadata about the dataset.
        """
        logger.info(
            f"Starting extraction of precipitation data for {len(self.cities)} cities, "
            f"{len(self.models)} climate model(s) and {len(self.scenarios)} scenario(s)"
            )
        for model in self.models:
            for scenario in self.scenarios:
                # TODO: check for temp file
                extractor = NetCDFExtractor(
                    PathValidator.validate_precipitation_source_path(
                        model, scenario, self.source_dir))
                for city_name, details in self.cities.items():
                    latitude, longitude = CoordinatesValidator.get_coordinates(details)
                    # TODO: cache results based on coordinates, to prevent re-extraction in case of same coordinates
                    data_series = extractor.extract_precipitation(latitude, longitude)
                    if not data_series.any():
                        self._count_error(
                            city=city_name,
                            model=model,
                            scenario=scenario,
                            target_coordinates=(latitude, longitude))
                        continue
                    metadata = {
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
                    yield data_series, metadata
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
            metadata["target_coordinates"] = (
                metadata.pop("latitude"), metadata.pop("longitude"))
            self._count_processed(**metadata)
            yield indices

    def consolidate_dataset(self) -> pd.DataFrame:
        """
        Concatenates the results of all precipitation indices datasets into a single
        dataframe.

        Returns:
            pd.DataFrame: Dataframe containing the data relative to all cities, models and
            scenarios that are available, generated by `generate_precipitation_dataset()`
        """
        return pd.concat(
            self.generate_precipitation_indices(),
            ignore_index=True,
            copy=False)
