import logging
from collections.abc import Generator
from pathlib import Path

import pandas as pd

from calculator import IndicesCalculator, estimate_combinations
from extractor import NetCDFExtractor
from validators import CoordinatesValidator, PathValidator

logger = logging.getLogger("rainfall_transformation")


class Consolidator:

    cities: dict[str, dict[str, dict[str, float]]]
    scenarios: dict[str, list[dict[str, str]]]
    models: list[str]
    source_dir: Path
    state: dict[str, int]

    def __init__(
            self,
            cities: dict[str, dict[str, dict[str, float]]],
            scenarios: dict[str, list[dict[str, str]]],
            models: list[str],
            source_dir: Path) -> None:
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

    def _count_error(self, **kwargs) -> None:
        self.state["processed"] += 1
        self.state["error"] += 1
        details = [f"{key}={value}" for key, value in kwargs.items()]
        logger.error(
            f"[{self.state['processed']}/{self.state['total']}] Error during processing. "
            f"Details: {', '.join(details)}")

    def _count_processed(self, **kwargs) -> None:
        self.state["processed"] += 1
        details = [f"{key}={value}" for key, value in kwargs.items()]
        logger.info(
            f"[{self.state['processed']}/{self.state['total']}] Completed processing. "
            f"Details: {', '.join(details)}")

    def _set_final_state(self) -> None:
        self.state["skipped"] = self.state["total"] - self.state["processed"]
        self.state["success"] = self.state["processed"] - self.state["error"]
        self.state["process_rate"] = 100*self.state['processed']/self.state['total']
        self.state["success_rate"] = 100*(
            self.state['success'])/(self.state['total'] - self.state['skipped'])
        logger.info(
            f"Processed {self.state['processed']} combinations, from a total of "
            f"{self.state['total']}, skipped {self.state['skipped']} items, encountered "
            f"{self.state['error']} error(s). "
            f"Success rate: {self.state['success_rate']:2f}%. "
            f"Effective processed rate: {self.state['process_rate']:.2f}%")

    def _insert_metadata(self, dataframe: pd.DataFrame, **kwargs) -> None:
        for key, value in kwargs.items():
            dataframe[key.capitalize()] = value

    def compose_precipitation_dataset(self) -> Generator[pd.DataFrame]:
        for city_name, details in self.cities.items():
            latitude, longitude = CoordinatesValidator.get_coordinates(details)
            # TODO: check for temp file
            for model in self.models:
                for scenario in self.scenarios:
                    extractor = NetCDFExtractor(PathValidator.validate_source_path(
                        model, scenario, self.source_dir))
                    data_series = extractor.extract_precipitation(latitude, longitude)
                    if not data_series.any():
                        self._count_error(
                            city=city_name,
                            model=model,
                            scenario=scenario,
                            target_coordinates=(latitude, longitude))
                        continue
                    indices = IndicesCalculator(data_series).compute_climate_indices()
                    self._insert_metadata(
                        indices,
                        city=city_name,
                        model=model,
                        scenario=scenario,
                        latitude=latitude,
                        longitude=longitude)
                    self._count_processed(
                        city=city_name,
                        model=model,
                        scenario=scenario,
                        target_coordinates=(latitude, longitude))
                    yield indices
            # TODO: create temp file
            logger.info(
                f"Completed processing of city '{city_name}', coordinates "
                f"({latitude}, {longitude})")
        self._set_final_state()

    def consolidate_dataset(self) -> pd.DataFrame:
        return pd.concat(
            self.compose_precipitation_dataset(),
            ignore_index=True,
            copy=False)
