from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.ma import MaskedArray

from globals.constants import INPUT_FILENAME_FORMAT
from globals.errors import (
    CoordinatesNotAvailableError, InvalidClimateScenarioError, InvalidSourceDirectoryError,
    InvalidSourceFileError)


class PrecipitationValidator:

    @staticmethod
    def normalize_data_series(
            data_series: MaskedArray,
            latitude_index: int,
            longitude_index: int) -> np.ndarray:
        """
        Fills all the masked values in the portion of data series limited by the coordinates
        indices with 'NaN', then converts all 'NaN' entries to 0.

        Args:
            data_series (MaskedArray): Multidimensional masked array representing
                geo-referenced data (including time and coordinates).
            latitude_index (int): Index of the target latitude in the data series.
            longitude_index (int): Index of the target longitude in the data series.

        Returns:
            np.ndarray: Data series containing exclusively numeric values.
        """
        return np.nan_to_num(data_series[
            :, latitude_index, longitude_index].filled(np.nan), nan=0, copy=False)

    @staticmethod
    def filter_by_date(
            data_series: Sequence[tuple[datetime, Any]],
            start_date: datetime,
            end_date: datetime) -> np.ndarray[datetime, Any]:
        """
        Filters a data series using a reference time period.

        Args:
            data_series (Sequence): Data series.
            start_date (datetime): Start of the time period.
            end_date (datetime): End of the time period.

        Returns:
            np.ndarray[datetime, Any]: Filtered data series, containing only the entries
            where the datetime is within the given period.
        """
        return np.array(data_series)[np.nonzero(
            [start_date <= date <= end_date for date, _ in data_series])]


class CoordinatesValidator:

    @staticmethod
    def get_coordinates(details: dict[str, dict[str, float]]) -> tuple[float, float]:
        """
        Retrieves the coordinates from a dictionary for a specific location.

        Args:
            details (dict[str, dict[str, float]]): Dictionary containing the details for the
            location, including the 'nearest' coordinates.

        Raises:
            CoordinatesNotAvailableError: If either latitude or longitude are not
            successfully retrieved.

        Returns:
            tuple[float, float]: Tuple of latitude and longitude retrieved from the
            dictionary.
        """
        latitude: float | None = details.get("nearest", {}).get("lat")
        longitude: float | None = details.get("nearest", {}).get("lon")
        coordinates = (latitude, longitude)
        if not all(coordinates):
            raise CoordinatesNotAvailableError
        return coordinates


class PathValidator:

    @staticmethod
    def validate_precipitation_source_path(
            model: str, scenario: str, source_dir: Path | str) -> Path:
        """
        Validates a source file path for precipitation data from a given directory, given a
        specific model and scenario.

        Args:
            model (str): Name of the climate model.
            scenario (str): Name of the climate scenario.
            source_dir (Path | str): Directory containing the precipitation files.

        Raises:
            InvalidClimateScenarioError: If no file format matches the provided climate
                scenario.
            InvalidSourceFileError: If a file does not exist in the expected path.

        Returns:
            Path: Path to the file containing the corresponding precipitation data.
        """
        try:
            source_path = Path(
                source_dir, INPUT_FILENAME_FORMAT[scenario].format(model=model))
        except KeyError:
            raise InvalidClimateScenarioError
        else:
            if not source_path.is_file():
                raise InvalidSourceFileError
            return source_path
