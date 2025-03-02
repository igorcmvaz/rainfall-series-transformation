import csv
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from netCDF4 import Dataset

from agents.validators import PrecipitationValidator
from globals.errors import InvalidSourceFileError, InvalidTargetCoordinatesError
from globals.types import ParsedVariable

logger = logging.getLogger("rainfall_transformation")


class NetCDFExtractor:
    """Provides functions to extract precipitation data from a NetCDF4 file."""

    variables: dict[str, ParsedVariable]

    def __init__(self, source_path: Path) -> None:
        self.variables = {}
        self._get_dataset_variables(source_path)
        logger.debug(
            f"Loaded variables ({', '.join(self.variables.keys())}) from file at "
            f"'{source_path.resolve()}'")

    def _get_dataset_variables(self, source_path: Path) -> None:
        """
        Retrieves the dataset variables and their units.

        Args:
            source_path (Path): Path to the NetCDF4 file.
        """
        with Dataset(source_path) as dataset:
            for name, variable in dataset.variables.items():
                self.variables[name] = ParsedVariable(variable.units, variable[:])

    def _parse_reference_date(
            self, datetime_format: str = "%Y-%m-%dT%H:%M:%S") -> datetime:
        """
        Parses the reference date from the time variable's units.

        Args:
            datetime_format (str, optional): Format string used to parse the reference date
                from the dataset's time variable unit. Defaults to "%Y-%m-%dT%H:%M:%S".

        Returns:
            datetime: Reference date parsed from the units information in the dataset's
            time variable.

        Notes:
            The function assumes that the time variable's unit follows the format:
            "days since {datetime_format}"
        """
        return datetime.strptime(
            self.variables["time"].units.split("since")[-1].strip(), datetime_format)

    def _find_coordinates_indices(
            self,
            target_latitude: float,
            target_longitude: float) -> tuple[int, int]:
        """
        Finds the indices of target coordinates among coordinate variables.

        Args:
            target_latitude (int): Target latitude coordinate.
            target_longitude (int): Target longitude coordinate.

        Returns:
            tuple[int, int]: Tuple containing the indices of the latitude and longitude
            coordinates found for each corresponding variable.

        Raises:
            InvalidTargetCoordinatesError: If either target latitude or longitude are not
            found.
        """
        try:
            coordinates_indices = (
                np.argwhere(self.variables["lat"].values == target_latitude)[0][0],
                np.argwhere(self.variables["lon"].values == target_longitude)[0][0]
            )
        except IndexError:
            logger.exception(
                f"Target coordinates ({target_latitude}, {target_longitude}) are not "
                f"present in the dataset")
            raise InvalidTargetCoordinatesError(target_latitude, target_longitude)
        return coordinates_indices

    def _relative_to_absolute_date(
            self, data_series: np.ndarray[Any]) -> np.ndarray[tuple[datetime, Any]]:
        """
        Transforms time variable of the dataset from relative dates (defined by a reference
        date) into absolute datetime values and merges it with the data from a given series.

        Args:
            data_series (np.ndarray[Any]): Data series to be merged with absolute datetime
            values.

        Returns:
            np.ndarray[tuple[datetime, Any]]: Array of tuples, each containing an absolute
            datetime and the corresponding values from the data series, by index.
        """
        reference_date = self._parse_reference_date()
        logger.debug(f"Reference date: {reference_date.strftime("%Y-%m-%d")}")
        dates = [
            reference_date + timedelta(days=float(t))
            for t in self.variables["time"].values
        ]
        return np.array([(dates[index], value) for index, value in enumerate(data_series)])

    def extract_precipitation(
            self,
            target_latitude: float,
            target_longitude: float) -> np.ndarray[tuple[datetime, float]]:
        """
        Extracts normalized precipitation data along with absolute datetimes from the
        dataset, for a given pair of coordinates.

        Args:
            target_latitude (float): Target latitude coordinate.
            target_longitude (float): Target longitude coordinate.

        Returns:
            np.ndarray[tuple[datetime, float]]: Precipitation data series, with each value
            mapped to a datetime, for a given pair of coordinates.
        """
        latitude_index, longitude_index = self._find_coordinates_indices(
            target_latitude, target_longitude)

        return self._relative_to_absolute_date(
            PrecipitationValidator.normalize_data_series(
                self.variables["pr"].values, latitude_index, longitude_index))


class BaseCoordinatesExtractor:

    source_path: Path

    def __init__(self, source_path: Path) -> None:
        if not source_path.is_file():
            raise InvalidSourceFileError(source_path.resolve())
        self.source_path = source_path

    def get_coordinates(self) -> dict[str, Any]:
        raise NotImplementedError


class StructuredCoordinatesExtractor(BaseCoordinatesExtractor):

    def get_coordinates(self) -> dict[str, dict[str, list[float]]]:
        """
        Retrieves the latitude and longitude coordinates from a structured JSON file and
        returns them.

        Returns:
            dict[str, dict[str, list[float]]]: Dictionary mapping city names to target and
            nearest latitude and longitude coordinates, as well as any other defining
            key-value pairs for each city.
        """
        with open(self.source_path, encoding="utf-8") as file:
            city_coordinates = json.load(file)
        logger.info(
            f"Successfully extracted coordinates from file at "
            f"'{self.source_path.resolve()}'")
        return city_coordinates


class RawCoordinatesExtractor(BaseCoordinatesExtractor):

    def get_coordinates(self) -> dict[str, dict[str, int | float]]:
        """
        Retrieves the IBGE code and latitude and longitude coordinates from a 'raw' CSV file
        and returns them in a structure format.

        Returns:
            dict[str, dict[str, int | float]]: Dictionary mapping city names to their IBGE
            code, as well as their latitude and longitude coordinates.
        """
        raw_coordinates: dict[str, dict[str, int | float]] = {}
        with open(self.source_path, encoding="utf-8", newline="") as file:
            reader = csv.reader(file, delimiter=",")
            next(reader)
            for row in reader:
                raw_coordinates[row[1]] = {
                    "ibge_code": int(row[0]),
                    "latitude": float(row[2]),
                    "longitude": float(row[3])
                }
        logger.info(
            f"Successfully extracted coordinates and codes from file at "
            f"'{self.source_path.resolve()}'")
        return raw_coordinates
