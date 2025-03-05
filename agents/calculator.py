import logging
from collections.abc import Sequence
from datetime import datetime
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd

from agents.extractors import NetCDFExtractor, RawCoordinatesExtractor
from agents.validators import PrecipitationValidator
from globals.errors import ReachedCoordinatesOffsetLimitError

logger = logging.getLogger("rainfall_transformation")


def estimate_combinations(*args) -> int:
    """
    Estimates the number of combinations based on the length of the parameters.

    Returns:
        int: The total number of combinations possible given the length of the parameters.
    """
    return reduce(lambda x, y: x*y, [len(arg) for arg in args])


def find_max_consecutive_run_length(series: pd.Series) -> int:
    """
    Finds the maximum length of consecutive runs of Truthy values in a Pandas Series.

    This function groups the input Series based on consecutive runs of Truthy values
    (meaning they result in True if bool() is applied to them), computes the length (sum)
    of each consecutive run, and returns the maximum length among all of them.

    Args:
        series (pd.Series): Pandas Series containing numeric values.

    Returns:
        int: Maximum length of all consecutive runs in the given Series. Returns 0 for an
        empty Series.

    Notes:
        - Numeric values are not strictly required, values are compared for equality
        regardless of type.
        - Values are only considered for consecutive runs if they are 'Truthy', meaning a
        consecutive run of 0's (or 'False', 'None', '', etc) is not counted.
        - An increase in a consecutive run is only considered when the values are equal
        (according to their definition of __eq__).

    Examples:
        >>> import pandas as pd
        >>> data = [1, 1, 0, 0, 0, 1, 1, 1, 1, 0]
        >>> find_max_consecutive_run_length(pd.Series(data))
        4
        >>> data = [0, 0, 0, 0, 2, 2, 2]
        >>> find_max_consecutive_run_length(pd.Series(data))
        3
        >>> data = ["", "", "", "", "", "", None, -1, -1, -1, "foo", "foo", "foo", "foo"]
        >>> find_max_consecutive_run_length(pd.Series(data))
        4
    """
    if not any(series):
        return 0
    return series.astype(bool).groupby((series != series.shift()).cumsum()).sum().max()


def compute_seasonality_index(df: pd.DataFrame) -> float:
    """
    Computes the seasonality index for the data frame of a given year of precipitation data.

    The Seasonality Index is a non-dimensional metric that quantifies the seasonal variation
    in rainfall patterns. It is calculated as the sum of the absolute differences between
    the monthly average precipitation and the monthly average precipitation if the yearly
    precipitation were distributed evenly throughout the year, divided by the yearly
    precipitation. Reference: https://www.mdpi.com/2073-4441/15/6/1112.

    Args:
        df (pd.DataFrame): Data frame containing columns "month" and "precipitation", with
        data corresponding to a **single** year.

    Returns:
        float: Seasonality index for the given data from a particular year.
    """
    yearly_precipitation: float = df["precipitation"].sum()
    if yearly_precipitation <= 0:
        return np.nan
    monthly_precipitation: pd.Series[float] = df.groupby("month")["precipitation"].sum()
    return (1 / yearly_precipitation) * (
        monthly_precipitation - (yearly_precipitation / 12)).abs().sum()


class IndicesCalculator:

    df: pd.DataFrame
    prcptot: pd.Series
    r95p: pd.Series
    rx1day: pd.Series
    rx5day: pd.Series
    sdii: pd.Series
    r20mm: pd.Series
    cdd: pd.Series
    cwd: pd.Series
    seasonality_indices: pd.Series

    def __init__(
            self,
            data_series: Sequence[tuple[datetime, float]],
            variable_name: str = "precipitation") -> None:
        self.df = pd.DataFrame(data_series, columns=["date", variable_name])
        self._set_auxiliary_columns()

    def _set_auxiliary_columns(self) -> None:
        """
        Sets various auxiliary data columns used for the computation of climate indices from
        the base dataframe, such as dates, series of dry and wet days and series with
        rolling windows of precipitation.
        """
        self.df["year"] = self.df["date"].dt.year
        self.df["month"] = self.df["date"].dt.month
        self.df["dry_days"] = self.df["precipitation"] < 1
        self.df["wet_days"] = self.df["precipitation"] >= 1
        self.df["rolling_5day"] = self.df["precipitation"].rolling(
            window=5, min_periods=1).sum()

    # TODO: use async for computation and file management
    def compute_climate_indices(self) -> pd.DataFrame:
        """
        Computes various climate indices related to precipitation data from the data frame.

        The computed indices* are:

            - RX1day: Monthly maximum 1-day precipitation.
            - RX5day: Monthly maximum consecutive 5-day precipitation.
            - SDII: Simple precipitation intensity index (mean precipitation on wet days).
            - R20mm: Annual count of days when precipitation ≥ 20mm.
            - CDD: Maximum length of dry spell, maximum number of consecutive days with
                precipitation < 1mm.
            - CWD: Maximum length of wet spell, maximum number of consecutive days with
                precipitation ≥ 1mm.
            - R95p: Annual total precipitation from days exceeding the 95th percentile for
                the entire period.
            - PRCPTOT: Annual total precipitation in wet days.
            - Seasonality Index: Seasonality index quantifying seasonal variation within a
                year.

        *Most indices are computed according to
        (https://etccdi.pacificclimate.org/list_27_indices.shtml), and seasonality index is
        computed as derived by Walsh and Lawler (1981).

        Returns:
            pd.DataFrame: Data frame containing the computed climate indices, with each
            index as a column and the corresponding values for each year as rows.
        """
        self.rx1day = self.df.groupby("year")["precipitation"].max()
        self.rx5day = self.df.groupby("year")["rolling_5day"].max()
        self.sdii = self.df[self.df["wet_days"]].groupby("year")["precipitation"].mean()
        self.r20mm = self.df[self.df["precipitation"] >= 20].groupby("year").size()
        self.cdd = self.df.groupby("year")["dry_days"].apply(
            find_max_consecutive_run_length)
        self.cwd = self.df.groupby("year")["wet_days"].apply(
            find_max_consecutive_run_length)
        self.r95p = self.df[
            self.df["precipitation"] > self.df["precipitation"].quantile(0.95)
            ].groupby("year")["precipitation"].sum()
        self.prcptot = self.df[self.df["wet_days"]].groupby("year")["precipitation"].sum()
        self.seasonality_indices = self.df.groupby(
            "year")[["month", "precipitation"]].apply(compute_seasonality_index)

        precipitation_indices = pd.DataFrame({
            "PRCPTOT": self.prcptot,
            "R95p": self.r95p,
            "RX1day": self.rx1day,
            "RX5day": self.rx5day,
            "SDII": self.sdii,
            "R20mm": self.r20mm,
            "CDD": self.cdd,
            "CWD": self.cwd,
            "Seasonality_Index": self.seasonality_indices
            })
        precipitation_indices.reset_index(inplace=True)
        return precipitation_indices


class CoordinatesFinder:

    MAX_OFFSET = 15

    def __init__(self, netcdf_reference_path: Path, raw_coordinates_path: Path) -> None:
        netcdf_extractor = NetCDFExtractor(netcdf_reference_path)
        self.latitudes = netcdf_extractor.variables["lat"].values
        self.longitudes = netcdf_extractor.variables["lon"].values
        self.precipitation = netcdf_extractor.variables["pr"].values

        self.city_coordinates = RawCoordinatesExtractor(
            raw_coordinates_path).get_coordinates()

    def _search_around_coordinates(
            self,
            original_latitude_index: int,
            original_longitude_index: int) -> tuple[float, float]:
        """
        Searches around a set of coordinate indices for valid precipitation data, until a
        maximum offset is reached.

        Uses an expanding spiral pattern until the maximum index offset from the original
        coordinate indices is reached.

        Args:
            original_latitude_index (int): Index of the reference latitude value.
            original_longitude_index (int): Index of the reference longitude value.

        Raises:
            ReachedCoordinatesOffsetLimitError: If no valid precipitation data is found
            before reaching the maximum index offset.

        Returns:
            tuple[float, float]: Tuple of latitude and longitude coordinates closest to the
            original ones that have valid precipitation data.
        """
        MAX_LATITUDE_INDEX = len(self.precipitation[0, :, 0] - 1)
        MAX_LONGITUDE_INDEX = len(self.precipitation[0, 0, :] - 1)
        offset = 1
        checked_coordinates: set[tuple[int, int]] = {
            (original_latitude_index, original_longitude_index)}
        while offset <= self.MAX_OFFSET:
            for latitude_offset in (limits := sorted(range(-offset, offset + 1), key=abs)):
                for longitude_offset in limits:
                    new_latitude_index = min(
                        original_latitude_index + latitude_offset, MAX_LATITUDE_INDEX)
                    new_longitude_index = min(
                        original_longitude_index + longitude_offset, MAX_LONGITUDE_INDEX)
                    if (new_latitude_index, new_longitude_index) in checked_coordinates:
                        continue

                    if PrecipitationValidator.coordinates_have_precipitation_data(
                            self.precipitation, new_latitude_index, new_longitude_index):
                        return (
                            self.latitudes[new_latitude_index],
                            self.longitudes[new_longitude_index])
                    checked_coordinates.add((new_latitude_index, new_longitude_index))
            offset += 1
        raise ReachedCoordinatesOffsetLimitError(
            original_latitude_index, original_longitude_index, self.MAX_OFFSET)

    def _search_nearest_coordinates(
            self, target_latitude: float, target_longitude: float) -> tuple[float, float]:
        """
        Searches for coordinates that are closest to target coordinates and contain valid
        precipitation data.

        If the target coordinates contain valid precipitation data, they are returned right
        away. If not, a specific method (`_search_around_coordinates()`) is used to find the
        nearest coordinates that do.

        Args:
            target_latitude (float): Latitude component for which to find the nearest valid
                coordinates.
            target_longitude (float): Longitude component for which to find the nearest
                valid coordinates.

        Returns:
            tuple[float, float]: A tuple containing the nearest coordinates that contain
            valid precipitation data.
        """
        latitude_index = np.abs(self.latitudes - target_latitude).argmin()
        longitude_index = np.abs(self.longitudes - target_longitude).argmin()
        if PrecipitationValidator.coordinates_have_precipitation_data(
                self.precipitation, latitude_index, longitude_index):
            return self.latitudes[latitude_index], self.longitudes[longitude_index]
        return self._search_around_coordinates(latitude_index, longitude_index)

    def find_matching_coordinates(self) -> dict[str, dict[str, int | dict[str, float]]]:
        """
        Finds the nearest matching coordinates that contain valid precipitation data in the
        given NetCDF4 file, for each city present in the given raw coordinates file.

        All matching coordinates are consolidated in a dictionary for each city. If no
        matching coordinates are found for a city, it is not included in the resulting
        dictionary.

        Returns:
            dict[str, dict[str, int | dict[str, float]]]: Dictionary where keys are city
            names and values are dictionaries with 'target' and 'nearest' keys, each
            containing 'lat' and 'lon' keys with corresponding values. Any other key-value
            pairs present from the extracted data are also included.
        """
        nearest_matching_coordinates: dict[str, dict[str, int | dict[str, float]]] = {}
        for city_name, details in self.city_coordinates.items():
            target_latitude: float = details.pop("latitude")
            target_longitude: float = details.pop("longitude")

            try:
                valid_latitude, valid_longitude = self._search_nearest_coordinates(
                    target_latitude, target_longitude)
            except ReachedCoordinatesOffsetLimitError:
                logger.exception(
                    f"Could not find valid precipitation data near the original coordinates"
                    f" ({target_latitude}, {target_longitude})")
            else:
                nearest_matching_coordinates[city_name] = dict(
                    {
                        "target": {
                            "lat": float(target_latitude),
                            "lon": float(target_longitude)
                        },
                        "nearest": {
                            "lat": float(valid_latitude),
                            "lon": float(valid_longitude)
                        }
                    },
                    **details)
        return nearest_matching_coordinates
