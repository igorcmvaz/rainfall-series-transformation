from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.ma import MaskedArray

from constants import INPUT_FILENAME_FORMAT
from errors import CoordinatesNotAvailableError, InvalidSourceFileError


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
