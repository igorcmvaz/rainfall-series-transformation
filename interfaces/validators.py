from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Sequence

import numpy as np
from numpy.ma import MaskedArray


class AbstractValidator(ABC):

    @staticmethod
    @abstractmethod
    def normalize_data_series(
            data_series: MaskedArray,
            latitude_index: int,
            longitude_index: int) -> np.ndarray:
        """
        Provides a normalized version of the data series, containing only numeric values.

        Args:
            data_series (MaskedArray): Multidimensional masked array representing
                geo-referenced data (including time and coordinates).
            latitude_index (int): Index of the target latitude in the data series.
            longitude_index (int): Index of the target longitude in the data series.

        Returns:
            np.ndarray: Normalized data series containing.
        """

    @staticmethod
    @abstractmethod
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
