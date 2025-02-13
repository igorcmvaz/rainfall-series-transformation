import logging
from collections.abc import Sequence
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger("rainfall_transformation")


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

    def compute_climate_indices(self) -> pd.DataFrame:
        """
        Computes various climate indices related to precipitation data from the data frame.

        The computed indices* are:

            - RX1day: Monthly maximum 1-day precipitation.
            - RX5day: Monthly maximum consecutive 5-day precipitation.
            - SDII: Simple pricipitation intensity index (mean precipitation on wet days).
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
