import json
import logging
import time
from argparse import ArgumentParser, Namespace
from collections.abc import Sequence
from pathlib import Path
from typing import Iterable

import pandas as pd
import numpy as np

import app_logging
from constants import CLIMATE_MODELS, INPUT_FILENAME_FORMAT, SSP_SCENARIOS, PARQUET_CONF
from nc_to_csv_timeseries import extract_precipitation

logger = logging.getLogger("rainfall_transformation")


def compute_seasonality_index(df: pd.DataFrame) -> float:
    """
    Computes the seasonality index for the data frame of a given year of precipitation data.

    The Seasonality Index is a non-dimensional metric that quantifies the seasonal variation
    in rainfall patterns. It is calculated as the sum of the absolute differences between
    the monthly average precipitation and the monthly average precipitation if the yearly
    precipitation were distributed evenly throughout the year, divided by the yearly
    precipitation. Reference: https://www.mdpi.com/2073-4441/15/6/1112

    Args:
        df (pd.DataFrame): Data frame containing columns "month" and "precipitation", with
        data corresponding to a **single** year.

    Returns:
        float: Seasonality index for the given data from a particular year.
    """
    yearly_precipitation: float = df["precipitation"].sum()
    if yearly_precipitation <= 0:
        return np.nan
    monthly_averages: pd.Series[float] = df.groupby("month")["precipitation"].mean()
    return (1 / yearly_precipitation) * (
        monthly_averages - (yearly_precipitation / 12)).abs().sum()


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


def compute_indices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes various climate indices related to precipitation data from a given data frame.

    This function takes a data frame containing precipitation and time data and computes
    several climate indices commonly used to analyze precipitation patterns. The computed
    indices* are:

        - RX1day: Monthly maximum 1-day precipitation.
        - RX5day: Monthly maximum consecutive 5-day precipitation.
        - SDII: Simple pricipitation intensity index (mean precipitation on wet days).
        - R20mm: Annual count of days when precipitation ≥ 20mm.
        - CDD: Maximum length of dry spell, maximum number of consecutive days with
            precipitation < 1mm.
        - CWD: Maximum length of wet spell, maximum number of consecutive days with
            precipitation ≥ 1mm.
        - R95p: Annual total precipitation from days exceeding the 95th percentile for the
            entire period.
        - PRCPTOT: Annual total precipitation in wet days.
        - Seasonality Index: Seasonality index quantifying seasonal variation within a year.

    *Most indices are computed according to
    (https://etccdi.pacificclimate.org/list_27_indices.shtml), and seasonality index is
    computed as derived by Walsh and Lawler (1981).

    Args:
        df (pd.DataFrame): Data frame containing columns "date" (datetime) and
        "precipitation" (numeric).

    Returns:
        pd.DataFrame: Data frame containing the computed climate indices, with each
        index as a column and the corresponding values for each year as rows.

    Note:
        Additional columns are created in the input data frame for intermediate
        calculations, such as "year", "month", "dry_days", "wet_days", and "rolling_5day".
    """
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["dry_days"] = df["precipitation"] < 1
    df["wet_days"] = df["precipitation"] >= 1
    df["rolling_5day"] = df["precipitation"].rolling(window=5, min_periods=1).sum()

    rx1day: pd.Series = df.groupby("year")["precipitation"].max()
    rx5day: pd.Series = df.groupby("year")["rolling_5day"].max()
    sdii: pd.Series = df[df["wet_days"]].groupby("year")["precipitation"].mean()
    r20mm: pd.Series = df[df["precipitation"] >= 20].groupby("year").size()
    cdd: pd.Series = df.groupby("year")["dry_days"].apply(find_max_consecutive_run_length)
    cwd: pd.Series = df.groupby("year")["wet_days"].apply(find_max_consecutive_run_length)
    r95p: pd.Series = df[
        df["precipitation"] > df["precipitation"].quantile(0.95)
        ].groupby("year")["precipitation"].sum()
    prcptot: pd.Series = df[df["wet_days"]].groupby("year")["precipitation"].sum()
    seasonality_indices: pd.Series = df.groupby("year")[["month", "precipitation"]].apply(
        compute_seasonality_index)

    precipitation_indices = pd.DataFrame({
        "PRCPTOT": prcptot,
        "R95p": r95p,
        "RX1day": rx1day,
        "RX5day": rx5day,
        "SDII": sdii,
        "R20mm": r20mm,
        "CDD": cdd,
        "CWD": cwd,
        "Seasonality_Index": seasonality_indices
        })
    precipitation_indices.reset_index(inplace=True)
    return precipitation_indices


def estimate_combinations(
        models: list[str],
        scenarios: dict[str, list[dict[str, str]]],
        city_coordinates: dict[str, dict[str, Sequence[float]]]) -> int:
    """
    Estimates the number of combinations of cities, climate models and scenarios.

    Args:
        models (list[str]): List of climate model names.
        scenarios (dict[str, list[dict[str, str]]]): Dictionary mapping scenario names to
            their details per period.
        city_coordinates (dict[str, dict[str, Sequence[float]]]): Dictionary mapping city
            names to their coordinates.

    Returns:
        int: Estimated number of combinations of cities, climate models and scenarios.
    """
    return len(models)*len(scenarios.keys())*len(city_coordinates.keys())


def consolidate_precipitation_data(
        input_path: Path,
        city_coordinates: dict[str, dict[str, Sequence[float]]]
        ) -> Iterable[pd.DataFrame]:
    """
    Generates data frames containing precipitation data and climate indices for multiple
    cities, climate models and scenarios.

    This function takes an input path of directory containing NetCDF4 files and a dictionary
    of city coordinates, then retrieves precipitation data from existing files for each
    city, climate model, and scenario combination. It then computes various climate indices
    related to the precipitation data and yields a data frame.

    Args:
        input_path (Path): Path to the directory containing NetCDF4 files.
        city_coordinates (dict[str, dict[str, Sequence[float]]]): Dictionary where keys are
            city names and values are dictionaries containing the nearest valid latitude
            and longitude coordinates.

    Yields:
        Iterable[pd.DataFrame]: Data frame containing the computed climate indices for a
        specific combination of city, climate model, and scenario combination.

    Notes:
        - In case of invalid coordinates or no valid precipitation data, the function skips
        that combination and continues with the next one.
    """
    counts = {
        "total": estimate_combinations(CLIMATE_MODELS, SSP_SCENARIOS, city_coordinates),
        "processed": 0,
        "error": 0
    }
    logger.info(f"Starting consolidation process for {counts['total']} items")
    for model in CLIMATE_MODELS:
        for scenario_name in SSP_SCENARIOS.keys():
            source_path = Path(
                input_path, INPUT_FILENAME_FORMAT[scenario_name].format(model=model))
            if not source_path.is_file():
                logger.warning(
                    f"[{counts['processed']}/{counts['total']}] Could not find source file "
                    f"'{source_path.name}' for model '{model}' and scenario "
                    f"'{scenario_name}', skipping")
                continue

            for city_name, details in city_coordinates.items():
                latitude: float | None = details.get("nearest", {}).get("lat")
                longitude: float | None = details.get("nearest", {}).get("lon")
                if not all((latitude, longitude)):
                    logger.warning(
                        f"[{counts['processed']}/{counts['total']}] Coordinates not "
                        f"available for the city of '{city_name}', model '{model}', climate"
                        f" scenario '{scenario_name}', skipping")
                    continue

                start_time = time.perf_counter()
                data_series = extract_precipitation(source_path, latitude, longitude)
                if not data_series:
                    counts["processed"] += 1
                    counts["error"] += 1
                    logger.error(
                        f"[{counts['processed']}/{counts['total']}] No valid precipitation"
                        f" data found for the city of '{city_name}', model '{model}', "
                        f"climate scenario '{scenario_name}' in file at "
                        f"'{source_path.name}'")
                    continue

                indices = compute_indices(
                    pd.DataFrame(data_series, columns=["date", "precipitation"]))

                metadata = {
                    "City": city_name,
                    "Model": model,
                    "Scenario": scenario_name,
                    "Latitude": latitude,
                    "Longitude": longitude
                }
                for key, value in metadata.items():
                    indices[key] = value
                counts["processed"] += 1
                logger.info(
                    f"[{counts['processed']}/{counts['total']}] Compiled precipitation "
                    f"indices for the city of '{city_name}', model '{model}', climate "
                    f"scenario '{scenario_name}' in "
                    f"{time.perf_counter() - start_time:.3f}s")
                yield indices
    counts["skipped"] = counts["total"] - counts["processed"]
    counts["success"] = counts["processed"] - counts["error"]
    logger.info(
        f"Processed {counts['processed']} combinations, from a total of {counts['total']},"
        f" skipped {counts['skipped']} items, encountered {counts['error']} error(s). "
        f"Success rate: "
        f"{100*(counts['success'])/(counts['total'] - counts['skipped']):.2f}%. Effective "
        f"processed rate: {100*counts['processed']/counts['total']:.2f}%")


def main(args: Namespace) -> None:
    app_logging.setup(args.quiet)

    input_path: Path = Path(args.input)
    if not input_path.is_dir():
        logger.critical(f"Input path '{input_path.resolve()}' is not a directory")
        return
    logger.info(f"Input path set to '{input_path.resolve()}'")

    coordinates_path: Path = Path(args.coordinates)
    if not coordinates_path.is_file():
        logger.critical(
            f"File with cities coordinates not found at '{coordinates_path.resolve()}'")
        return
    logger.debug(f"Using coordinates from file at '{coordinates_path.resolve()}'")

    with open(coordinates_path) as file:
        city_coordinates: dict[str, dict[str, Sequence[float]]] = json.load(file)

    output_path: Path = Path(args.output)
    logger.debug("Setup completed, starting data frame generation")

    # TODO: parallelize data consolidation to see if it improves times
    start_time = time.perf_counter()
    consolidated_dataframe = pd.concat(
        consolidate_precipitation_data(input_path, city_coordinates),
        ignore_index=True,
        copy=False)
    logger.info(
        f"Consolidated all precipitation data frame(s) in a total of "
        f"{time.perf_counter() - start_time:.3f}s")

    start_time = time.perf_counter()
    consolidated_dataframe.to_parquet(output_path, index=False, **PARQUET_CONF)
    logger.info(
        f"Generated .parquet file at '{output_path.resolve()}' in "
        f"{round(1000*(time.perf_counter() - start_time))}ms")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "coordinates", type=str, metavar="path/to/coordinates.json",
        help="path to a JSON file containing coordinates of the desired cities")
    parser.add_argument(
        "input", type=str, metavar="path/to/input",
        help="path to a directory where the input NetCDF4 files are stored")
    parser.add_argument(
        "-o", "--output", type=str, metavar="path/to/output.parquet",
        default="consolidated.parquet", help="path to output Parquet file. Defaults to "
        "'./consolidated.parquet'")
    parser.add_argument(
        "-q", "--quiet", action="count", default=0,
        help="turn on quiet mode (cumulative), which hides log entries of levels lower "
        "than INFO/WARNING")
    main(parser.parse_args())
