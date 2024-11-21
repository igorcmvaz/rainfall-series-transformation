import json
import logging
import time
from argparse import ArgumentParser
from collections.abc import Iterator, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from netCDF4 import Dataset  # type: ignore

CLIMATE_MODELS: list[str] = [
    "ACCESS-CM2",
    "ACCESS-ESM1-5",
    "CMCC-ESM2",
    "EC-EARTH3",
    "GFDL-CM4",
    "GFDL-ESM4",
    "HadGEM3-GC31-LL",
    "INM-CM4_8",
    "INM-CM5",
    "IPSL-CM6A-LR",
    "KACE",
    "KIOST",
    "MIROC6",
    "MPI-ESM1-2",
    "MRI-ESM2",
    "NESM3",
    "NorESM2-MM",
    "TaiESM1",
    "UKESM1-0-LL",
    ]

SCENARIOS: dict[str, list[dict[str, str | tuple[str, str]]]] = {
    "Histórico": [
        {
            "label": "Histórico",
            "period": ("1980-01-01", "2000-01-01")
        }
    ],
    "SSP245": [
        {
            "label": "SSP245_2015_2035",
            "period": ("2015-01-01", "2035-01-01")
        },
        {
            "label": "SSP245_2024_2074",
            "period": ("2024-01-01", "2074-01-01")
        },
        {
            "label": "SSP245_2040_2060",
            "period": ("2040-01-01", "2060-01-01")
        },
        {
            "label": "SSP245_2060_2080",
            "period": ("2060-01-01", "2080-01-01")
        },
    ],
    "SSP585": [
        {
            "label": "SSP585_2015_2035",
            "period": ("2015-01-01", "2035-01-01")
        },
        {
            "label": "SSP585_2024_2074",
            "period": ("2024-01-01", "2074-01-01")
        },
        {
            "label": "SSP585_2040_2060",
            "period": ("2040-01-01", "2060-01-01")
        },
        {
            "label": "SSP585_2060_2080",
            "period": ("2060-01-01", "2080-01-01")
        },
    ]
    }

INPUT_FILENAME_FORMAT: dict[str, str] = {
        "Histórico": "{model}-pr-hist.nc",
        "SSP245": "{model}-pr-ssp245.nc",
        "SSP585": "{model}-pr-ssp585.nc"
    }


def validate_data_point(
        data_series: Dataset,
        time_index: int,
        latitude_index: int,
        longitude_index: int) -> float | None:
    """
    Retrieves the value of a data point given indices for time, latitude and longitude, if
    it is a number.

    Args:
        data_series (Dataset): Data series containing time, latitude and longitude marks.
        time_index (int): Index for the time mark.
        latitude_index (int): Index for the latitude mark.
        longitude_index (int): Index for the longitude mark.

    Raises:
        IndexError: if there is no point in the data series that simultaneously corresponds
            to the time, latitude and longitude marks.

    Returns:
        float | None: Value of the data point converted to float or None, if unavailable.
    """
    try:
        value = data_series[time_index, latitude_index, longitude_index]
    except IndexError:
        logging.exception(
            f"Index error at {time_index=}, {latitude_index=}, {longitude_index=}")
        return None
    if np.isnan(value):
        return None
    return float(value)


def extract_precipitation(
        source_path: Path,
        target_latitude: float,
        target_longitude: float) -> Sequence[tuple[datetime, float]] | None:
    """
    Loads a NetCDF4 file from specified path and extracts the precipitation data for the
    desired coordinates.

    Args:
        source_path (Path): Path to the NetCDF4 file containing the data.
        target_latitude (float): Latitude corresponding to the data points to be extracted.
        target_longitude (float): Longitude corresponding to the data points to be
            extracted.

    Returns:
        Sequence[tuple[datetime, float]] | None: Precipitation data series, with each value
            mapped to a datetime.
    """
    start_time: float = time.perf_counter()
    try:
        dataset = Dataset(source_path)
    except Exception as e:
        logging.exception(f"Error while processing file, details below:\n{e}")
        return None
    logging.info(
        f"Successfully loaded NetCDF4 dataset from '{source_path.resolve()}' in "
        f"{round(1000*(time.perf_counter() - start_time))}ms")

    latitudes: Sequence[float] = dataset.variables["lat"][:]
    longitudes: Sequence[float] = dataset.variables["lon"][:]
    timestamps: Sequence[float] = dataset.variables["time"][:]
    precipitation: Dataset = dataset.variables["pr"][:]

    try:
        latitude_index: int = np.argwhere(latitudes == target_latitude)[0][0]
        longitude_index: int = np.argwhere(longitudes == target_longitude)[0][0]
    except IndexError:
        logging.error(
            f"Could not find target coordinates ({target_latitude}, {target_longitude}) "
            f"in file '{source_path}'")
        dataset.close()
        return None
    logging.debug(
        f"Found latitude index={int(latitude_index)} and "
        f"longitude index={int(longitude_index)}")

    reference_date = datetime.strptime(
        dataset.variables["time"].units.split("since")[1].strip(), "%Y-%m-%dT%H:%M:%S")
    dates: list[datetime] = [reference_date + timedelta(days=float(t)) for t in timestamps]
    logging.debug(f"Found {len(dates)} time indices for precipitation data")

    start_time = time.perf_counter()
    precipitation_series: Sequence[tuple[datetime, float]] = []
    for time_index in range(len(dates)):
        mean_precipitation = validate_data_point(
            precipitation, time_index, latitude_index, longitude_index)
        if mean_precipitation is not None:
            precipitation_series.append((dates[time_index], mean_precipitation))
    logging.info(
        f"Successfully validated {len(precipitation_series)} data points in "
        f"{round(1000*(time.perf_counter() - start_time))}ms")

    dataset.close()
    return precipitation_series


def filter_by_date(
        data_series: Sequence,
        start_date: datetime,
        end_date: datetime) -> Iterator[tuple[datetime, Any]]:
    """
    Filter a data series using a reference time period, returning an iterator.

    Args:
        data_series (Sequence): Data series.
        start_date (datetime): Start of the time period.
        end_date (datetime): End of the time period.

    Returns:
        Iterator[tuple[datetime, Any]]: Iterator which yields the value with its datetime if
            within the specified period.
    """
    for date, value in data_series:
        if start_date <= date <= end_date:
            yield (date, value)


def generate_csv_files(
        model: str,
        city_name: str,
        latitude: float,
        longitude: float,
        input_path: Path,
        output_path: Path) -> None:
    """
    Generates CSV files with precipitation data from specified climate models for given
    periods of time based on a set of NetCDF4 files and a specified location.

    Args:
        model (str): Climate model to be used for data extraction.
        city_name (str): Name of the city to which the data is related.
        latitude (float): Latitude of the location in the NetCDF4 file.
        longitude (float): Longitude of the location in the NetCDF4 file.
        input_path (Path): Path to the directory containing the NetCDF4 files.
        output_path (Path): Path to the directory where the CSV files should be saved.
    """
    start_time: float = time.perf_counter()
    file_counter: int = 0
    logging.info(
        f"Starting extraction of data from model '{model}', for the city of '{city_name}' "
        f"with coordinates ({latitude}, {longitude})")
    for scenario_name, time_periods in SCENARIOS.items():
        logging.debug(
            f"Found {len(time_periods)} time period(s) for scenario '{scenario_name}'")
        source_file = INPUT_FILENAME_FORMAT[scenario_name].format(model=model)
        logging.debug(
            f"Source file name for model '{model}' and scenario '{scenario_name}': "
            f"{source_file}")

        source_path = Path(input_path, source_file)
        if not source_path.is_file():
            logging.error(f"Could not find source file at '{source_path.resolve()}'")
            continue

        data_series = extract_precipitation(source_path, latitude, longitude)
        if not data_series:
            logging.error(
                f"No data could be extracted from source file at '{source_path.resolve()}'")
            continue

        for details in time_periods:
            filtered_series = filter_by_date(
                data_series,
                datetime.strptime(details["period"][0], "%Y-%m-%d"),
                datetime.strptime(details["period"][1], "%Y-%m-%d"))

            df = pd.DataFrame(filtered_series, columns=["date", "precipitation"])
            complete_file_path = Path(
                output_path, f"{city_name}_{model}_{details['label']}").with_suffix(".csv")
            df.to_csv(complete_file_path, index=False)
            file_counter += 1
            logging.info(f"Successfully saved file at '{complete_file_path.resolve()}'")

        logging.info(
            f"Successfully generated {file_counter} file(s) for model '{model}', "
            f"city '{city_name}' in {time.perf_counter() - start_time:.3f}s")


def main(args):
    setup_start: float = time.perf_counter()
    logging.basicConfig(
        format="%(asctime)s    %(levelname)-8.8s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG,
    )
    if args.quiet == 1:
        logging.getLogger().setLevel(logging.INFO)
    elif args.quiet == 2:
        logging.getLogger().setLevel(logging.WARNING)
    elif args.quiet >= 3:
        logging.getLogger().setLevel(logging.ERROR)

    input_path: Path = Path(args.input)
    if not input_path.is_dir():
        logging.critical(f"Input path '{input_path.resolve()}' is not a directory")
        return None
    logging.info(f"Input path set to '{input_path.resolve()}'")

    coordinates_path: Path = Path(args.coordinates)
    if not coordinates_path.is_file():
        logging.critical(
            f"File with cities coordinates not found at '{coordinates_path.resolve()}'")
        return None

    output_path: Path = Path(args.output)
    output_path.mkdir(exist_ok=True)
    logging.info(f"Output path set to '{output_path.resolve()}'")

    with open(coordinates_path) as file:
        city_coordinates: dict[str, dict[str, Sequence[float]]] = json.load(file)

    logging.info(f"Setup time: {round(1000*(time.perf_counter() - setup_start))}ms")
    operation_start = time.perf_counter()
    for city, details in city_coordinates.items():
        latitude = details["Nearest Coordinates"][0]
        longitude = details["Nearest Coordinates"][1]
        for model in CLIMATE_MODELS:
            generate_csv_files(model, city, latitude, longitude, input_path, output_path)

    logging.info(
        f"Completed all operations in {time.perf_counter() - operation_start:.3f}s")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "input", type=str, metavar="path/to/input",
        help="path to a directory where the input NetCDF4 files are stored")
    parser.add_argument(
        "-o", "--output", type=str, metavar="path/to/output", default="output",
        help="path to a directory where the output files will be saved. "
        "Defaults to 'output'")
    parser.add_argument(
        "-q", "--quiet", action="count", default=0,
        help="turn on quiet mode (cumulative), which hides log entries of levels lower "
        "than INFO/WARNING")
    parser.add_argument(
        "-c", "--coordinates", type=str, metavar="path/to/coordinates.json",
        help="path to a JSON file containing coordinates of the desired cities")
    main(parser.parse_args())