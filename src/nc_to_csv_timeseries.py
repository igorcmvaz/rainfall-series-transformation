import json
import logging
from argparse import ArgumentParser
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path

from netCDF4 import Dataset     # type: ignore
import numpy as np
import pandas as pd


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
    """_summary_

    Args:
        data_series (Dataset): _description_
        time_index (int): _description_
        latitude_index (int): _description_
        longitude_index (int): _description_

    Returns:
        float: _description_
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
    """_summary_

    Args:
        source_path (Path): _description_
        target_latitude (float): _description_
        target_longitude (float): _description_

    Returns:
        Sequence[tuple[datetime, float]] | None: _description_
    """
    try:
        dataset = Dataset(source_path)
    except Exception as e:
        logging.exception(f"Error while processing file, details below:\n{e}")
        return None
    logging.info(f"Successfully loaded NetCDF4 dataset from '{source_path.resolve()}'")

    latitudes: Sequence[float] = dataset.variables["lat"][:]
    longitudes: Sequence[float] = dataset.variables["lon"][:]
    timestamps: Sequence[str] = dataset.variables["time"][:]
    precipitation: Dataset = dataset.variables["pr"][:]

    if (target_latitude not in latitudes) or (target_longitude not in longitudes):
        logging.error(
            f"Could not find target coordinates ({target_latitude}, {target_longitude}) "
            f"in file '{source_path}'")
        dataset.close()
        return None

    latitude_index: int = np.where(latitudes == target_latitude)[0][0]
    longitude_index: int = np.where(longitudes == target_longitude)[0][0]
    logging.debug(
        f"Found latitude index={int(latitude_index)} and "
        f"longitude index={int(longitude_index)}")

    reference_date = datetime.strptime(
        dataset.variables["time"].units.split("since")[1].strip(), "%Y-%m-%dT%H:%M:%S")
    dates: list[datetime] = [reference_date + timedelta(days=float(t)) for t in timestamps]
    logging.debug(f"Found {len(dates)} time indices for precipitation data")

    precipitation_series: Sequence[tuple[datetime, float]] = []
    for time_index in range(len(dates)):
        mean_precipitation = validate_data_point(
            precipitation, time_index, latitude_index, longitude_index)
        if mean_precipitation is not None:
            precipitation_series.append((dates[time_index], mean_precipitation))

    dataset.close()
    return precipitation_series


def filter_by_date(
        data_series: Sequence,
        start_date: datetime,
        end_date: datetime) -> Sequence[tuple[datetime, float]]:
    """_summary_

    Args:
        data_series (Sequence): _description_
        start_date (datetime): _description_
        end_date (datetime): _description_

    Returns:
        Sequence[tuple[datetime, float]]: _description_
    """
    return [(date, value) for date, value in data_series if start_date <= date <= end_date]


def generate_csv_files(
        model: str,
        city_name: str,
        latitude: float,
        longitude: float,
        input_path: Path,
        output_path: Path) -> None:
    """_summary_

    Args:
        model (str): _description_
        city_name (str): _description_
        latitude (float): _description_
        longitude (float): _description_
        input_path (Path): _description_
        output_path (Path): _description_
    """
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
            df.to_csv(Path(
                output_path,
                f"{city_name}_{model}_{details['label']}"
                ).with_suffix(".csv"), index=False)


def main(args):
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

    input_path = Path(args.input)
    if not input_path.is_dir():
        logging.critical(f"Input path '{input_path.resolve()}' is not a directory")
        return None
    logging.info(f"Input path set to '{input_path.resolve()}'")

    coordinates_path = Path(args.coordinates)
    if not coordinates_path.is_file():
        logging.critical(
            f"File with cities coordinates not found at '{coordinates_path.resolve()}'")
        return None

    output_path = Path(args.output)
    output_path.mkdir(exist_ok=True)
    logging.info(f"Output path set to '{output_path.resolve()}'")

    with open(coordinates_path) as file:
        city_coordinates: dict[str, Sequence[float]] = json.load(file)

    for city, (latitude, longitude) in city_coordinates.items():
        for model in CLIMATE_MODELS:
            generate_csv_files(model, city, latitude, longitude, input_path, output_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "input", type=str, metavar="path/to/input",
        help="path to a directory where the input NC files are stored")
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
