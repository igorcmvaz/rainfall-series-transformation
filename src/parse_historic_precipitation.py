import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.core.groupby import DataFrameGroupBy

import app_logging
from nc_to_parquet import compute_seasonality_index

logger = logging.getLogger("rainfall_parsing")

FILE_PREFIX = "dados"
PRECIPITATION_METADATA_LENGTH = 9
METADATA_INDICES = {
    "name": 0,
    "station_code": 1,
    "latitude": 2,
    "longitude": 3,
    "altitude": 4,
    "situation": 5,
    "start_date": 6,
    "end_date": 7,
    "measurement_frequency": 8,
}


def has_year_round_data(group: DataFrameGroupBy) -> bool:
    """
    Checks if a group of monthly precipitation data contains values for all 12 months.

    This function takes a group of monthly precipitation data (typically a year's worth of
    data) and checks if there are valid (non-null) precipitation values for all 12 months.
    It returns True if all 12 months have valid data, and False otherwise.

    Args:
        group (DataFrameGroupBy): Group of monthly precipitation data, typically a year's
        worth of data.

    Returns:
        bool: True if the group contains valid precipitation data for all 12 months,
        False otherwise.
    """
    valid_months = group["month"][group["precipitation"].notnull()]
    return len(valid_months) == 12


def parse_precipitation(file_path: Path) -> pd.DataFrame:
    """
    Parses precipitation data from a CSV file and computes the corresponding seasonality
    index.

    This function reads precipitation data from a CSV file, considering the first few rows
    contain metadata, and the remaining rows contain monthly precipitation values. It
    processes the data, filters out years with incomplete data, computes the seasonality
    index, and returns a data frame with the station name, coordinates, and seasonality
    index.

    Args:
        file_path (Path): Path to the CSV file containing precipitation data.

    Returns:
        pd.DataFrame: DataFrame containing station name, latitude, longitude, and
        seasonality index.
    """
    metadata: pd.DataFrame = pd.read_csv(
        file_path,
        sep=":",
        nrows=PRECIPITATION_METADATA_LENGTH,
        names=["field", "value"],
        usecols=["value"])

    df: pd.DataFrame = pd.read_csv(
        file_path,
        sep=";",
        skiprows=PRECIPITATION_METADATA_LENGTH,
        header=0,
        names=["date", "precipitation", "drop"],
        usecols=["date", "precipitation"],
        parse_dates=[0],
        date_format="%Y-%m-%d")

    df["year"] = pd.DatetimeIndex(df["date"]).year
    df["month"] = pd.DatetimeIndex(df["date"]).month
    df.drop(columns="date", inplace=True)
    df = df.groupby("year").filter(has_year_round_data)[["year", "month", "precipitation"]]

    seasonality_indices: pd.Series = df.groupby("year")[["month", "precipitation"]].apply(
        compute_seasonality_index)
    station_seasonality_index: float = seasonality_indices.mean()
    station_name: str = metadata.iloc[METADATA_INDICES["name"]].value.lstrip()
    if len(df.index) == 0:
        logger.warning(
            f"No precipitation data after filtering invalid years for station "
            f"'{station_name}'")
        station_seasonality_index = np.nan

    result = {
        "name": [station_name],
        "latitude": [float(metadata.iloc[METADATA_INDICES["latitude"]].value)],
        "longitude": [float(metadata.iloc[METADATA_INDICES["longitude"]].value)],
        "seasonality_index": [station_seasonality_index]
    }
    return pd.DataFrame.from_dict(result)


def process_single_file(input_path: Path) -> None:
    """
    Processes a single precipitation data file and prints its contents.

    This function takes a file path as input, parses the precipitation data in the file,
    and prints the contents of the resulting data frame in a formatted manner.

    Args:
        input_path (Path): Path to the precipitation data file to be processed.
    """
    for key, values in parse_precipitation(input_path).items():
        print(f"{key}: {values[0]}")


def process_directory(input_path: Path) -> pd.DataFrame:
    """
    Processes precipitation CSV files in a directory and concatenates the results.

    This function iterates over all files in the specified input directory that have a
    '.csv' extension and start with a specific file prefix. It parses each valid file and
    concatenates the results into a single DataFrame.

    Args:
        input_path (Path): Path to the directory containing the precipitation CSV files.

    Returns:
        pd.DataFrame: Data frame containing the concatenated precipitation data from all
        valid files in the input directory.
    """
    processed_data: list[pd.DataFrame] = []
    for file_path in input_path.iterdir():
        if (file_path.suffix.casefold() == ".csv"
                and file_path.name.startswith(FILE_PREFIX)):
            logger.debug(f"Found file valid file '{file_path.name}' in input path")
            processed_data.append(parse_precipitation(file_path))
    return pd.concat(processed_data, ignore_index=True, copy=False)


def main(args: Namespace) -> None:
    app_logging.setup(args.quiet)

    input_path: Path = Path(args.input)
    if input_path.is_file():
        logger.debug(f"Input path '{input_path.resolve()}' is a precipitation file")
        process_single_file(input_path)
        return
    elif input_path.is_dir():
        logger.debug(f"Input path '{input_path.resolve()}' is a directory")
        output_path: Path = Path(args.output)
        logger.debug(f"Output path set to '{output_path.resolve()}'")

        consolidated_dataframe: pd.DataFrame = process_directory(input_path)
        consolidated_dataframe.to_csv(output_path, index=False, encoding="utf-8")
        logger.info(f"Saved output file at '{output_path.resolve()}'")
        return
    else:
        logger.exception(
            f"Input path '{input_path.resolve()}' is neither a file or directory")
        return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "input", type=str, metavar="path/to/input.csv",
        help="path to CSV or directory with CSV files containing historic precipitation "
        "data")
    parser.add_argument(
        "-o", "--output", type=str, metavar="path/to/output.csv",
        default="consolidated.csv", help="path to output CSV file (when input is a "
        "directory). Defaults to './consolidated.csv'")
    parser.add_argument(
        "-q", "--quiet", action="count", default=0,
        help="turn on quiet mode (cumulative), which hides log entries of levels lower "
        "than INFO/WARNING")
    main(parser.parse_args())
