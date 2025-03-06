import logging
from argparse import ArgumentParser
from collections.abc import Generator
from pathlib import Path

from agents.calculator import CoordinatesFinder
from agents.consolidator import Consolidator
from agents.exporters import (
    CSVExporter, JSONCoordinatesExporter, NetunoExporter, ParquetExporter)
from agents.extractors import StructuredCoordinatesExtractor
from agents.validators import CommandLineArgsValidator
from globals.constants import CLIMATE_MODELS, SSP_SCENARIOS
from globals.errors import InvalidCoordinatesFileError, InvalidSourceDirectoryError

logger = logging.getLogger("rainfall_transformation")


def setup_logger(logger: logging.Logger, quiet_count: int, verbose: bool) -> None:
    """
    Configures a logger for the application, defining output format and log level according
    to quiet or verbose arguments.

    Args:
        logger (logging.Logger): Logger channel to be configured.
        quiet_count (int): Number of times the 'quiet' flag was provided.
        verbose (bool): Whether the 'verbose' flag was provided.
    """
    if verbose:
        log_level = logging.DEBUG
    elif quiet_count == 1:
        log_level = logging.WARNING
    elif quiet_count >= 2:
        log_level = logging.ERROR
    else:
        log_level = logging.INFO

    logger.propagate = True
    logger.setLevel(log_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8.8s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z")
    handler = logging.StreamHandler()
    handler.setLevel(log_level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def filter_by_suffix(directory_path: Path, suffix: str = ".nc") -> Generator[Path]:
    """
    Generates a filtered sequence of paths in a directory based on each file's suffix.

    Args:
        directory_path (Path): Path to the directory where the files are stored.
        suffix (str): Desired suffix for the files. Any file whose last suffix is different
        will be filtered out. Defaults to ".nc".

    Yields:
        Generator[Path]: A generator of paths that have a last suffix matching the given
        suffix. Comparison is case insensitive.
    """
    yield from (
        path for path in directory_path.iterdir()
        if path.suffix.casefold() == suffix.casefold())


def find_smallest_file(directory_path: Path) -> Path:
    """
    Finds and returns the smallest NetCDF4 file (in bytes) from a given directory.

    NetCDF4 format is assumed from the file suffix (expected ".nc"), and not validated in
    any other way.

    Args:
        directory_path (Path): Path to the directory.

    Returns:
        Path: Path to the smallest NetCDF4 file (in bytes) found in the directory.
    """
    return min(filter_by_suffix(directory_path), key=lambda p: p.stat().st_size)


def process_coordinates_files(coordinates_path: Path, input_path: Path) -> None:
    """
    Uses the smallest file (assuming NetCDF4 format) from a given path to validate the
    coordinates found in another (assumed CSV) file and exports a validated JSON file.

    The smallest file is found by querying all files for their stats and comparing their
    size in bytes. It is assumed the entire directory contains only NetCDF4 files.

    Args:
        coordinates_path (Path): Path to the CSV file containing raw city coordinates and
            IBGE code.
        input_path (Path): Path to a directory containing reference NetCDF4 files.
    """
    output_path = coordinates_path.with_suffix(".json")
    validation_file_path = find_smallest_file(input_path)
    logger.info(
        f"Coordinates at '{coordinates_path.name}' will be validated against reference "
        f"file '{validation_file_path.name}'")
    validated_coordinates = CoordinatesFinder(
        validation_file_path, coordinates_path).find_matching_coordinates()
    JSONCoordinatesExporter.generate_json(validated_coordinates, output_path)


def get_csv_exporter(args: CommandLineArgsValidator) -> CSVExporter | None:
    """
    Define the CSV exporter to be used based on the command line arguments.

    Args:
        args (CommandLineArgsValidator): Instance of validated command line arguments
        container.

    Returns:
        CSVExporter | None: CSVExporter type (or one of its subclasses) if CSV files are
        required, else None.
    """
    csv_exporter = None
    if args.netuno_required:
        csv_exporter = NetunoExporter(args.input_path.parent)
    elif args.csv_required:
        csv_exporter = CSVExporter(args.input_path.parent)
    return csv_exporter


def main(args: CommandLineArgsValidator) -> None:
    if args.only_process_coordinates:
        return process_coordinates_files(args.coordinates_path, args.input_path)
    city_coordinates = StructuredCoordinatesExtractor(
        args.coordinates_path).get_coordinates()

    csv_exporter = get_csv_exporter(args)
    consolidator = Consolidator(
        city_coordinates,
        SSP_SCENARIOS,
        CLIMATE_MODELS,
        args.input_path,
        args.recovery_required,
        csv_exporter)

    if args.parquet_required:
        parquet_exporter = ParquetExporter(args.input_path.parent)
        parquet_exporter.generate_parquet(consolidator.consolidate_indices_dataset())
    else:
        consolidator.generate_all_precipitation_series()

    if args.recovery_required and not args.keep_temp_files:
        consolidator.clear_temp_files()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "coordinates_path", metavar="path/to/coordinates.json", type=Path,
        help=(
            "path to a JSON file containing coordinates of the desired cities. "
            "Alternatively, a CSV file can be provided along with the --raw-coordinates "
            "option, so the operation is restricted to validating the coordinates and "
            "exporting them to a JSON file"))
    parser.add_argument(
        "input_path", metavar="path/to/netcdf4_dir", type=Path,
        help=(
            "path to a directory containing the input NetCDF4 files. The first file in the "
            "directory is used for coordinate validation if --raw-coordinates is present"))
    parser.add_argument(
        "-p", "--to-parquet", dest="parquet_required", action="store_true", default=False,
        help=(
            "indicates a consolidated Parquet file with precipitation indices should be "
            "exported. Ignored if --raw-coordinates is present"))
    parser.add_argument(
        "-c", "--to-csv", dest="csv_required", action="store_true", default=False, help=(
            "indicates a CSV file with datetime and precipitation data for each city, model"
            " and scenario should be generated. Ignored if --raw-coordinates is present"))
    parser.add_argument(
        "-n", "--to-netuno", dest="netuno_required", action="store_true", default=False,
        help=(
            "indicates CSV files should be exported containing only precipitation data "
            "(no headers). Overrides --to-csv. Ignored if --raw-coordinates is present"))
    parser.add_argument(
        "-r", "--raw-coordinates", action="store_true", dest="only_process_coordinates",
        default=False, help=(
            "indicates the coordinates file contains raw coordinates in CSV format, "
            "restricting the operation to their validation and creation of a formatted JSON"
            " file. Overrides --to-parquet, --to-csv- and --to-netuno"))
    parser.add_argument(
        "-k", "--keep-temp", action="store_true", dest="keep_temp_files", default=False,
        help=(
            "indicates the temporary binary recovery files created throughout the operation"
            " should be kept. Ignored if --raw-coordinates is present"))
    parser.add_argument(
        "--no-recovery", action="store_false", dest="recovery_required", default=True,
        help=(
            "indicates the temporary recovery files should NOT be created. Recovery files "
            "aid in case the operation cannot be completed in one run, but slow the process"
            " with extra serialization and I/O operations. Ignored if --raw-coordinates is "
            "present"))
    parser.add_argument(
        "-q", "--quiet", action="count", default=0,
        help="turn on quiet mode (cumulative), which hides log entries of levels lower "
        "than WARNING, then ERROR. Ignored if --verbose is present")
    parser.add_argument(
        "-v", "--verbose", action="store_true", default=False, help=(
            "turn on verbose mode, to display all log messages of level DEBUG and above. "
            "Overrides --quiet"))

    validator = CommandLineArgsValidator()
    parser.parse_args(namespace=validator)
    setup_logger(logger, validator.quiet, validator.verbose)

    try:
        validator.validate_arguments()
    except (InvalidCoordinatesFileError, InvalidSourceDirectoryError) as exception:
        logger.exception(f"Command line arguments validation failed. Details:\n{exception}")
    else:
        main(validator)
