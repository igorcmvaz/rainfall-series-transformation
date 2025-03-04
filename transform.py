import logging
from argparse import ArgumentParser
from pathlib import Path

from agents.calculator import CoordinatesFinder
from agents.consolidator import Consolidator
from agents.exporters import (
    JSONCoordinatesExporter, ParquetExporter, CSVExporter, NetunoExporter)
from agents.extractors import StructuredCoordinatesExtractor
from agents.validators import CommandLineArgsValidator
from globals.constants import CLIMATE_MODELS, SSP_SCENARIOS
from globals.errors import InvalidCoordinatesFileError, InvalidSourceDirectoryError

logger = logging.getLogger("rainfall_transformation")


def setup_logger(quiet_count: int, verbose: bool) -> None:
    """
    Configure logging for the application, defining format and log level according to quiet
    or verbose arguments.

    Args:
        quiet_count (int): Number of times the 'quiet' flag was provided.
        verbose (bool): Whether the 'verbose' flag was provided.
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    if quiet_count == 1:
        log_level = logging.WARNING
    elif quiet_count >= 2:
        log_level = logging.ERROR

    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8.8s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        level=log_level,
    )


def find_smallest_file(directory_path: Path) -> Path:
    """
    Finds and returns the smallest file (in bytes) from a given directory.

    Args:
        directory_path (Path): Path to the directory.

    Returns:
        Path: Path to the smallest file (in bytes) found in the directory.
    """
    return min(directory_path.iterdir(), key=lambda p: p.stat().st_size)


def process_coordinates_files(coordinates_path: Path, input_path: Path) -> None:
    """
    Uses the smallest file (assuming NetCDF4 format) from a given path to validate the
    coordinates of found in another (assumed CSV) file and exports a validated JSON file.

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


def main(args: CommandLineArgsValidator) -> None:
    if args.only_process_coordinates:
        return process_coordinates_files(args.coordinates_path, args.input_path)
    city_coordinates = StructuredCoordinatesExtractor(
        args.coordinates_path).get_coordinates()

    csv_exporter = None
    if args.netuno_required:
        csv_exporter = NetunoExporter(args.input_path.parent)
    elif args.csv_required:
        csv_exporter = CSVExporter(args.input_path.parent)
    consolidator = Consolidator(
        city_coordinates, SSP_SCENARIOS, CLIMATE_MODELS, args.input_path, csv_exporter)

    if args.parquet_required:
        parquet_exporter = ParquetExporter(args.input_path.parent)
        parquet_exporter.generate_parquet(consolidator.consolidate_indices_dataset())
    else:
        consolidator.generate_all_precipitation_series()
    if not args.keep_temp_files:
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
            "whether a consolidated Parquet file with precipitation indices should be "
            "exported. Ignored if --raw-coordinates is present"))
    parser.add_argument(
        "-c", "--to-csv", dest="csv_required", action="store_true", default=False, help=(
            "whether a CSV file with datetime and precipitation data for each city, model "
            "and scenario should be generated. Ignored if --raw-coordinates is present"))
    parser.add_argument(
        "-n", "--to-netuno", dest="netuno_required", action="store_true", default=False,
        help=(
            "whether CSV files should be exported containing only precipitation data "
            "(no headers). Overrides --to-csv. Ignored if --raw-coordinates is present"))
    parser.add_argument(
        "-r", "--raw-coordinates", action="store_true", dest="only_process_coordinates",
        default=False, help=(
            "whether the coordinates file contains raw coordinates in CSV format, "
            "restricting the operation to their validation and creation of a formatted JSON"
            " file. Overrides --to-parquet, --to-csv- and --to-netuno"))
    parser.add_argument(
        "-k", "--keep-temp", action="store_true", dest="keep_temp_files", default=False,
        help=(
            "whether to keep the temporary binary recovery files created throughout the "
            "operation. Ignored if --raw-coordinates is present "))
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
    setup_logger(validator.quiet, validator.verbose)

    try:
        validator.validate_arguments()
    except (InvalidCoordinatesFileError, InvalidSourceDirectoryError) as exception:
        logger.exception(f"Command line arguments validation failed. Details:\n{exception}")
    else:
        main(validator)
