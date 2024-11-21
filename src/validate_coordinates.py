import json
import logging
from argparse import ArgumentParser
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.ma import MaskedArray
from netCDF4 import Dataset, Variable



def generate_valid_coordinates_json(
        source_path: Path,
        original_coordinates: dict[str, Sequence[float]],
        output_file_path: Path) -> None:
    """
    Generate a JSON file with coordinates of cities corresponding to data available from a
    precipitation dataset.

    Args:
        source_path (Path): Path to the NetCDF4 precipitation dataset file.
        original_coordinates (dict[str, Sequence[float]]): Dictionary mapping city names to
            target latitude and longitude.
        output_file_path (Path): Path where the generated JSON file containing valid
            coordinates should be saved.
    """

    valid_coordinates: dict[str, dict[str, tuple[float, float]]] = {}

    for city_name, (target_latitude, target_longitude) in original_coordinates.items():
        valid_latitude, valid_longitude = find_nearest_valid_coordinate(
            latitudes, longitudes, precipitation, target_latitude, target_longitude)

        if all((valid_latitude, valid_longitude)):
            valid_coordinates[city_name] = {
                "Target Coordinates": (target_latitude, target_longitude),
                "Nearest Coordinates": (valid_latitude, valid_longitude)
            }
        else:
            logging.error(
                f"Could not find valid coordinates for city '{city_name}' in file at "
                f"'{source_path.resolve()}'")

    with open(output_file_path, "w", encoding="utf-8") as file:
        json.dump(valid_coordinates, file, indent=2, ensure_ascii=False)
    logging.info(
        f"Successfully generated coordinates file at '{output_file_path.resolve()}'")


def has_precipitation_data(
        precipitation: Variable,
        latitude_index: int,
        longitude_index: int) -> bool:
    """
    Checks if there are any valid (non-missing) precipitation data points in the dataset for
    given coordinates.

    Args:
        precipitation (Variable): Multidimensional array representing precipitation data
            (including time and coordinates).
        latitude_index (int): Index of the desired latitude dimension in the precipitation
            dataset.
        longitude_index (int): Index of the desired longitude  dimension in the
            precipitation dataset.

    Returns:
        bool: True if there is at least one valid data point for the specified coordinates,
        False otherwise.
    """
    return not np.all(np.ma.getmaskarray(precipitation[:, latitude_index, longitude_index]))


def find_nearest_valid_coordinate(
        latitudes: Variable,
        longitudes: Variable,
        precipitation: Dataset,
        target_latitude: float,
        target_longitude: float) -> tuple[float | None, float | None]:
    """
    Finds the nearest valid coordinates that contain precipitation data in a given dataset.

    Uses an expanding spiral pattern until a maximum 'distance' from the original coordinate
    indices is reached.

    Args:
        latitudes (Variable): 1D array containing the latitude values in the dataset.
        longitudes (Variable): 1D array containing the longitude values in the dataset.
        precipitation (Dataset): Multidimensional array representing precipitation data
            (including time and coordinates).
        target_latitude (float): Latitude component for which to find the nearest valid
            coordindates.
        target_longitude (float): Longitude component for which to find the nearest valid
            coordindates.

    Returns:
        tuple[float | None, float | None]: A tuple containing the nearest coordinates that
        contain valid precipitation data, if found. Else, a tuple of (None, None).
    """

    latitude_index: int = np.abs(latitudes - target_latitude).argmin()
    longitude_index: int = np.abs(longitudes - target_longitude).argmin()
    if has_precipitation_data(precipitation, latitude_index, longitude_index):
        logging.debug(
            f"Found precipitation data at coordinates "
            f"({latitudes[latitude_index]}, {longitudes[longitude_index]})")
        return latitudes[latitude_index], longitudes[longitude_index]

    offset = 1
    checked_offsets = {(0, 0)}
    MAX_OFFSET = 15
    while offset <= MAX_OFFSET:
        for latitude_offset in (limits := sorted(range(-offset, offset + 1), key=abs)):
            for longitude_offset in limits:
                if (latitude_offset, longitude_offset) in checked_offsets:
                    continue
                checked_offsets.add((latitude_offset, longitude_offset))

                new_latitude_index = latitude_index + latitude_offset
                new_longitude_index = longitude_index + longitude_offset
                if has_precipitation_data(
                        precipitation, new_latitude_index, new_longitude_index):
                    logging.debug(
                        f"Found precipitation data at coordinates "
                        f"({latitudes[new_latitude_index]}, "
                        f"{longitudes[new_longitude_index]})")
                    return latitudes[new_latitude_index], longitudes[new_longitude_index]
        offset += 1

    logging.warning(
        f"Could not find valid precipitation data at or around initial coordinates "
        f"({target_latitude}, {target_longitude}). Max index offset from target "
        f"coordinates: {MAX_OFFSET}.")
    return None, None


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

    coordinates_path: Path = Path(args.coordinates)
    if not coordinates_path.is_file():
        logging.critical(
            f"File with cities coordinates not found at '{coordinates_path.resolve()}'")
        return

    reference_path: Path = Path(args.reference)
    if not reference_path.is_file():
        logging.critical(
            f"File with geo-located references not found at '{reference_path.resolve()}'")
        return

    output_dir: Path = Path(args.output)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logging.exception(
            f"Could not create directory at '{output_dir.resolve()}', default output "
            f"directory will be used. Details: {e}")
        output_dir = Path("sample_output")

    output_dir.mkdir(exist_ok=True)
    logging.debug(f"Output directory set to '{output_dir.resolve()}'")
    output_file_path = Path(
        output_dir, f"validated_{coordinates_path.stem}").with_suffix(".json")

    with open(coordinates_path) as file:
        original_coordinates: dict[str, Sequence[float]] = json.load(file)

    generate_valid_coordinates_json(reference_path, original_coordinates, output_file_path)
    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "coordinates", type=str, metavar="path/to/coordinates.json",
        help="path to a JSON file containing city names and their coordinates")
    parser.add_argument(
        "reference", type=str, metavar="path/to/reference.nc",
        help="path to a NetCDF4 file containing geo-located precipitation data")
    parser.add_argument(
        "-o", "--output", metavar="path/to/output", default="sample_output",
        help="path to the directory where the output JSON file will be saved. Creates it "
        "if it doesn't exist. Defaults to './sample_output'")
    parser.add_argument(
        "-q", "--quiet", action="count", default=0,
        help="turn on quiet mode (cumulative), which hides log entries of levels lower "
        "than INFO/WARNING")
    main(parser.parse_args())
