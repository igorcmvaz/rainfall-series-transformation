from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from extractor import ParsedVariable

SAMPLE_NC_PATH = Path(__file__).parent / "sample.nc"


class NetCDFStubGenerator:

    rng = np.random.default_rng(seed=42)
    precipitation_sample = rng.uniform(size=(100, 168, 162))

    @classmethod
    def _generate_sample_netcdf4(
            cls,
            output_path: Path,
            creation_date: str,
            time_length: int,
            reference_date: str,
            latitudes: list[float],
            longitudes: list[float],
            precipitation: list[float]) -> None:
        with Dataset(output_path, mode="w", format="NETCDF4") as dataset:
            dataset.creation_date = creation_date
            dataset.frequency = "day"
            dataset.mip_era = "CMIP6"
            dataset.source_id = "ACCESS-CM2-"
            dataset.variable_id = "pr"

            dataset.createDimension("lon", len(longitudes))
            dataset.createDimension("lat", len(latitudes))
            dataset.createDimension("time", time_length)

            longitude_variable = dataset.createVariable("lon", "f4", ("lon",))
            longitude_variable.units = "degrees_east"
            longitude_variable.standard_name = "longitude"

            latitude_variable = dataset.createVariable("lat", "f4", ("lat",))
            latitude_variable.standard_name = "latitude"
            latitude_variable.units = "degrees_north"

            time_variable = dataset.createVariable("time", "f8", ("time",))
            time_variable.standard_name = "time"
            time_variable.calendar = "gregorian"
            time_variable.units = f"days since {reference_date}T00:00:00"

            precipitation_variable = dataset.createVariable(
                "pr", "f8", ("time", "lat", "lon",))
            precipitation_variable.units = "mm"

            longitude_variable[:] = longitudes
            latitude_variable[:] = latitudes
            time_variable[:] = np.arange(time_length)
            precipitation_variable[:] = precipitation

    @classmethod
    def create_sample_stub(cls, output_path: Path) -> None:
        """
        Creates a NetCDF4 file with sample values for latitude, longitude, time and
        precipitation.

        Args:
            output_path (Path): Path to the output NetCDF4 file (including extension).
        """
        cls._generate_sample_netcdf4(
            output_path=output_path,
            creation_date="2025-01-01",
            time_length=100,
            reference_date="2020-01-01",
            latitudes=[-34.125 + 0.25*step for step in range(168)],
            longitudes=[-74.125 + 0.25*step for step in range(162)],
            precipitation=cls.precipitation_sample)

    @classmethod
    def create_sample_variables(cls) -> dict[str, ParsedVariable]:
        return {
            "lon": ParsedVariable(
                "degrees_east", np.ma.masked_array(
                    [-74.125 + 0.25*step for step in range(162)])),
            "lat": ParsedVariable(
                "degrees_north", np.ma.masked_array(
                    [-34.125 + 0.25*step for step in range(168)])),
            "time": ParsedVariable(
                "days since 2020-01-01T00:00:00", np.ma.masked_array(np.arange(100))),
            "pr": ParsedVariable(
                "mm", np.ma.masked_array(cls.precipitation_sample)),
        }


if __name__ == "__main__":
    NetCDFStubGenerator.create_sample_stub(SAMPLE_NC_PATH)
