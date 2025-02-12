from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from extractor import ParsedVariable

SAMPLE_NC_PATH = Path(__file__).parent / "sample.nc"


class NetCDFStubGenerator:

    rng = np.random.default_rng(seed=42)
    precipitation_sample = rng.uniform(size=(100, 168, 162))

    @classmethod
    def create_sample_stub(cls, output_path: Path) -> None:
        with Dataset(output_path, mode="w", format="NETCDF4") as dataset:
            dataset.creation_date = "2025-01-01"
            dataset.frequency = "day"
            dataset.mip_era = "CMIP6"
            dataset.source_id = "ACCESS-CM2-"
            dataset.variable_id = "pr"

            dataset.createDimension("lon", 162)
            dataset.createDimension("lat", 168)
            dataset.createDimension("time", 100)

            longitudes = dataset.createVariable("lon", "f4", ("lon",))
            longitudes.units = "degrees_east"
            longitudes.standard_name = "longitude"

            latitudes = dataset.createVariable("lat", "f4", ("lat",))
            latitudes.standard_name = "latitude"
            latitudes.units = "degrees_north"

            times = dataset.createVariable("time", "f8", ("time",))
            times.standard_name = "time"
            times.calendar = "gregorian"
            times.units = "days since 2020-01-01T00:00:00"

            precipitation = dataset.createVariable("pr", "f8", ("time", "lat", "lon",))
            precipitation.units = "mm"

            longitudes[:] = [-74.125 + 0.25*step for step in range(162)]
            latitudes[:] = [-34.125 + 0.25*step for step in range(168)]
            times[:] = np.arange(100)
            precipitation[:, :, :] = cls.precipitation_sample

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
