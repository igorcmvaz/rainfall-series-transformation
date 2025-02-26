import json
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from constants import INPUT_FILENAME_FORMAT
from tests.stub_precipitation import SAMPLE_JSON_PATH


def main():
    with open(SAMPLE_JSON_PATH) as file:
        content = json.load(file)

    output_path = Path(
        SAMPLE_JSON_PATH.parent,
        INPUT_FILENAME_FORMAT[content["scenario"]].format(model=content["model"]))
    sample_precipitation = [value[1] for value in content["data"]]
    period = len(content["data"])

    with Dataset(output_path, mode="w", format="NETCDF4") as dataset:
        dataset.creation_date = "2025-01-01"
        dataset.frequency = "day"
        dataset.mip_era = "CMIP6"
        dataset.source_id = "ACCESS-CM2-"
        dataset.variable_id = "pr"

        dataset.createDimension("lon", 1)
        dataset.createDimension("lat", 1)
        dataset.createDimension("time", period)

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

        longitudes[:] = [-27.625]
        latitudes[:] = [-48.875]
        times[:] = np.arange(period)
        precipitation[:] = sample_precipitation


if __name__ == "__main__":
    main()
