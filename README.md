# Rainfall Series Transformation
Transformation of Brazilian '.nc' precipitation files (such as the ones found at [CLIMBra - Climate Change Dataset for Brazil](https://www.scidb.cn/en/detail?dataSetId=609b7ff93f0d4d1a9ba6eb709027c6ad), under `Gridded data/pr`") into location and model specific output files.

Climate change indices are computed according to their definition by the Expert Team on Climate Change Detection and Indices (ETCCDI), as available at [Climate Change Indices - Definitions of the 27 core indices](https://etccdi.pacificclimate.org/list_27_indices.shtml).

## Setup
```bash
git clone git@github.com:igorcmvaz/rainfall-series-transformation.git
cd rainfall-series-transformation
python -m venv .venv     # optional (but recommended) to create a virtual environment
python -m pip install -r requirements.txt
cd src
```

## Usage
### Validation of Cities Coordinates
In order to validate the coordinates, a reference of the actual coordinates is used (from a JSON file with city names and coordinates), as well as a NetCDF4 file with gridded precipitation data. Input coordinates should follow the format below:
```json
{
    "Aracaju": [-10.9472, -37.0731],
    "Belém": [-1.4558, -48.5039]
}
```

Output will be another JSON file with both original and validated (nearest) coordinates found in the given NetCDF4 file. Example:

```json
{
  "Aracaju": {
    "target": {
      "lat": -10.9472,
      "lon": -37.0731
    },
    "nearest": {
      "lat": -10.875,
      "lon": -37.125
    }
  },
  "Belém": {
    "target": {
      "lat": -1.4558,
      "lon": -48.5039
    },
    "nearest": {
      "lat": -1.375,
      "lon": -48.875
    }
  }
}
```


### Transforming to CSV
<!-- # TODO: improve instructions -->
This generates a series of CSV files, one per city per climate model, each with one data row for each day in the period specified. The inputs are NetCDF4 files and a file with city coordinates. Only precipitation data is present in the CSV files.


### Transforming to Parquet
<!-- # TODO: improve instructions -->
This generates a single Parquet file, containing data from each city based on each climate model. The inputs are NetCDF4 files and a file with city coordinates. Data is used to compute various Climate Change Indices related to precipitation and only those are stored. The parquet file, therefore, does not contain individual data points, rather only different groupings and aggregate indices.


# Commits
When committing to this repository, the following convention is advised:

* chore: regular maintenance unrelated to source code (dependencies, config, etc)
* docs: updates to any documentation
* feat: new features
* fix: bug fixes
* ref: refactored code (no new feature or bug fix)
* revert: reverts on previous commits
* test: updates to tests

For further reference on writing good commit messages, see [Conventional Commits](https://www.conventionalcommits.org).


# Roadmap
Next steps, planned development, pending issues, known bugs, etc:
- [ ] Add progress/file counter in [nc_to_csv_timeseries.py](/src/nc_to_csv_timeseries.py)
- [ ] Review computation of **CDD** and CWD metrics in [nc_to_parquet.py](/src/nc_to_parquet.py)
- [ ] Remove locations with same final coordinates in [validate_coordinates.py](/src/validate_coordinates.py)
- [ ] Reestructure code using classes to distribute functionality and provide abstractions between layers
- [ ] Add a single point of entry to the code
- [ ] Rename files and directories for improved readability
- [ ] Implement parallel computing for expensive functions
- [ ] Add tests
- [ ] Finish README, including proper instructions and examples
