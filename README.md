# Rainfall Series Transformation

Transformation of Brazilian NetCDF4 precipitation files (such as the ones found at [CLIMBra - Climate Change Dataset for Brazil](https://www.scidb.cn/en/detail?dataSetId=609b7ff93f0d4d1a9ba6eb709027c6ad), under `Gridded data/pr`") into location and model specific output files. Precipitation data is extracted from the closest grid point found in the dataset, up to a certain limit, if available. It is worth noting that missing data (`NaN`, `None`, `null`, etc) is interpreted as `0.0` (no precipitation) rather than raising errors.

Climate change indices are computed according to their definition by the Expert Team on Climate Change Detection and Indices (ETCCDI), as available at [Climate Change Indices - Definitions of the 27 core indices](https://etccdi.pacificclimate.org/list_27_indices.shtml). Consolidated climate indices are exported to a Parquet file, and the precipitation data series can be optionally exported to CSV files in two distinct formats: a standard format that includes precipitation data and datetime values corresponding to each day (as well as file headers); a simpler format that only includes precipitation data (one row for each day), with no datetime or headers (adequate for usage in software such as Netuno).

The application is developed such that all functionality is available through a single interface file, [transform.py](/transform.py), which is supposed to be executed with command-line arguments in order to define the actual operation to be executed and the source of data. The reference climate models and scenarios are defined as `CLIMATE_MODELS` and `SSP_SCENARIOS` at [`constants.py`](globals/constants.py).

## Setup

```bash
git clone git@github.com:igorcmvaz/rainfall-series-transformation.git
cd rainfall-series-transformation
python3.x -m venv .venv     # use any 3.10+ Python version
source .venv/bin/activate   # activate your virtual environment before installing anything
python -m pip install -r requirements.txt
python transform.py -h
```

## Usage

### Validation of Cities Coordinates

In order to validate the coordinates, a file with the target coordinates is used (CSV file with city names, [IBGE code](https://www.ibge.gov.br/explica/codigos-dos-municipios.php) and coordinates), as well as a NetCDF4 file with gridded precipitation data expected to contain the same (or close) locations. Input coordinates should follow the format below (whitespace added for legibility):

```csv
ibge_code, city          , latitude, longitude
3550308  , São Paulo     , -23.533 , -46.64
3304557  , Rio de Janeiro, -22.913 , -43.2
5300108  , Brasília      , -15.78  , -47.93
```

Output will be a JSON file of the same name containing both original and validated (nearest) coordinates found in the given NetCDF4 file. The keys (which are city names) are sorted in alphabetical order (special/accented characters at the end). Example:

```json
{
  "Brasília": {
    "ibge_code": 5300108,
    "nearest": {
      "lat": -15.875,
      "lon": -47.875
    },
    "target": {
      "lat": -15.78,
      "lon": -47.93
    }
  },
    "Rio de Janeiro": {
      "ibge_code": 3304557,
      "nearest": {
        "lat": -22.875,
        "lon": -43.375
      },
      "target": {
        "lat": -22.913,
        "lon": -43.2
      }
    },
  "São Paulo": {
    "ibge_code": 3550308,
    "nearest": {
      "lat": -23.625,
      "lon": -46.625
    },
    "target": {
      "lat": -23.533,
      "lon": -46.64
    }
  }
}
```

In order to, specifically, run the program to validate an existing CSV file with coordinates, use the `transform.py` file as follows:

```bash
python transform.py path/to/raw_coordinates.csv path/to/netcdf_dir --raw-coordinates
python transform.py path/to/raw_coordinates.csv path/to/netcdf_dir --raw-coordinates --verbose  # to show DEBUG logs
python transform.py path/to/raw_coordinates.csv path/to/netcdf_dir --raw-coordinates -q         # to hide INFO logs
python transform.py path/to/raw_coordinates.csv path/to/netcdf_dir --raw-coordinates -qq        # to also hide WARNING logs
```

This command would use the smallest file found at `path/to/netcdf_dir` to validate the coordinates found in `path/to/raw_coordinates.csv` and generate a corresponding JSON file at `path/to/raw_coordinates.json`.

### Transforming to Parquet

The main goal of the application is to generate **climate indices from precipitation** data for given locations, using both historic data and climate models applied in different climate scenarios, such as the Shared Socioeconomic Pathways (SSPs) defined as `SSP2-4.5` and `SSP5-8.5` (refer to [CMIP6 and Shared Socio-economic Pathways overview](https://climate-scenarios.canada.ca/?page=cmip6-overview-notes#shared-socio-economic-pathways(ssps))). The output is a single Parquet file, containing climate indices for each city of interest based on each climate model for each scenario. The original individual data points are *not* included in the output, only their aggregation in the form of these indices. The climate indices included in the current version are the following:

* RX1day: Monthly maximum 1-day precipitation.
* RX5day: Monthly maximum consecutive 5-day precipitation.
* SDII: Simple precipitation intensity index (mean precipitation on wet days).
* R20mm: Annual count of days when precipitation ≥ 20mm.
* CDD: Maximum length of dry spell, maximum number of consecutive days with precipitation < 1mm.
* CWD: Maximum length of wet spell, maximum number of consecutive days with precipitation ≥ 1mm.
* R95p: Annual total precipitation from days exceeding the 95th percentile for the entire period.
* PRCPTOT: Annual total precipitation in wet days.
* Seasonality Index: Seasonality index quantifying seasonal variation within a year.

For more details about the climate indices, see the docstring from [`IndicesCalculator.compute_climate_indices`](agents/calculator.py).

In order to execute such operation, the application requires a source of NetCDF4 files and a file with the desired city coordinates. Here's an example:

```bash
python transform.py path/to/validated_coordinates.json path/to/netcdf_dir --to-parquet
python transform.py path/to/validated_coordinates.json path/to/netcdf_dir --to-parquet --verbose      # to show DEBUG logs
python transform.py path/to/validated_coordinates.json path/to/netcdf_dir --to-parquet -q             # to hide INFO logs
python transform.py path/to/validated_coordinates.json path/to/netcdf_dir --to-parquet -qq            # to also hide WARNING logs
python transform.py path/to/validated_coordinates.json path/to/netcdf_dir --to-parquet --keep-temp    # to keep the temporary recovery files
python transform.py path/to/validated_coordinates.json path/to/netcdf_dir --to-parquet --no-recovery  # to skip the creation of temporary recovery files
```

This command would use the (already validated) coordinates found in `path/to/validated_coordinates.json` and go through each NetCDF4 file in `path/to/netcdf_dir`, evaluating the climate indices for each combination of city, climate model and climate scenario, then output a consolidated Parquet file with timestamp such as `path/to/2025-01-01T09-15-consolidated.parquet`.

The operation has an optional **recovery method** based on temporary binary files, which are written to a temporary directory created specifically for this purpose at the directory above the NetCDF4 directory. These recovery files are useful in case a previous operation could not be properly finished due to the circumstances, so that, when executed again, any results from climate models and scenarios already covered will be retrieved from these files rather than have all extraction and computation processes take place again. This also allows one to manually interrupt the operation without fear of losing much of the current progress. The maximum progress that can be lost is restricted to the operation of the latest combination of model and scenario (recovery files are saved for each such combination). The `--keep-temp` option makes it so these temporary recovery files are not deleted at the end of the operation. The creation of new recovery files can be skipped entirely using the `--no-recovery` option, while still retrieving any data from already existing recovery files.

### Transforming to CSV

An alternative use of the application is to export the actual precipitation data for the desired cities, given the various combinations of climate models and scenarios. Since the data is available during the process to be able to properly compute the climate indices, it is also possible to export it without going through this computation step. The option to export the series of precipitation data is independent from the option to export a Parquet file with climate indices and both can be done during the same execution.

The same input arguments are required to use the application in this format. Here's an example:

```bash
python transform.py path/to/validated_coordinates.json path/to/netcdf_dir --to-csv
python transform.py path/to/validated_coordinates.json path/to/netcdf_dir --to-netuno           # for CSV files to have only precipitation values and no header
python transform.py path/to/validated_coordinates.json path/to/netcdf_dir --to-parquet --to-csv # to export a consolidated Parquet file and the CSVs for each city/model/scenario
```

The output from the CSV operation is a new directory labeled with a timestamp (such as `path/2025-01-01T09-15-output`) containing one CSV file for each combination of city, model and scenario. The output file name has the format `{city}_{model}_{scenario}.csv` and might include a `(Netuno)` flag at the beginning, such as `(Netuno)Brasília_GFDL-ESM4_SSP245.csv`.

### Samples

In the [example](/example) directory, one can find files that can be used to run the program with a reduced scope, in order to check its operation and understand the process. Here are the details of the resources available in that directory:

* `/CLIMBra` directory: a directory to store any NetCDF4 files (which are not stored in the repository because of their size). By default, it contains only a blank [`.gitkeep`](/example/CLIMBra/.gitkeep) file, which is used to ensure the directory is tracked by Git (empty directories cannot be tracked).
* `brazilian_cities_over_50k.csv`: file with raw coordinates for all brazilian cities with estimated population of at least 50k people.
* `brazilian_cities_over_50k.json`: file with validated coordinates (from the corresponding CSV file) for all brazilian cities with estimated population of at least 50k people.

So, in order to use these sample files, follow this process:

```bash
# Download a NetCDF4 file (for example, "Gridded data/pr/hist/ACCESS-CM2-pr-hist.nc") and save it with corresponding name at "/example/CLIMBra/". This might take a couple minutes to complete since the file has around 250MB
curl -o example/CLIMBra/ACCESS-CM2-pr-hist.nc https://china.scidb.cn/download?fileId=42efd8ad21bf9790fc8044cf247e1808&traceId=58820d62-21ec-4a52-a95b-be0fd14066f1

# Validate the coordinates of the cities in the example CSV (this step is optional, and will overwrite the existing JSON file -- with exactly the same content)
python transform.py example/brazilian_cities_over_50k.csv example/CLIMBra --raw-coordinates

# Run the program with the corresponding parameters to generate a consolidated Parquet file
python transform.py example/brazilian_cities_over_50k.json example/CLIMBra --to-parquet

# Or, run the program with the corresponding parameters to generate CSV files with precipitation data
python transform.py example/brazilian_cities_over_50k.json example/CLIMBra --to-csv

# Or, generate precipitation data in Netuno format
python transform.py example/brazilian_cities_over_50k.json example/CLIMBra --to-netuno
```

## Tests

All the tests are contained in the [tests](/tests/) directory, along with any sample files and stubs required to effectively run the tests. The dependencies for tests are already included in the [requirements](/requirements.txt) file, and they should be only a few - besides `pytest` itself and `pytest-cov` (only required if coverage analysis is desired), the test suite for the current version depends only on `time-machine`, specifically for tests involving datetimes.

In order to run the test suite (entirely, or part of it), see the example commands below:

```bash
python -m pytest
python -m pytest -v               # verbose option
python -m pytest -o log_cli=true  # to show application logs in the terminal
python -m pytest --cov            # to run coverage analysis (requires pytest-cov)
```

## Commits

When committing to this repository, the following convention is advised:

* chore: regular maintenance unrelated to source code (dependencies, config, etc)
* docs: updates to any documentation
* feat: new features
* fix: bug fixes
* ref: refactored code (no new feature or bug fix)
* test: updates to tests

For further reference on writing good commit messages, see [Conventional Commits](https://www.conventionalcommits.org).

## Roadmap

Next steps, planned development, pending issues, known bugs, etc:

* [x] Add progress/file counter in [nc_to_csv_timeseries.py](/src/nc_to_csv_timeseries.py)
* [x] Review validations for precipitation data from NetCDF4 files
* [x] Review computation of **CDD** and CWD metrics in [nc_to_parquet.py](/src/nc_to_parquet.py)
* [x] Reestructure code using classes to distribute functionality and decouple sections
* [x] Add tests
* [x] Add new console argument to export for Netuno (only precipitation values, no header)
* [x] Rename files and directories for improved readability (move stubs and samples into sub directory)
* [x] Reimplement coordinates validation as a dedicated option (not part of the main workflow)
* [x] Cache results from precipitation extraction not to re-extract data that was already extracted (in case of equal coordinates in different cities)
* [x] Check throughout the code which functions can be cached for improved performance
* [x] Add a single point of entry to the code
* [x] Break the process by batches and create recovery method (temporary directory from which a last saved state can be recovered, for long operations, excluded upon successful completion)
* [x] Finish README, including proper instructions and examples
* [ ] Add docstrings to classes
* [ ] Implement parallel computing for expensive functions
* [ ] Implement logging off the main thread
