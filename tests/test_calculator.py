import logging
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from agents.calculator import (
    CoordinatesFinder, IndicesCalculator, compute_seasonality_index, estimate_combinations,
    find_max_consecutive_run_length, logger)
from globals.errors import ReachedCoordinatesOffsetLimitError
from tests.samples.stub_netCDF4 import SAMPLE_NC_PATH, NetCDFStubGenerator
from tests.samples.stub_precipitation import PrecipitationGenerator
from tests.samples.stub_raw_coordinates import SAMPLE_RAW_CITIES_PATH


precipitation_generator = PrecipitationGenerator()


class TestMaxRunLengthFinder(unittest.TestCase):

    def test_short_numeric_run(self):
        data = [1, 0, 0, 0, 0]
        self.assertEqual(find_max_consecutive_run_length(pd.Series(data)), 1)

    def test_empty_series(self):
        data = []
        self.assertEqual(find_max_consecutive_run_length(pd.Series(data)), 0)

    def test_long_numeric_run(self):
        data = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 0, 0, 0]
        self.assertEqual(find_max_consecutive_run_length(pd.Series(data)), 12)

    def test_short_string_run(self):
        data = ["a", "a", "a", "", "", "", ""]
        self.assertEqual(find_max_consecutive_run_length(pd.Series(data)), 3)

    def test_short_mixed_run(self):
        data = ["a", "a", "a", 1, 1, 0, 0, 0, 0]
        self.assertEqual(find_max_consecutive_run_length(pd.Series(data)), 3)

    def test_negative_number_mixed_run(self):
        data = ["a", "a", "a", "", "", "", "", -1, -1, -1, -1, -1]
        self.assertEqual(find_max_consecutive_run_length(pd.Series(data)), 5)

    def test_distinct_truthy_values(self):
        data = ["1", 1, -1, "a", "B", 0, "", None]
        self.assertEqual(find_max_consecutive_run_length(pd.Series(data)), 1)

    def test_distinct_falsy_values(self):
        data = ["", "", "", 0, 0, None, False, False, False, False]
        self.assertEqual(find_max_consecutive_run_length(pd.Series(data)), 0)

    def test_sample_cdd_and_cwd_year_1980(self):
        YEAR = 1980
        data = precipitation_generator.by_year(YEAR)

        self.assertEqual(
            find_max_consecutive_run_length(data["dry_days"]),
            precipitation_generator.climate_indices[YEAR]["CDD"])
        self.assertEqual(
            find_max_consecutive_run_length(data["wet_days"]),
            precipitation_generator.climate_indices[YEAR]["CWD"])

    def test_sample_cdd_and_cwd_year_1984(self):
        YEAR = 1984
        data = precipitation_generator.by_year(YEAR)

        self.assertEqual(
            find_max_consecutive_run_length(data["dry_days"]),
            precipitation_generator.climate_indices[YEAR]["CDD"])
        self.assertEqual(
            find_max_consecutive_run_length(data["wet_days"]),
            precipitation_generator.climate_indices[YEAR]["CWD"])

    def test_sample_cdd_and_cwd_year_1990(self):
        YEAR = 1990
        data = precipitation_generator.by_year(YEAR)

        self.assertEqual(
            find_max_consecutive_run_length(data["dry_days"]),
            precipitation_generator.climate_indices[YEAR]["CDD"])
        self.assertEqual(
            find_max_consecutive_run_length(data["wet_days"]),
            precipitation_generator.climate_indices[YEAR]["CWD"])

    def test_sample_cdd_and_cwd_year_1999(self):
        YEAR = 1999
        data = precipitation_generator.by_year(YEAR)

        self.assertEqual(
            find_max_consecutive_run_length(data["dry_days"]),
            precipitation_generator.climate_indices[YEAR]["CDD"])
        self.assertEqual(
            find_max_consecutive_run_length(data["wet_days"]),
            precipitation_generator.climate_indices[YEAR]["CWD"])


class TestSeasonalityIndexComputation(unittest.TestCase):

    def test_sample_seasonalities_all_years(self):
        for year, indices in precipitation_generator.climate_indices.items():
            with self.subTest(year=year):
                data = precipitation_generator.by_year(year)

                self.assertAlmostEqual(
                    compute_seasonality_index(data), indices["seasonality_index"])

    def test_sample_with_no_seasonality(self):
        precipitation = [(month, 200) for month in range(1, 13)]
        data = pd.DataFrame(precipitation, columns=["month", "precipitation"])

        self.assertEqual(compute_seasonality_index(data), 0)

    def test_sample_medium_seasonality(self):
        precipitation = [
            (1, 480),
            (2, 420),
            (3, 550),
            (4, 150),
            (5, 120),
            (6, 120),
            (7, 180),
            (8, 600),
            (9, 500),
            (10, 150),
            (11, 150),
            (12, 150),
            ]
        data = pd.DataFrame(precipitation, columns=["month", "precipitation"])

        self.assertAlmostEqual(compute_seasonality_index(data), 0.595238095)

    def test_sample_high_seasonality(self):
        precipitation = [
            (1, 600),
            (2, 600),
            (3, 600),
            (4, 0),
            (5, 0),
            (6, 0),
            (7, 0),
            (8, 0),
            (9, 0),
            (10, 0),
            (11, 0),
            (12, 0),
            ]
        data = pd.DataFrame(precipitation, columns=["month", "precipitation"])

        self.assertEqual(compute_seasonality_index(data), 1.5)

    def test_seasonality_with_no_precipitation(self):
        precipitation = [(month, 0) for month in range(1, 13)]

        data = pd.DataFrame(precipitation, columns=["month", "precipitation"])

        self.assertTrue(np.isnan(compute_seasonality_index(data)))


class TestIndicesCalculator(unittest.TestCase):

    def setUp(self):
        self.calculator = IndicesCalculator(precipitation_generator.df)
        self.calculator.compute_climate_indices()

    def test_sample_rx1day_calculation(self):
        for year, indices in precipitation_generator.climate_indices.items():
            with self.subTest(year=year):
                self.assertAlmostEqual(self.calculator.rx1day.loc[year], indices["Rx1day"])

    def test_sample_rx5day_calculation(self):
        for year, indices in precipitation_generator.climate_indices.items():
            with self.subTest(year=year):
                self.assertAlmostEqual(self.calculator.rx5day.loc[year], indices["Rx5day"])

    def test_sample_SDII_calculation(self):
        for year, indices in precipitation_generator.climate_indices.items():
            with self.subTest(year=year):
                self.assertAlmostEqual(self.calculator.sdii.loc[year], indices["SDII"])

    def test_sample_R20mm_calculation(self):
        for year, indices in precipitation_generator.climate_indices.items():
            with self.subTest(year=year):
                self.assertEqual(self.calculator.r20mm.loc[year], indices["R20mm"])

    def test_sample_CDD_calculation(self):
        for year, indices in precipitation_generator.climate_indices.items():
            with self.subTest(year=year):
                self.assertEqual(self.calculator.cdd.loc[year], indices["CDD"])

    def test_sample_CWD_calculation(self):
        for year, indices in precipitation_generator.climate_indices.items():
            with self.subTest(year=year):
                self.assertEqual(self.calculator.cwd.loc[year], indices["CWD"])

    def test_sample_R95p_calculation(self):
        for year, indices in precipitation_generator.climate_indices.items():
            with self.subTest(year=year):
                self.assertAlmostEqual(self.calculator.r95p.loc[year], indices["R95p"])

    def test_sample_PRCPTOT_calculation(self):
        for year, indices in precipitation_generator.climate_indices.items():
            with self.subTest(year=year):
                self.assertAlmostEqual(
                    self.calculator.prcptot.loc[year], indices["PRCPTOT"])

    def test_sample_seasonality_calculation(self):
        for year, indices in precipitation_generator.climate_indices.items():
            with self.subTest(year=year):
                self.assertAlmostEqual(
                    self.calculator.seasonality_indices.loc[year],
                    indices["seasonality_index"])


class TestCombinationEstimation(unittest.TestCase):

    def test_single_argument(self):
        self.assertEqual(estimate_combinations([1]), 1)

    def test_nested_single_argument(self):
        self.assertEqual(estimate_combinations([[1, 2], [2, 3], [3, 4, 5, 6]]), 3)

    def test_multiple_sequences(self):
        self.assertEqual(estimate_combinations([1, 2], [2, 3], [3, 4, 5, 6]), 16)

    def test_multiple_dictionaries(self):
        self.assertEqual(estimate_combinations(
            {"a": 1, "b": 2},
            {"A": 3, "B": 4},
            {0: None}), 4)

    def test_empty_argument(self):
        self.assertEqual(estimate_combinations([1, 2], [2, 3], []), 0)

    def test_various_types(self):
        self.assertEqual(estimate_combinations(
            [1, 2], {"1": 1, "2": 2}, {1, 2, 3}, "1234"), 48)


class TestCoordinatesFinder(unittest.TestCase):

    def setUp(self):
        self.finder = CoordinatesFinder(SAMPLE_NC_PATH, SAMPLE_RAW_CITIES_PATH)
        sample_variables = NetCDFStubGenerator.create_sample_variables()
        self.sample_latitudes = sample_variables["lat"].values
        self.sample_longitudes = sample_variables["lon"].values
        self.STEP_SIZE = self.sample_latitudes[1] - self.sample_latitudes[0]

    def test_search_around_coordinates_success(self):
        LATITUDE_INDEX = LONGITUDE_INDEX = 1
        result = self.finder._search_around_coordinates(LATITUDE_INDEX, LONGITUDE_INDEX)
        self.assertTupleEqual((
                self.sample_latitudes[LATITUDE_INDEX],
                self.sample_longitudes[LONGITUDE_INDEX-1]),
                    result)

    def test_search_around_coordinates_failure(self):
        with self.assertRaises(ReachedCoordinatesOffsetLimitError), patch(
            "agents.validators.PrecipitationValidator.coordinates_have_precipitation_data"
                ) as precipitation_mock:
            precipitation_mock.return_value = False
            self.finder._search_around_coordinates(0, 0)

    def test_search_nearest_coordinates_exact_match(self):
        result = self.finder._search_nearest_coordinates(
            self.sample_latitudes[0], self.sample_longitudes[0])
        self.assertTupleEqual((self.sample_latitudes[0], self.sample_longitudes[0]), result)

    def test_search_nearest_coordinates_approximate_match(self):
        DIFF = self.STEP_SIZE / 4
        LATITUDE_INDEX = LONGITUDE_INDEX = 5
        TARGET_LATITUDE = self.sample_latitudes[LATITUDE_INDEX] + DIFF
        TARGET_LONGITUDE = self.sample_longitudes[LONGITUDE_INDEX] - DIFF
        result = self.finder._search_nearest_coordinates(TARGET_LATITUDE, TARGET_LONGITUDE)
        self.assertTupleEqual((
            self.sample_latitudes[LATITUDE_INDEX], self.sample_longitudes[LONGITUDE_INDEX]),
                result)

    def test_search_nearest_coordinates_far_match(self):
        DIFF = self.STEP_SIZE + 0.1
        LATITUDE_INDEX = LONGITUDE_INDEX = 5
        TARGET_LATITUDE = self.sample_latitudes[LATITUDE_INDEX] + DIFF
        TARGET_LONGITUDE = self.sample_longitudes[LONGITUDE_INDEX] - DIFF

        self.finder.precipitation.mask = np.zeros(
            shape=self.finder.precipitation.shape, dtype=bool)
        self.finder.precipitation.mask[:, LATITUDE_INDEX+1, LONGITUDE_INDEX-1] = True

        result = self.finder._search_nearest_coordinates(TARGET_LATITUDE, TARGET_LONGITUDE)
        self.assertTupleEqual((
            self.sample_latitudes[LATITUDE_INDEX+1],
            self.sample_longitudes[LONGITUDE_INDEX-2]
            ), result)

        self.finder.precipitation.mask = False

    def test_find_matching_coordinates_all_valid(self):
        EXPECTED_RESULT = {
            "Elsweyr": {
                "target": {
                    "lat": -34.0058,
                    "lon": -74.0085
                },
                "nearest": {
                    "lat": -34.125,
                    "lon": -74.125
                },
                "ibge_code": 8850308,
            },
            "Auridon": {
                "target": {
                    "lat": -33.6025,
                    "lon": -73.6052
                },
                "nearest": {
                    "lat": -33.625,
                    "lon": -73.625
                },
                "ibge_code": 8804557,
            },
            "Summerset": {
                "target": {
                    "lat": -32.9123,
                    "lon": -72.8035
                },
                "nearest": {
                    "lat": -32.875,
                    "lon": -72.875
                },
                "ibge_code": 8500108,
            },
            "Cyrodiil": {
                "target": {
                    "lat": -31.4040,
                    "lon": -71.3033
                },
                "nearest": {
                    "lat": -31.375,
                    "lon": -71.375
                },
                "ibge_code": 9904400,
            },
            "Skyrim": {
                "target": {
                    "lat": -30.1111,
                    "lon": -70.1313
                },
                "nearest": {
                    "lat": -30.125,
                    "lon": -70.125
                },
                "ibge_code": 8927408,
            },
        }
        result = self.finder.find_matching_coordinates()
        self.assertDictEqual(EXPECTED_RESULT, result)

    def test_log_record_from_failed_find_matching_coordinates(self):
        EXPECTED_LOG_MESSAGE = (
            "Could not find valid precipitation data near the original coordinates (")
        with self.assertLogs(logger, level=logging.ERROR) as log_context, patch(
            "agents.validators.PrecipitationValidator.coordinates_have_precipitation_data"
                ) as precipitation_mock:
            precipitation_mock.return_value = False
            result = self.finder.find_matching_coordinates()
            self.assertIn(EXPECTED_LOG_MESSAGE, log_context.output[-1])
            self.assertDictEqual({}, result)


if __name__ == '__main__':
    unittest.main()
