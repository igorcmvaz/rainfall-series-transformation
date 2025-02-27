import json
import shutil
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from agents.validators import CoordinatesValidator, PathValidator, PrecipitationValidator
from globals.constants import CLIMATE_MODELS, INPUT_FILENAME_FORMAT, SSP_SCENARIOS
from globals.errors import (
    CoordinatesNotAvailableError, InvalidClimateScenarioError, InvalidSourceFileError)

SAMPLE_CITIES_PATH = Path(Path(__file__).parent, "samples", "sample_city_coordinates.json")


class TestNormalization(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(seed=42)
        self.mask = np.array([
            [[
                bool((t+1)*(lat+1)*(lon+1) % 3) for lon in range(2)
            ] for lat in range(3)] for t in range(20)
        ])

    def test_normalize_data_series_only_numeric(self):
        data_series = self.rng.uniform(0, 100, size=(20, 3, 2))
        masked_data = np.ma.masked_array(data_series, self.mask)
        expected_result = np.array([data_series[index, 0, 0] if not self.mask[
            index, 0, 0] else 0 for index in range(len(self.mask[:, 0, 0]))
        ])

        result = PrecipitationValidator.normalize_data_series(masked_data, 0, 0)

        self.assertListEqual(result.tolist(), expected_result.tolist())

    def test_normalize_data_series_only_nan(self):
        data_series = np.empty((20, 3, 2))
        data_series[:] = np.nan
        masked_data = np.ma.masked_array(data_series, self.mask)
        expected_result = np.array([0 for _ in range(len(self.mask[:, 0, 0]))])

        result = PrecipitationValidator.normalize_data_series(masked_data, 0, 0)

        self.assertListEqual(result.tolist(), expected_result.tolist())

    def test_normalize_data_series_mixed(self):
        data_series = self.rng.uniform(0, 100, size=(20, 3, 2))
        data_series[2:14, 1:3, 0] = np.nan
        masked_data = np.ma.masked_array(data_series, self.mask)
        expected_result = np.array([
            data_series[index, 1, 0] if (
                not self.mask[index, 1, 0] and index not in range(2, 14)
                ) else 0 for index in range(len(self.mask[:, 1, 0]))
        ])

        result = PrecipitationValidator.normalize_data_series(masked_data, 1, 0)

        self.assertListEqual(result.tolist(), expected_result.tolist())


class TestDateFiltering(unittest.TestCase):

    def setUp(self):
        SIZE = 20
        self.rng = np.random.default_rng(seed=42)
        values = self.rng.uniform(0, 100, size=SIZE)
        self.data_series = np.array([
            (datetime(2020, 1, 1) + timedelta(days=t), values[t]) for t in range(SIZE)
        ])
        self.earlier_date = datetime(2020, 1, 1)
        self.later_date = datetime(2020, 1, SIZE)

    def test_filter_by_date_longer_period(self):
        result = PrecipitationValidator.filter_by_date(
            self.data_series, self.earlier_date, self.later_date)

        self.assertListEqual(result.tolist(), self.data_series.tolist())

    def test_filter_by_date_starts_later(self):
        OFFSET = 2
        start_date = self.earlier_date + timedelta(days=OFFSET)
        result = PrecipitationValidator.filter_by_date(
            self.data_series, start_date, self.later_date)

        self.assertListEqual(result.tolist(), self.data_series[OFFSET:].tolist())

    def test_filter_by_date_ends_earlier(self):
        OFFSET = 2
        end_date = self.later_date - timedelta(days=OFFSET)
        result = PrecipitationValidator.filter_by_date(
            self.data_series, self.earlier_date, end_date)

        self.assertListEqual(result.tolist(), self.data_series[:-OFFSET].tolist())

    def test_filter_by_date_shorter_period(self):
        OFFSET = 2
        start_date = self.earlier_date + timedelta(days=OFFSET)
        end_date = self.later_date - timedelta(days=OFFSET)
        result = PrecipitationValidator.filter_by_date(
            self.data_series, start_date, end_date)

        self.assertListEqual(result.tolist(), self.data_series[OFFSET:-OFFSET].tolist())


class TestCoordinatesValidation(unittest.TestCase):

    def setUp(self):
        with open(SAMPLE_CITIES_PATH) as file:
            self.cities: dict[str, dict[str, dict[str, float]]] = json.load(file)

    def test_retrieve_coordinates_from_sample_cities(self):
        for city, details in self.cities.items():
            with self.subTest(city=city):
                latitude = details["nearest"]["lat"]
                longitude = details["nearest"]["lon"]
                self.assertTupleEqual(
                    (latitude, longitude), CoordinatesValidator.get_coordinates(details))

    def test_coordinates_not_available(self):
        cities = {
            "Florianópolis": {
                "nearest": {
                    "not_lat": 12.5,
                    "not_lon": 13.5
                }
            },
            "Rio de Janeiro": {
                "nearest": {
                    "lat": 12.5,
                    "not_lon": 13.5
                }
            },
            "São Paulo": {
                "nearest": {
                    "not_lat": 12.5,
                    "lon": 13.5
                }
            },
            "Vitória": {
                "not_nearest": {
                    "lat": -20.375,
                    "lon": -40.375
                }
            }
        }
        for city, details in cities.items():
            with self.subTest(city=city), self.assertRaises(CoordinatesNotAvailableError):
                CoordinatesValidator.get_coordinates(details)


class TestPathValidation(unittest.TestCase):

    def setUp(self):
        self.temp_dir = Path("temp")
        self.temp_dir.mkdir()
        self._create_sample_files()

    def _create_sample_files(self):
        for model in CLIMATE_MODELS:
            for scenario_name in SSP_SCENARIOS:
                new_file = Path(
                    self.temp_dir, INPUT_FILENAME_FORMAT[scenario_name].format(model=model))
                new_file.touch()

    def test_valid_paths_all_models_and_scenarios(self):
        for model in CLIMATE_MODELS:
            for scenario_name in SSP_SCENARIOS:
                with self.subTest(model=model, scenario=scenario_name):
                    expected_path = Path(self.temp_dir, INPUT_FILENAME_FORMAT[
                        scenario_name].format(model=model))
                    self.assertEqual(PathValidator.validate_precipitation_source_path(
                        model, scenario_name, self.temp_dir), expected_path)

    def test_invalid_scenarios(self):
        models = ["Test", "something", "not-a-thing", "gisele-bundchen", "kalman"]
        wrong_scenarios = ["scp3848", "SCP7789", "pré-histórico"]
        for model in models:
            for scenario_name in wrong_scenarios:
                with (
                        self.subTest(model=model, scenario=scenario_name),
                        self.assertRaises(InvalidClimateScenarioError)):
                    PathValidator.validate_precipitation_source_path(
                        model, scenario_name, self.temp_dir)

    def test_no_file_in_resulting_path(self):
        models = ["Test", "something", "not-a-thing", "gisele-bundchen", "kalman"]
        for model in models:
            for scenario_name in SSP_SCENARIOS:
                with (
                        self.subTest(model=model, scenario=scenario_name),
                        self.assertRaises(InvalidSourceFileError)):
                    PathValidator.validate_precipitation_source_path(
                        model, scenario_name, self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)


if __name__ == '__main__':
    unittest.main()
