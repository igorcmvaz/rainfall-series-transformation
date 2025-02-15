import json
from pathlib import Path
import unittest

import numpy as np
from datetime import datetime, timedelta

from errors import CoordinatesNotAvailableError
from validators import PrecipitationValidator, CoordinatesValidator, PathValidator

SAMPLE_CITIES_PATH = Path(__file__).parent / "sample_city_coordinates.json"


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


if __name__ == '__main__':
    unittest.main()
