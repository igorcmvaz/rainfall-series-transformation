import unittest
from datetime import datetime, timedelta

import numpy as np

from errors import InvalidTargetCoordinatesError
from extractor import NetCDFExtractor
from tests.stub_netCDF4 import SAMPLE_NC_PATH, NetCDFStubGenerator

LATITUDES = [-34.125 + 0.25*step for step in range(16)]
LONGITUDES = [-74.125 + 0.25*step for step in range(12)]


class TestNetCDFExtractor(unittest.TestCase):

    def setUp(self):
        self.extractor = NetCDFExtractor(SAMPLE_NC_PATH)

    def test_get_dataset_variables(self):
        extractor = NetCDFExtractor.__new__(NetCDFExtractor)
        extractor.variables = {}
        expected_variables = NetCDFStubGenerator.create_sample_variables()

        extractor._get_dataset_variables(SAMPLE_NC_PATH)

        self.assertEqual(extractor.variables.keys(), expected_variables.keys())
        for name, variable in expected_variables.items():
            with self.subTest(f"Checking variable '{name}'"):
                self.assertEqual(extractor.variables[name].units, variable.units)
                self.assertListEqual(
                    extractor.variables[name].values.tolist(), variable.values.tolist())

    def test_parse_reference_date_default_format(self):
        REFERENCE_DATETIME = datetime(2020, 1, 1)
        self.assertEqual(self.extractor._parse_reference_date(), REFERENCE_DATETIME)

    def test_parse_reference_date_custom_format(self):
        ORIGINAL_UNITS = self.extractor.variables["time"].units
        REFERENCE_DATETIME = datetime(2024, 5, 2)
        self.extractor.variables["time"].units = "days since 02/05/24"

        self.assertEqual(
            self.extractor._parse_reference_date("%d/%m/%y"), REFERENCE_DATETIME)

        self.extractor.variables["time"].units = ORIGINAL_UNITS

    def test_find_coordinates_indices_present(self):
        for latitude_index, target_latitude in enumerate(LATITUDES):
            for longitude_index, target_longitude in enumerate(LONGITUDES):
                with self.subTest(coordinates=(target_latitude, target_longitude)):
                    self.assertTupleEqual(
                        self.extractor._find_coordinates_indices(
                            target_latitude, target_longitude),
                        (latitude_index, longitude_index))

    def test_find_coordinates_indices_absent(self):
        with self.assertRaises(InvalidTargetCoordinatesError):
            self.extractor._find_coordinates_indices(-34.005, -74.977)

    def test_relative_to_absolute_date(self):
        expected_values = NetCDFStubGenerator.create_sample_variables()

        for latitude_index in range(16):
            for longitude_index in range(12):
                with self.subTest(indices=(latitude_index, longitude_index)):

                    expected_output = np.array([
                        (
                            datetime(2020, 1, 1) + timedelta(days=float(t)),
                            expected_values["pr"].values[t, latitude_index, longitude_index]
                        ) for t in range(100)])

                    result = self.extractor._relative_to_absolute_date(
                        self.extractor.variables[
                            "pr"].values[:, latitude_index, longitude_index])

                    self.assertListEqual(expected_output.tolist(), result.tolist())

    def test_extract_sample_precipitation(self):
        expected_values = NetCDFStubGenerator.create_sample_variables()

        for latitude_index, target_latitude in enumerate(LATITUDES):
            for longitude_index, target_longitude in enumerate(LONGITUDES):
                with self.subTest(coordinates=(target_latitude, target_longitude)):
                    expected_output = np.array([
                        (
                            datetime(2020, 1, 1) + timedelta(days=float(t)),
                            expected_values["pr"].values[t, latitude_index, longitude_index]
                        ) for t in range(100)])

                    result = self.extractor.extract_precipitation(
                        target_latitude, target_longitude)
                    self.assertListEqual(expected_output.tolist(), result.tolist())


if __name__ == '__main__':
    unittest.main()
