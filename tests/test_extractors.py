import json
import logging
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from agents.extractors import (
    BaseCoordinatesExtractor, NetCDFExtractor, RawCoordinatesExtractor,
    StructuredCoordinatesExtractor, logger)
from globals.errors import InvalidSourceFileError, InvalidTargetCoordinatesError
from tests.samples.stub_netCDF4 import SAMPLE_NC_PATH, NetCDFStubGenerator
from tests.test_validators import SAMPLE_CITIES_PATH

LATITUDES = [-34.125 + 0.25*step for step in range(16)]
LONGITUDES = [-74.125 + 0.25*step for step in range(12)]
SAMPLE_RAW_CITIES_PATH = Path(
    Path(__file__).parent, "samples", "sample_raw_city_coordinates.csv")


class TestNetCDFExtractor(unittest.TestCase):

    def setUp(self):
        if not SAMPLE_NC_PATH.is_file():
            NetCDFStubGenerator.create_sample_stub(SAMPLE_NC_PATH)
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


class TestBaseCoordinatesExtractor(unittest.TestCase):

    def test_initialization_with_valid_file(self):
        valid_path = Path(__file__)
        extractor = BaseCoordinatesExtractor(valid_path)
        self.assertEqual(valid_path, extractor.source_path)

    def test_initialization_with_invalid_file(self):
        invalid_path = Path(__file__).parent / "noi.xyz"
        with self.assertRaises(InvalidSourceFileError):
            BaseCoordinatesExtractor(invalid_path)

    def test_get_coordinates_not_implemented(self):
        valid_path = Path(__file__)
        extractor = BaseCoordinatesExtractor(valid_path)
        with self.assertRaises(NotImplementedError):
            extractor.get_coordinates()


class TestStructuredCoordinatesExtractor(unittest.TestCase):

    def setUp(self):
        with open(SAMPLE_CITIES_PATH, encoding="utf-8") as file:
            self.expected_city_coordinates = json.load(file)

    def test_initialization_with_invalid_file(self):
        invalid_path = Path(__file__).parent / "noi.xyz"
        with self.assertRaises(InvalidSourceFileError):
            StructuredCoordinatesExtractor(invalid_path)

    def test_get_coordinates(self):
        extractor = StructuredCoordinatesExtractor(SAMPLE_CITIES_PATH)
        coordinates = extractor.get_coordinates()
        self.assertDictEqual(self.expected_city_coordinates, coordinates)

    def test_log_record_from_get_coordinates(self):
        EXPECTED_LOG_MESSAGE = (
            f"Successfully extracted coordinates from file at "
            f"'{SAMPLE_CITIES_PATH.resolve()}'")
        extractor = StructuredCoordinatesExtractor(SAMPLE_CITIES_PATH)

        with self.assertLogs(logger, level=logging.INFO) as log_context:
            extractor.get_coordinates()
            self.assertIn(EXPECTED_LOG_MESSAGE, log_context.output[0])


class TestRawCoordinatesExtractor(unittest.TestCase):

    def test_initialization_with_invalid_file(self):
        invalid_path = Path(__file__).parent / "noi.xyz"
        with self.assertRaises(InvalidSourceFileError):
            RawCoordinatesExtractor(invalid_path)

    def test_get_coordinates(self):
        EXPECTED_RESULT = {
            "São Paulo": {
                "ibge_code": 3550308,
                "latitude": -23.533,
                "longitude": -46.64
            },
            "Rio de Janeiro": {
                "ibge_code": 3304557,
                "latitude": -22.913,
                "longitude": -43.2
            },
            "Brasília": {
                "ibge_code": 5300108,
                "latitude": -15.78,
                "longitude": -47.93
            },
            "Fortaleza": {
                "ibge_code": 2304400,
                "latitude": -3.717,
                "longitude": -38.542
            },
            "Salvador": {
                "ibge_code": 2927408,
                "latitude": -12.972,
                "longitude": -38.501
            },
        }
        extractor = RawCoordinatesExtractor(SAMPLE_RAW_CITIES_PATH)
        coordinates = extractor.get_coordinates()
        self.assertDictEqual(EXPECTED_RESULT, coordinates)

    def test_log_record_from_get_coordinates(self):
        EXPECTED_LOG_MESSAGE = (
            f"Successfully extracted coordinates and codes from file at "
            f"'{SAMPLE_RAW_CITIES_PATH.resolve()}'")
        extractor = RawCoordinatesExtractor(SAMPLE_RAW_CITIES_PATH)

        with self.assertLogs(logger, level=logging.INFO) as log_context:
            extractor.get_coordinates()
            self.assertIn(EXPECTED_LOG_MESSAGE, log_context.output[0])


if __name__ == '__main__':
    unittest.main()
