import unittest
from datetime import datetime, timedelta

from errors import InvalidTargetCoordinatesError
from extractor import NetCDFExtractor
from tests.stub_netCDF4 import SAMPLE_NC_PATH, NetCDFStubGenerator
from validators import PrecipitationValidator


class TestNetCDFExtractor(unittest.TestCase):

    def setUp(self):
        self.extractor = NetCDFExtractor(SAMPLE_NC_PATH, PrecipitationValidator)

    def test_get_dataset_variables(self):
        extractor = NetCDFExtractor.__new__(NetCDFExtractor)
        extractor.validator = PrecipitationValidator
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
        self.assertTupleEqual(
            self.extractor._find_coordinates_indices(-34.125, -74.125), (0, 0))

    def test_find_coordinates_indices_absent(self):
        with self.assertRaises(InvalidTargetCoordinatesError):
            self.extractor._find_coordinates_indices(-34.005, -74.977)


if __name__ == '__main__':
    unittest.main()
