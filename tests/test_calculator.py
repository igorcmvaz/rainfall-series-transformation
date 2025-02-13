import unittest
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from calculator import (
    IndicesCalculator, compute_seasonality_index, find_max_consecutive_run_length)
from tests.stub_netCDF4 import SAMPLE_NC_PATH, NetCDFStubGenerator


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


class TestSeasonalityIndexComputation(unittest.TestCase):

    def test_(self):
        pass


class TestIndicesCalculator(unittest.TestCase):

    def setUp(self):
        pass


if __name__ == '__main__':
    unittest.main()
