import unittest

import pandas as pd

from calculator import (
    IndicesCalculator, compute_seasonality_index, find_max_consecutive_run_length)
from tests.stub_precipitation import PrecipitationGenerator


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


if __name__ == '__main__':
    unittest.main()
