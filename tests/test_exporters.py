import json
import logging
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import time_machine

from agents.exporters import (
    BasePrecipitationExporter, CSVExporter, JSONCoordinatesExporter, NetunoExporter,
    ParquetExporter, logger)
from tests.samples.stub_raw_coordinates import SAMPLE_RAW_COORDINATES


ZONE_INFO = ZoneInfo("America/Sao_Paulo")


@time_machine.travel(datetime(2020, 11, 5, 23, 45, tzinfo=ZONE_INFO))
class TestBaseExporter(unittest.TestCase):

    def test_get_base_path(self):
        exporter = BasePrecipitationExporter(Path(__file__).parent)
        base_path = exporter._get_base_path()
        self.assertEqual(base_path, "2020-11-05T23-45-")


@time_machine.travel(datetime(2020, 11, 5, 23, 45, tzinfo=ZONE_INFO))
class TestParquetExporter(unittest.TestCase):

    SAMPLE_DATAFRAME = pd.DataFrame(
        [(0, 20), (1, 30), (2, 15)], columns=["date", "precipitation"])

    def setUp(self):
        self.exporter = ParquetExporter(Path(__file__).parent)
        self.expected_file_path = Path(
            self.exporter.parent_output_dir, "2020-11-05T23-45-consolidated.parquet")
        self.expected_file_path.unlink(missing_ok=True)

    def test_get_base_file_name(self):
        file_name = self.exporter._get_base_file_name()
        self.assertEqual(file_name, "2020-11-05T23-45-consolidated.parquet")

    def test_generate_parquet(self):
        self.exporter.generate_parquet(self.SAMPLE_DATAFRAME)

        self.assertTrue(self.expected_file_path.is_file())

        self.expected_file_path.unlink()

    def test_log_output_from_generated_parquet(self):
        EXPECTED_LOG_MESSAGE = (
            f"Successfully exported dataframe to '{self.expected_file_path.resolve()}'")
        with self.assertLogs(logger, level=logging.INFO) as log_context:
            self.exporter.generate_parquet(self.SAMPLE_DATAFRAME)
            self.assertIn(EXPECTED_LOG_MESSAGE, log_context.output[0])

        self.expected_file_path.unlink()

    def test_generated_parquet_content(self):
        self.exporter.generate_parquet(self.SAMPLE_DATAFRAME)

        content = pd.read_parquet(self.expected_file_path)

        for expected_values, actual_values in zip(
                self.SAMPLE_DATAFRAME.itertuples(), content.itertuples()):
            with self.subTest(index=expected_values.Index):
                self.assertEqual(expected_values, actual_values)

        self.expected_file_path.unlink()


@time_machine.travel(datetime(2020, 11, 5, 23, 45, tzinfo=ZONE_INFO))
class TestCSVExporter(unittest.TestCase):

    SAMPLE_DATASERIES: np.ndarray[tuple[datetime, int]] = np.array([
        (datetime(2015, 1, 1), 20), (datetime(2015, 1, 2), 30), (datetime(2015, 1, 3), 15)])

    def setUp(self):
        self.exporter = CSVExporter(Path(__file__).parent)
        self.expected_file_path = Path(
            self.exporter.output_dir, "Auridon_EC-NIRN3_SSP245_2015_2100.csv")
        for file in self.exporter.output_dir.iterdir():
            file.unlink()

    def test_get_base_directory_name(self):
        directory_name = self.exporter._get_base_directory_name()
        self.assertEqual(directory_name, "2020-11-05T23-45-output")

    def test_get_file_path(self):
        file_name = self.exporter._get_file_path("Auridon", "EC-NIRN3", "SSP245_2015_2100")
        self.assertEqual(file_name, self.expected_file_path)

    def test_generate_csv(self):
        self.exporter.generate_csv(
            self.SAMPLE_DATASERIES, "Auridon", "EC-NIRN3", "SSP245_2015_2100")

        self.assertTrue(self.expected_file_path.is_file())

        self.expected_file_path.unlink()

    def test_log_output_from_generated_csv(self):
        EXPECTED_LOG_MESSAGE = (
            f"Successfully exported precipitation data series to "
            f"'{self.expected_file_path.resolve()}'")
        with self.assertLogs(logger, level=logging.INFO) as log_context:
            self.exporter.generate_csv(
                self.SAMPLE_DATASERIES, "Auridon", "EC-NIRN3", "SSP245_2015_2100")
            self.assertIn(EXPECTED_LOG_MESSAGE, log_context.output[0])

        self.expected_file_path.unlink()

    def test_generated_csv_content(self):
        self.exporter.generate_csv(
            self.SAMPLE_DATASERIES, "Auridon", "EC-NIRN3", "SSP245_2015_2100")

        content = pd.read_csv(self.expected_file_path, date_format="%y-%m-%d")
        content["date"] = pd.to_datetime(content["date"], format="%Y-%m-%d")

        for expected_values, actual_values in zip(
                self.SAMPLE_DATASERIES, content.itertuples()):
            with self.subTest(index=actual_values.Index):
                self.assertEqual(expected_values[0], actual_values.date)
                self.assertEqual(expected_values[1], actual_values.precipitation)

        self.expected_file_path.unlink()

    def tearDown(self):
        self.exporter.output_dir.rmdir()


class TestNetunoExporter(unittest.TestCase):

    SAMPLE_DATASERIES: np.ndarray[tuple[datetime, int]] = np.array([
        (datetime(2015, 1, 1), 20), (datetime(2015, 1, 2), 30), (datetime(2015, 1, 3), 15)])

    def setUp(self):
        self.exporter = NetunoExporter(Path(__file__).parent)
        self.expected_file_path = Path(
            self.exporter.output_dir, "(Netuno)Auridon_EC-NIRN3_SSP245_2015_2100.csv")
        for file in self.exporter.output_dir.iterdir():
            file.unlink()

    def test_get_file_path(self):
        file_name = self.exporter._get_file_path("Auridon", "EC-NIRN3", "SSP245_2015_2100")
        self.assertEqual(file_name, self.expected_file_path)

    def test_generate_csv(self):
        self.exporter.generate_csv(
            self.SAMPLE_DATASERIES, "Auridon", "EC-NIRN3", "SSP245_2015_2100")

        self.assertTrue(self.expected_file_path.is_file())

        self.expected_file_path.unlink()

    def test_generated_csv_content(self):
        self.exporter.generate_csv(
            self.SAMPLE_DATASERIES, "Auridon", "EC-NIRN3", "SSP245_2015_2100")

        content = pd.read_csv(self.expected_file_path, header=None, names=["precipitation"])

        self.assertIn("precipitation", content.columns)
        self.assertNotIn("date", content.columns)

        for expected_values, actual_values in zip(
                self.SAMPLE_DATASERIES, content.itertuples()):
            with self.subTest(index=actual_values.Index):
                self.assertEqual(expected_values[1], actual_values[1])

        self.expected_file_path.unlink()

    def tearDown(self):
        self.exporter.output_dir.rmdir()


class TestJSONCoordinatesExporter(unittest.TestCase):

    def setUp(self):
        self.output_path = Path(Path(__file__).parent, "samples", "coordinates.json")

    def test_generate_json(self):
        JSONCoordinatesExporter.generate_json(SAMPLE_RAW_COORDINATES, self.output_path)
        with open(self.output_path) as file:
            content = json.load(file)
        self.assertDictEqual(SAMPLE_RAW_COORDINATES, content)

    def test_log_record_from_generate_json(self):
        EXPECTED_LOG_MESSAGE = (
            f"Exported new coordinates file to '{self.output_path.resolve()}'")
        with self.assertLogs(logger, level=logging.INFO) as log_context:
            JSONCoordinatesExporter.generate_json(SAMPLE_RAW_COORDINATES, self.output_path)
            self.assertIn(EXPECTED_LOG_MESSAGE, log_context.output[0])

    def tearDown(self):
        self.output_path.unlink()


if __name__ == '__main__':
    unittest.main()
