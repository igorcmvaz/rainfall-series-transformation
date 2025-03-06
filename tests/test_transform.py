import json
import logging
import shutil
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import time_machine

from agents.exporters import CSVExporter, NetunoExporter
from agents.validators import CommandLineArgsValidator
from globals.constants import CLIMATE_MODELS, SSP_SCENARIOS
from transform import (
    filter_by_suffix, find_smallest_file, get_csv_exporter, logger, main,
    process_coordinates_files, setup_logger)

ZONE_INFO = ZoneInfo("America/Sao_Paulo")
PATCH_STRINGS = {
    "Consolidator": {
        "__init__": "agents.consolidator.Consolidator.__init__",
        "consolidate_indices": (
            "agents.consolidator.Consolidator.consolidate_indices_dataset"),
        "all_precipitation": (
            "agents.consolidator.Consolidator.generate_all_precipitation_series"),
        "clear": "agents.consolidator.Consolidator.clear_temp_files"
    },
    "CoordinatesFinder": {
        "__init__": "agents.calculator.CoordinatesFinder.__init__",
        "find": "agents.calculator.CoordinatesFinder.find_matching_coordinates"
    },
    "get_coordinates": "agents.extractors.StructuredCoordinatesExtractor.get_coordinates",
    "generate_parquet": "agents.exporters.ParquetExporter.generate_parquet",
    "get_csv_exporter": "transform.get_csv_exporter",
    "process_coordinates": "transform.process_coordinates_files",
}


@time_machine.travel(datetime(2020, 11, 5, 23, 45, tzinfo=ZONE_INFO))
class TestLoggerSetup(unittest.TestCase):

    def reset_logger(self, logger: logging.Logger):
        logging.shutdown()
        for handler in logger.handlers:
            logger.removeHandler(handler)
            handler.close()

    def test_logger_setup_no_quiet(self):
        self.reset_logger(logger)

        setup_logger(logger, 0, False)
        self.assertEqual(logger.level, logging.INFO)

    def test_logger_setup_quiet_once(self):
        self.reset_logger(logger)

        setup_logger(logger, 1, False)
        self.assertEqual(logger.level, logging.WARNING)

    def test_logger_setup_quiet_twice(self):
        self.reset_logger(logger)

        setup_logger(logger, 2, False)
        self.assertEqual(logger.level, logging.ERROR)

    def test_logger_setup_quiet_many(self):
        self.reset_logger(logger)

        setup_logger(logger, 168, False)
        self.assertEqual(logger.level, logging.ERROR)

    def test_logger_setup_verbose(self):
        self.reset_logger(logger)

        setup_logger(logger, 0, True)
        self.assertEqual(logger.level, logging.DEBUG)

    def test_logger_setup_quiet_and_verbose(self):
        self.reset_logger(logger)

        setup_logger(logger, 2, True)
        self.assertEqual(logger.level, logging.DEBUG)


class TestDirectoryFunctions(unittest.TestCase):

    BASE_PATH = Path(__file__).parent / "xyz"

    def setUp(self):
        self.test_files: list[Path] = []
        self.BASE_PATH.mkdir(exist_ok=True)
        self._create_test_files()
        self._add_data_to_files([file for file in self.test_files if "01" in file.name])

    def _create_test_files(self):
        suffixes = (".nc", ".json")
        for suffix in suffixes:
            for index in range(2):
                new_file = Path(self.BASE_PATH, f"test-{index:02}{suffix}")
                new_file.touch()
                self.test_files.append(new_file)

    def _add_data_to_files(self, files: list[Path]):
        SAMPLE_DATA = {
            f"data-{index:02}": [value for value in range(1000)] for index in range(100)
        }
        for file_path in files:
            with open(file_path, "w") as file:
                json.dump(SAMPLE_DATA, file, indent=2, sort_keys=True)

    def test_filter_files_by_suffix_default(self):
        filtered_files = sorted(list(filter_by_suffix(self.BASE_PATH)))
        self.assertListEqual(filtered_files, self.test_files[:2])

    def test_filter_files_by_suffix_custom(self):
        filtered_files = sorted(list(filter_by_suffix(self.BASE_PATH, ".json")))
        self.assertListEqual(filtered_files, self.test_files[2:])

    def test_find_smallest_file(self):
        EXPECTED_FILE = Path(self.BASE_PATH, "test-00.nc")
        smallest_file = find_smallest_file(self.BASE_PATH)
        self.assertEqual(EXPECTED_FILE, smallest_file)

    def tearDown(self):
        shutil.rmtree(self.BASE_PATH)


class TestProcessCoordinatesFunction(unittest.TestCase):

    BASE_PATH = Path(__file__).parent / "xyz"
    SAMPLE_COORDINATES = {
        "Florianópolis": {
            "nearest": {
                "lat": 22.55,
                "lon": 44.99
                },
            "target": {
                "lat": 22.5055,
                "lon": 44.9877
            }
        }
    }

    def setUp(self):
        self.BASE_PATH.mkdir(exist_ok=True)
        self.reference_file = Path(self.BASE_PATH, "test.nc")
        self.reference_file.touch()
        self.coordinates_file = Path(self.BASE_PATH, "coordinates.csv")
        self.expected_output_path = self.coordinates_file.with_suffix(".json")

    def test_process_coordinates_files(self):
        with (
                patch(PATCH_STRINGS["CoordinatesFinder"]["__init__"]) as init_mock,
                patch(PATCH_STRINGS["CoordinatesFinder"]["find"]) as finder_mock):
            init_mock.return_value = None
            finder_mock.return_value = self.SAMPLE_COORDINATES
            process_coordinates_files(self.coordinates_file, self.BASE_PATH)

        self.assertTrue(self.expected_output_path.is_file())
        with open(self.expected_output_path) as file:
            content = json.load(file)
        self.assertDictEqual(self.SAMPLE_COORDINATES, content)

    def test_log_output_from_process_coordinates_files(self):
        EXPECTED_LOG_MESSAGE = (
            f"Coordinates at '{self.coordinates_file.name}' will be validated against "
            f"reference file '{self.reference_file.name}'")
        with (
                self.assertLogs(logger, level=logging.INFO) as log_context,
                patch(PATCH_STRINGS["CoordinatesFinder"]["__init__"]) as init_mock,
                patch(PATCH_STRINGS["CoordinatesFinder"]["find"]) as finder_mock):
            init_mock.return_value = None
            finder_mock.return_value = self.SAMPLE_COORDINATES
            process_coordinates_files(self.coordinates_file, self.BASE_PATH)
            self.assertIn(EXPECTED_LOG_MESSAGE, log_context.output[0])

    def tearDown(self):
        shutil.rmtree(self.BASE_PATH)


class TestGetCSVExporter(unittest.TestCase):

    def setUp(self):
        self.args = CommandLineArgsValidator()
        self.args.input_path = Path(__file__).parent

    def test_get_csv_exporter_netuno_required(self):
        self.args.netuno_required = True
        self.args.csv_required = False

        result = get_csv_exporter(self.args)

        self.assertIsInstance(result, NetunoExporter)

        result.output_dir.rmdir()

    def test_get_csv_exporter_only_csv_required(self):
        self.args.netuno_required = False
        self.args.csv_required = True

        result = get_csv_exporter(self.args)

        self.assertIsInstance(result, CSVExporter)

        result.output_dir.rmdir()

    def test_get_csv_exporter_both_required(self):
        self.args.netuno_required = True
        self.args.csv_required = True

        result = get_csv_exporter(self.args)

        self.assertIsInstance(result, NetunoExporter)

        result.output_dir.rmdir()

    def test_get_csv_exporter_none_required(self):
        self.args.netuno_required = False
        self.args.csv_required = False

        result = get_csv_exporter(self.args)

        self.assertIsNone(result)


class TestMainOperation(unittest.TestCase):

    def setUp(self):
        self.args = CommandLineArgsValidator()
        self.args.coordinates_path = Path(__file__)
        self.args.input_path = Path(__file__).parent
        self.args.csv_required = True
        self.args.netuno_required = False
        self.args.recovery_required = False

    def test_only_process_coordinates(self):
        self.args.only_process_coordinates = True
        with patch(PATCH_STRINGS["process_coordinates"]) as process_mock:
            main(self.args)
            process_mock.assert_called_once_with(Path(__file__), Path(__file__).parent)

    def test_parquet_required(self):
        self.args.only_process_coordinates = False
        self.args.parquet_required = True
        expected_exporter = CSVExporter(self.args.input_path.parent)
        with (
                patch(PATCH_STRINGS["Consolidator"]["__init__"]) as consolidator_mock,
                patch(PATCH_STRINGS["Consolidator"]["consolidate_indices"]) as indices_mock,
                patch(PATCH_STRINGS["get_coordinates"]) as coordinates_mock,
                patch(PATCH_STRINGS["get_csv_exporter"]) as get_csv_mock,
                patch(PATCH_STRINGS["generate_parquet"]) as parquet_mock):
            consolidator_mock.return_value = None
            indices_mock.return_value = "INDICES"
            coordinates_mock.return_value = "COORDINATES"
            get_csv_mock.return_value = expected_exporter

            main(self.args)

            coordinates_mock.assert_called_once()
            consolidator_mock.assert_called_once_with(
                "COORDINATES",
                SSP_SCENARIOS,
                CLIMATE_MODELS,
                Path(__file__).parent,
                False,
                expected_exporter)
            parquet_mock.assert_called_once_with("INDICES")

        expected_exporter.output_dir.rmdir()

    def test_parquet_not_required(self):
        self.args.only_process_coordinates = False
        self.args.parquet_required = False
        with (
                patch(PATCH_STRINGS["Consolidator"]["__init__"]) as consolidator_mock,
                patch(PATCH_STRINGS["Consolidator"]["all_precipitation"]) as series_mock,
                patch(PATCH_STRINGS["Consolidator"]["consolidate_indices"]) as indices_mock,
                patch(PATCH_STRINGS["get_csv_exporter"]),
                patch(PATCH_STRINGS["get_coordinates"]),
                patch(PATCH_STRINGS["generate_parquet"])):
            consolidator_mock.return_value = None

            main(self.args)

            indices_mock.assert_not_called()
            series_mock.assert_called_once_with()

    def test_recovery_required_no_temp(self):
        self.args.only_process_coordinates = False
        self.args.parquet_required = False
        self.args.recovery_required = True
        self.args.keep_temp_files = False
        with (
                patch(PATCH_STRINGS["Consolidator"]["__init__"]) as consolidator_mock,
                patch(PATCH_STRINGS["Consolidator"]["clear"]) as clear_mock,
                patch(PATCH_STRINGS["Consolidator"]["all_precipitation"]),
                patch(PATCH_STRINGS["get_csv_exporter"]),
                patch(PATCH_STRINGS["get_coordinates"]),
                patch(PATCH_STRINGS["generate_parquet"])):
            consolidator_mock.return_value = None

            main(self.args)

            clear_mock.assert_called_once_with()

    def test_recovery_required_keep_temp(self):
        self.args.only_process_coordinates = False
        self.args.parquet_required = False
        self.args.recovery_required = True
        self.args.keep_temp_files = True
        with (
                patch(PATCH_STRINGS["Consolidator"]["__init__"]) as consolidator_mock,
                patch(PATCH_STRINGS["Consolidator"]["clear"]) as clear_mock,
                patch(PATCH_STRINGS["Consolidator"]["all_precipitation"]),
                patch(PATCH_STRINGS["get_csv_exporter"]),
                patch(PATCH_STRINGS["get_coordinates"]),
                patch(PATCH_STRINGS["generate_parquet"])):
            consolidator_mock.return_value = None

            main(self.args)

            clear_mock.assert_not_called()

    def test_no_recovery(self):
        self.args.only_process_coordinates = False
        self.args.parquet_required = False
        self.args.recovery_required = False
        self.args.keep_temp_files = True
        with (
                patch(PATCH_STRINGS["Consolidator"]["__init__"]) as consolidator_mock,
                patch(PATCH_STRINGS["Consolidator"]["clear"]) as clear_mock,
                patch(PATCH_STRINGS["Consolidator"]["all_precipitation"]),
                patch(PATCH_STRINGS["get_csv_exporter"]),
                patch(PATCH_STRINGS["get_coordinates"]),
                patch(PATCH_STRINGS["generate_parquet"])):
            consolidator_mock.return_value = None

            main(self.args)

            clear_mock.assert_not_called()


if __name__ == '__main__':
    unittest.main()
