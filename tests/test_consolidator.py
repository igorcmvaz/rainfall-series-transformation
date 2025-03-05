import json
import logging
import pickle
import shutil
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd

from agents.calculator import estimate_combinations
from agents.consolidator import Consolidator, logger
from agents.exporters import CSVExporter
from globals.constants import (
    CLIMATE_MODELS, INPUT_FILENAME_FORMAT, SSP_SCENARIOS, RECOVERY_FILENAME_FORMAT)
from tests.samples.stub_netCDF4 import NetCDFStubGenerator
from tests.samples.stub_precipitation import SAMPLE_JSON_PATH
from tests.test_validators import SAMPLE_CITIES_PATH


class TestConsolidatorInternalFunctions(unittest.TestCase):

    SAMPLE_SOURCE_DIR = Path(__file__).parent / "test_temp"
    LONGITUDES = [-74.125 + 0.25*step for step in range(6)]
    LATITUDES = [-34.125 + 0.25*step for step in range(6)]

    def setUp(self):
        with open(SAMPLE_CITIES_PATH) as file:
            self.cities = json.load(file)
        self.models = CLIMATE_MODELS[:5]
        self.consolidator = Consolidator(
            self.cities, SSP_SCENARIOS, self.models, self.SAMPLE_SOURCE_DIR)
        self.expected_total = estimate_combinations(
            self.models, SSP_SCENARIOS, self.cities)

        self._prepare_sample_precipitation()

        self.SAMPLE_SOURCE_DIR.mkdir(exist_ok=True)
        self._create_sample_files()

    def _prepare_sample_precipitation(self):
        sample_precipitation = NetCDFStubGenerator.create_sample_variables()
        updated_values = []
        for latitude_index in range(6):
            for longitude_index in range(6):
                updated_values.append([(
                    datetime(2020, 1, 1) + timedelta(days=float(t)),
                    sample_precipitation["pr"].values[t, latitude_index, longitude_index]
                    ) for t in range(100)
                ])
        self.precipitation = np.array(updated_values)

    def _create_sample_files(self):
        for model in self.models:
            for scenario_name in SSP_SCENARIOS:
                new_file = Path(
                    self.SAMPLE_SOURCE_DIR,
                    INPUT_FILENAME_FORMAT[scenario_name].format(model=model))
                new_file.touch(exist_ok=True)

    def test_state_after_count_error(self):
        ORIGINAL_PROCESSED_COUNT = self.consolidator.state["processed"]
        ORIGINAL_ERROR_COUNT = self.consolidator.state["error"]
        self.consolidator._count_error(test="test")

        for count_type, original_value in zip(
                ["processed", "error"],
                [ORIGINAL_PROCESSED_COUNT, ORIGINAL_ERROR_COUNT]):
            with self.subTest(count=count_type):
                self.assertEqual(original_value + 1, self.consolidator.state[count_type])

    def test_parameters_in_logs_after_count_error(self):
        ORIGINAL_PROCESSED_COUNT = self.consolidator.state["processed"]
        EXPECTED_LOG_MESSAGE = (
            f"[{ORIGINAL_PROCESSED_COUNT+1}/{self.expected_total}] Error during processing "
            f"for test=foo, bar=zii, el=psy")

        with self.assertLogs(logger, level=logging.ERROR) as log_context:
            self.consolidator._count_error(test="foo", bar="zii", el="psy")
            self.assertIn(EXPECTED_LOG_MESSAGE, log_context.output[0])

    def test_state_after_count_processed(self):
        ORIGINAL_PROCESSED_COUNT = self.consolidator.state["processed"]
        self.consolidator._count_processed(test="test")

        self.assertEqual(ORIGINAL_PROCESSED_COUNT + 1, self.consolidator.state["processed"])

    def test_parameters_in_logs_after_count_processed(self):
        ORIGINAL_PROCESSED_COUNT = self.consolidator.state["processed"]
        EXPECTED_LOG_MESSAGE = (
            f"[{ORIGINAL_PROCESSED_COUNT+1}/{self.expected_total}] Completed processing for"
            f" test=foo, bar=2, el=None")

        with self.assertLogs(logger, level=logging.INFO) as log_context:
            self.consolidator._count_processed(test="foo", bar=2, el=None)
            self.assertIn(EXPECTED_LOG_MESSAGE, log_context.output[0])

    def test_setting_final_state(self):
        ORIGINAL_STATE = self.consolidator.state.copy()
        self.consolidator.state["total"] = 100
        self.consolidator.state["processed"] = 90
        self.consolidator.state["error"] = 3

        self.consolidator._set_final_state()

        for metric, expected_value in zip(
                ["skipped", "success", "process_rate", "success_rate"],
                [10, 87, 90, 96.6666667]):
            with self.subTest(metric=metric):
                self.assertAlmostEqual(expected_value, self.consolidator.state[metric])

        self.consolidator.state = ORIGINAL_STATE

    def test_log_output_from_final_state(self):
        ORIGINAL_STATE = self.consolidator.state.copy()
        self.consolidator.state["total"] = 100
        self.consolidator.state["processed"] = 90
        self.consolidator.state["error"] = 3

        EXPECTED_LOG_MESSAGE = (
            "Processed 90 combinations, from a total of 100, skipped 10 item(s), "
            "encountered 3 error(s). Success rate: 96.67%. Effective processed rate: 90.00%"
            )

        with self.assertLogs(logger, level=logging.INFO) as log_context:
            self.consolidator._set_final_state()
            self.assertIn(EXPECTED_LOG_MESSAGE, log_context.output[0])

        self.consolidator.state = ORIGINAL_STATE

    def test_column_names_from_insert_metadata(self):
        dataframe = pd.DataFrame()

        self.consolidator._insert_metadata(dataframe, test=[1, 2, 3], foo=["a", "b", "c"])

        for column, expected_column in zip(dataframe.columns, ["Test", "Foo"]):
            with self.subTest(column=column):
                self.assertEqual(column, expected_column)

    def test_values_from_insert_metadata(self):
        dataframe = pd.DataFrame()

        self.consolidator._insert_metadata(dataframe, test=[1, 2, 3], foo=["a", "b", "c"])

        for value, expected_value in zip(
                dataframe.itertuples(),
                [(1, "a"), (2, "b"), (3, "c")]):
            with self.subTest(test=value[1], foo=value[2]):
                self.assertEqual(value[1], expected_value[0])
                self.assertEqual(value[2], expected_value[1])

    def test_evaluate_csv_generation(self):
        ORIGINAL_CSV_GENERATOR = self.consolidator.csv_generator
        self.consolidator.csv_generator = CSVExporter(Path(__file__).parent)

        with patch("agents.exporters.CSVExporter.generate_csv") as csv_mock:
            DATA_SERIES = [(1, 3.14), (2, 1.16)]
            METADATA = {"city": "Auridon", "model": "ACCESS", "scenario": "SSP245"}
            self.consolidator._evaluate_csv_generation(DATA_SERIES, METADATA)
            csv_mock.assert_called_once_with(
                DATA_SERIES, METADATA["city"], METADATA["model"], METADATA["scenario"])

        self.consolidator.csv_generator.output_dir.rmdir()
        self.consolidator.csv_generator = ORIGINAL_CSV_GENERATOR

    def test_evaluate_no_csv_generation(self):
        ORIGINAL_CSV_GENERATOR = self.consolidator.csv_generator
        self.consolidator.csv_generator = None

        with patch("agents.exporters.CSVExporter.generate_csv") as csv_mock:
            DATA_SERIES = [(1, 3.14), (2, 1.16)]
            METADATA = {"city": "Auridon", "model": "ACCESS", "scenario": "SSP245"}
            self.consolidator._evaluate_csv_generation(DATA_SERIES, METADATA)
            csv_mock.assert_not_called()

        self.consolidator.csv_generator = ORIGINAL_CSV_GENERATOR

    def tearDown(self):
        shutil.rmtree(self.SAMPLE_SOURCE_DIR)


class TestConsolidatorGeneration(unittest.TestCase):

    def setUp(self):
        with open(SAMPLE_JSON_PATH) as file:
            content = json.load(file)

        self._prepate_data_attributes(content)
        self._validate_sample_file(content)
        self._validate_empty_file()

    def _prepate_data_attributes(self, content: dict[str, Any]) -> None:
        self.sample_source_dir = Path(__file__).parent
        self.models = [content["model"]]
        self.scenarios = {
            key: value for key, value in SSP_SCENARIOS.items() if key == content["scenario"]
        }
        self.coordinates = (content["latitude"], content["longitude"])
        self.city_name = content["city"]
        self.cities = {
            content["city"]: {
                "nearest": {
                    "lat": -27.625,
                    "lon": -48.875
                },
                "target": {
                    "lat": -27.5954,
                    "lon": -48.548
                }
            }
        }
        self.expected_indices = {
            int(year): content["climate_indices"][year]
            for year in content["climate_indices"]
        }

    def _validate_sample_file(self, content: dict[str, Any]) -> None:
        sample_nc_path = Path(
            self.sample_source_dir,
            INPUT_FILENAME_FORMAT[content["scenario"]].format(model=content["model"]))
        if not sample_nc_path.is_file():
            self.sample_source_dir = NetCDFStubGenerator.from_sample_json(
                SAMPLE_JSON_PATH).parent

    def _validate_empty_file(self) -> None:
        self.empty_scenario = {"SSP245": SSP_SCENARIOS["SSP245"]}
        self.empty_model = "TaiESM1"
        self.empty_coordinates = (-34.125, -74.125)
        self.empty_cities = {
            "Vulkhel Guard": {
                "nearest": {
                    "lat": self.empty_coordinates[0],
                    "lon": self.empty_coordinates[1]
                }
            }
        }
        empty_file_path = Path(
            self.sample_source_dir,
            INPUT_FILENAME_FORMAT["SSP245"].format(model=self.empty_model))
        if not empty_file_path.is_file():
            NetCDFStubGenerator.create_empty_stub(empty_file_path)

    def test_generate_precipitation_dataset(self):
        mock_data_series = np.array([
                (date.strftime("%Y-%m-%d"), 20)
                for date in [datetime(2020, 1, 1) + timedelta(days=t) for t in range(100)]
            ])
        expected_metadata = {
            "city": self.city_name,
            "model": self.models[0],
            "scenario": list(self.scenarios.keys())[0],
            "latitude": self.cities[self.city_name]["nearest"]["lat"],
            "longitude": self.cities[self.city_name]["nearest"]["lon"],
        }
        consolidator = Consolidator(
            self.cities, self.scenarios, self.models, self.sample_source_dir)
        generator = consolidator.generate_precipitation_dataset()

        with patch(
                "agents.extractors.NetCDFExtractor.extract_precipitation"
                ) as precipitation_mock:
            precipitation_mock.return_value = mock_data_series
            result_data, result_meta = next(generator)

        self.assertListEqual(result_data.tolist(), mock_data_series.tolist())
        self.assertDictEqual(result_meta, expected_metadata)

    def test_generate_sample_precipitation_indices(self):
        consolidator = Consolidator(
            self.cities, self.scenarios, self.models, self.sample_source_dir)
        generator = consolidator.generate_precipitation_indices()
        result = next(generator)
        for year, indices in self.expected_indices.items():
            with self.subTest(year=year):
                yearly_result = result[result["year"] == year]
                self.assertAlmostEqual(
                    yearly_result["PRCPTOT"].values[0], indices["PRCPTOT"])
                self.assertAlmostEqual(yearly_result["R95p"].values[0], indices["R95p"])
                self.assertAlmostEqual(yearly_result["RX1day"].values[0], indices["Rx1day"])
                self.assertAlmostEqual(yearly_result["RX5day"].values[0], indices["Rx5day"])
                self.assertAlmostEqual(yearly_result["SDII"].values[0], indices["SDII"])
                self.assertAlmostEqual(yearly_result["R20mm"].values[0], indices["R20mm"])
                self.assertAlmostEqual(yearly_result["CDD"].values[0], indices["CDD"])
                self.assertAlmostEqual(yearly_result["CWD"].values[0], indices["CWD"])
                self.assertAlmostEqual(
                    yearly_result["Seasonality_Index"].values[0],
                    indices["seasonality_index"])

    def test_log_output_from_generate_indices(self):
        EXPECTED_LOG_MESSAGE = (
            f"Completed processing of city '{list(self.cities.keys())[0]}', coordinates "
            f"({', '.join(str(coordinate) for coordinate in self.coordinates)})")
        consolidator = Consolidator(
            self.cities, self.scenarios, self.models, self.sample_source_dir)
        generator = consolidator.generate_precipitation_indices()
        next(generator)
        with (
                self.assertRaises(StopIteration),
                self.assertLogs(logger, level=logging.INFO) as log_context):
            next(generator)
            self.assertIn(EXPECTED_LOG_MESSAGE, log_context.output[0])

    def test_fail_to_generate_indices(self):
        consolidator = Consolidator(
            self.empty_cities,
            self.empty_scenario,
            [self.empty_model],
            self.sample_source_dir)
        EXPECTED_LOG_MESSAGE = (
            f"[1/1] Error during processing. Details: city={list(self.cities.keys())[0]}, "
            f"model={self.empty_model}, scenario={list(self.empty_scenario.keys())[0]}, "
            f"target_coordinates={self.empty_coordinates}")
        generator = consolidator.generate_precipitation_indices()

        with (
                self.assertRaises(StopIteration),
                self.assertLogs(logger, level=logging.ERROR) as log_context):
            next(generator)
            self.assertIn(EXPECTED_LOG_MESSAGE, log_context.output[0])

    def test_consolidate_dataset(self):
        MOCK_DATAFRAME = pd.DataFrame([1, 2, 3])
        EXPECTED_OUTPUT = pd.concat((MOCK_DATAFRAME for _ in range(5)), ignore_index=True)

        consolidator = Consolidator(
            self.cities, self.scenarios, self.models, self.sample_source_dir)
        with patch(
                "agents.consolidator.Consolidator.generate_precipitation_indices"
                ) as generator_mock:
            generator_mock.return_value = (MOCK_DATAFRAME for _ in range(5))
            result = consolidator.consolidate_indices_dataset()
        self.assertTrue((result == EXPECTED_OUTPUT).all().values)

    def test_generate_all_precipitation_series(self):
        consolidator = Consolidator(
            self.cities, self.scenarios, self.models, self.sample_source_dir)
        with patch(
                "agents.consolidator.Consolidator.generate_precipitation_dataset"
                ) as generator_mock:
            generator_mock.return_value = ({1: 2} for _ in range(2))
            consolidator.generate_all_precipitation_series()
            with self.assertRaises(StopIteration):
                next(generator_mock.return_value)

    def test_log_output_from_generate_all_precipitation_series(self):
        EXPECTED_LOG_MESSAGE = "Completed process in "
        consolidator = Consolidator(
            self.cities, self.scenarios, self.models, self.sample_source_dir)
        with patch(
                "agents.consolidator.Consolidator.generate_precipitation_dataset"
                ) as generator_mock:
            generator_mock.return_value = ({1: 2} for _ in range(2))
            with self.assertLogs(logger, level=logging.INFO) as log_context:
                consolidator.generate_all_precipitation_series()
                self.assertIn(EXPECTED_LOG_MESSAGE, log_context.output[0])


class TestConsolidatorRecoveryFunctions(unittest.TestCase):

    SAMPLE_SOURCE_DIR = Path(__file__).parent / "test_temp"
    EXPECTED_TEMP_DIR = Path(SAMPLE_CITIES_PATH.parent.parent, "temp")
    LONGITUDES = [-74.125 + 0.25*step for step in range(6)]
    LATITUDES = [-34.125 + 0.25*step for step in range(6)]

    def setUp(self):
        with open(SAMPLE_CITIES_PATH) as file:
            self.cities = json.load(file)
        self.models = CLIMATE_MODELS[:5]
        self.expected_total = estimate_combinations(self.models, SSP_SCENARIOS, self.cities)
        self.SAMPLE_SOURCE_DIR.mkdir(exist_ok=True)

    def test_create_temp_dir(self):
        EXPECTED_LOG_MESSAGE = (
            f"Created temporary recovery directory at '{self.EXPECTED_TEMP_DIR.resolve()}'")
        shutil.rmtree(self.EXPECTED_TEMP_DIR, ignore_errors=True)
        self.assertFalse(self.EXPECTED_TEMP_DIR.is_dir())

        with self.assertLogs(logger, level=logging.DEBUG) as log_context:
            Consolidator(
                self.cities, SSP_SCENARIOS, self.models, self.SAMPLE_SOURCE_DIR, True)
            self.assertIn(EXPECTED_LOG_MESSAGE, log_context.output[0])
        self.assertTrue(self.EXPECTED_TEMP_DIR.is_dir())

        shutil.rmtree(self.EXPECTED_TEMP_DIR)

    def test_return_existing_temp_dir(self):
        self.EXPECTED_TEMP_DIR.mkdir(exist_ok=True)

        with self.assertNoLogs(logger, level=logging.DEBUG):
            Consolidator(
                self.cities, SSP_SCENARIOS, self.models, self.SAMPLE_SOURCE_DIR, True)
        self.assertTrue(self.EXPECTED_TEMP_DIR.is_dir())

        shutil.rmtree(self.EXPECTED_TEMP_DIR)

    def test_dump_no_recovery_data(self):
        consolidator = Consolidator(
            self.cities, SSP_SCENARIOS, self.models, self.SAMPLE_SOURCE_DIR, True)
        recovery_file_path = Path(
            consolidator.temp_dir,
            RECOVERY_FILENAME_FORMAT.format(model="test", scenario="ssp"))

        with self.assertNoLogs(logger, level=logging.INFO):
            consolidator._dump_recovery_data("test", "ssp", {})
        self.assertFalse(recovery_file_path.is_file())

    def test_content_of_dumped_data(self):
        EXPECTED_RESULT = {"test": "SSP"}
        consolidator = Consolidator(
            self.cities, SSP_SCENARIOS, self.models, self.SAMPLE_SOURCE_DIR, True)
        recovery_file_path = Path(
            consolidator.temp_dir,
            RECOVERY_FILENAME_FORMAT.format(model="test", scenario="ssp"))

        consolidator._dump_recovery_data("test", "ssp", EXPECTED_RESULT)
        with open(recovery_file_path, "rb") as file:
            result = pickle.load(file)

        self.assertDictEqual(result, EXPECTED_RESULT)

    def test_log_output_from_recovery_dump(self):
        consolidator = Consolidator(
            self.cities, SSP_SCENARIOS, self.models, self.SAMPLE_SOURCE_DIR, True)
        recovery_file_path = Path(
            consolidator.temp_dir,
            RECOVERY_FILENAME_FORMAT.format(model="test", scenario="ssp"))
        EXPECTED_LOG_MESSAGE = (
            f"Successfully saved recovery file at '{recovery_file_path.name}' in ")

        with self.assertLogs(logger, level=logging.INFO) as log_context:
            consolidator._dump_recovery_data("test", "ssp", {"test": "SSP"})
            self.assertIn(EXPECTED_LOG_MESSAGE, log_context.output[0])

    def test_validate_recovery_path_success(self):
        consolidator = Consolidator(
            self.cities, SSP_SCENARIOS, self.models, self.SAMPLE_SOURCE_DIR, True)
        recovery_file_path = Path(
            consolidator.temp_dir,
            RECOVERY_FILENAME_FORMAT.format(model="test", scenario="ssp"))
        recovery_file_path.touch(exist_ok=True)

        result = consolidator._validate_recovery_path("test", "ssp")

        self.assertEqual(result, recovery_file_path)

    def test_validate_recovery_path_failure(self):
        consolidator = Consolidator(
            self.cities, SSP_SCENARIOS, self.models, self.SAMPLE_SOURCE_DIR, True)

        result = consolidator._validate_recovery_path("something", "else")

        self.assertIsNone(result)

    def test_log_output_from_failed_recovery_path_validation(self):
        EXPECTED_LOG_MESSAGE = (
            "No recovery file found for model 'something', scenario 'else', proceeding "
            "with normal extraction")
        consolidator = Consolidator(
            self.cities, SSP_SCENARIOS, self.models, self.SAMPLE_SOURCE_DIR, True)

        with self.assertLogs(logger, level=logging.DEBUG) as log_context:
            consolidator._validate_recovery_path("something", "else")
            self.assertIn(EXPECTED_LOG_MESSAGE, log_context.output[0])

    def test_recover_data_from_file(self):
        SAVED_DATA = {"test": "SSP"}
        consolidator = Consolidator(
            self.cities, SSP_SCENARIOS, self.models, self.SAMPLE_SOURCE_DIR, True)
        PATH_TO_RECOVERY_FILE = Path(consolidator.temp_dir, "sample.pickle")
        with open(PATH_TO_RECOVERY_FILE, "wb") as temp_file:
            pickle.dump(SAVED_DATA, temp_file, protocol=pickle.HIGHEST_PROTOCOL)

        result = consolidator._recover_data_from_file(PATH_TO_RECOVERY_FILE)

        self.assertDictEqual(SAVED_DATA, result)

    def test_log_output_from_data_recovery(self):
        consolidator = Consolidator(
            self.cities, SSP_SCENARIOS, self.models, self.SAMPLE_SOURCE_DIR, True)
        PATH_TO_RECOVERY_FILE = Path(consolidator.temp_dir, "sample.pickle")
        EXPECTED_LOG_MESSAGE = "Retrieved data from recovery file 'sample.pickle' in "
        with open(PATH_TO_RECOVERY_FILE, "wb") as temp_file:
            pickle.dump({"test": "SSP"}, temp_file, protocol=pickle.HIGHEST_PROTOCOL)

        with self.assertLogs(logger, level=logging.INFO) as log_context:
            consolidator._recover_data_from_file(PATH_TO_RECOVERY_FILE)
            self.assertIn(EXPECTED_LOG_MESSAGE, log_context.output[0])

    def test_clear_temp_files(self):
        consolidator = Consolidator(
            self.cities, SSP_SCENARIOS, self.models, self.SAMPLE_SOURCE_DIR, True)
        for scenario in ["Histórico", "SSP245", "SSP585"]:
            file_path = Path(consolidator.temp_dir, scenario).with_suffix(".temp")
            file_path.touch()
        consolidator.clear_temp_files()

        self.assertFalse(self.EXPECTED_TEMP_DIR.is_dir())

    def test_generate_dataset_with_invalid_recovered_city(self):
        SAMPLE_RECOVERY_DATA = {
            "Vulkhel Guard": {
                "data": [
                    (datetime(2020, 1, 1), 13.2),
                    (datetime(2020, 1, 2), 2.3),
                    (datetime(2020, 1, 3), 10.0),
                ],
                "metadata": {
                    "city": "Vulkhel Guard",
                    "model": "ACCESS",
                    "scenario": "SSP245",
                    "latitude": 30.3012,
                    "longitude": 57.2920
                }
            }
        }
        consolidator = Consolidator(
            self.cities,
            {"Histórico": "test"},
            self.models[:1],
            self.SAMPLE_SOURCE_DIR,
            True)
        RECOVERY_PATH = Path(consolidator.temp_dir, RECOVERY_FILENAME_FORMAT.format(
                model=self.models[0], scenario="Histórico"))
        RECOVERY_PATH.touch()
        generator = consolidator.generate_precipitation_dataset()

        with (
            self.assertRaises(StopIteration),
            patch("agents.consolidator.Consolidator._count_processed") as count_mock,
            patch("agents.consolidator.Consolidator._set_final_state"),
            patch("agents.consolidator.Consolidator._recover_data_from_file")
                as recovery_mock):
            recovery_mock.return_value = SAMPLE_RECOVERY_DATA
            next(generator)
            count_mock.assert_not_called()

    def test_generate_dataset_with_valid_recovered_city(self):
        SAMPLE_RECOVERY_DATA = {
            "Elsweyr": {
                "data": [
                    (datetime(2020, 1, 1), 13.2),
                    (datetime(2020, 1, 2), 2.3),
                    (datetime(2020, 1, 3), 10.0),
                ],
                "metadata": {
                    "city": "Elsweyr",
                    "model": "ACCESS",
                    "scenario": "SSP245",
                    "latitude": 30.3012,
                    "longitude": 57.2920
                }
            }
        }
        consolidator = Consolidator(
            self.cities, {"SSP245": "test"}, ["ACCESS"], self.SAMPLE_SOURCE_DIR, True)
        RECOVERY_PATH = Path(consolidator.temp_dir, RECOVERY_FILENAME_FORMAT.format(
                model="ACCESS", scenario="SSP245"))
        RECOVERY_PATH.touch()
        generator = consolidator.generate_precipitation_dataset()

        with (
            patch("agents.consolidator.Consolidator._recover_data_from_file")
                as recovery_mock):
            recovery_mock.return_value = SAMPLE_RECOVERY_DATA
            data, metadata = next(generator)
        self.assertListEqual(data, SAMPLE_RECOVERY_DATA["Elsweyr"]["data"])
        self.assertDictEqual(metadata, SAMPLE_RECOVERY_DATA["Elsweyr"]["metadata"])

    def test_generate_dataset_with_recovered_data_exhausting_cities(self):
        with open(SAMPLE_CITIES_PATH) as file:
            cities: dict = json.load(file)
        SAMPLE_RECOVERY_DATA = {
            city_name: {
                "data": [
                    (datetime(2020, 1, 1), 13.2),
                    (datetime(2020, 1, 2), 2.3),
                    (datetime(2020, 1, 3), 10.0),
                ],
                "metadata": {
                    "city": "Elsweyr",
                    "model": "ACCESS",
                    "scenario": "SSP245",
                    "latitude": 30.3012,
                    "longitude": 57.2920
                }
            } for city_name in cities
        }
        consolidator = Consolidator(
            self.cities, {"SSP245": "test"}, ["ACCESS"], self.SAMPLE_SOURCE_DIR, True)
        RECOVERY_PATH = Path(consolidator.temp_dir, RECOVERY_FILENAME_FORMAT.format(
                model="ACCESS", scenario="SSP245"))
        RECOVERY_PATH.touch()
        generator = consolidator.generate_precipitation_dataset()
        EXPECTED_LOG_MESSAGE = (
            "Found recovery data for all cities under model 'ACCESS' and scenario 'SSP245'")

        with (
            patch("agents.consolidator.Consolidator._recover_data_from_file")
                as recovery_mock):
            recovery_mock.return_value = SAMPLE_RECOVERY_DATA
            for _ in SAMPLE_RECOVERY_DATA:
                next(generator)
        with self.assertRaises(StopIteration), self.assertLogs(
                logger, level=logging.DEBUG) as log_context:
            next(generator)
            self.assertIn(EXPECTED_LOG_MESSAGE, log_context.output[0])

    def test_generate_dataset_with_file_not_found(self):
        EXPECTED_LOG_MESSAGE = (
            "File corresponding to model 'ACCESS' and scenario 'SSP245' was not found, "
            "skipping")
        consolidator = Consolidator(
            self.cities, {"SSP245": "test"}, ["ACCESS"], self.SAMPLE_SOURCE_DIR)
        generator = consolidator.generate_precipitation_dataset()

        with (
                self.assertRaises(StopIteration),
                patch("agents.consolidator.Consolidator._set_final_state"),
                self.assertLogs(logger, level=logging.WARNING) as log_context):
            next(generator)
            self.assertIn(EXPECTED_LOG_MESSAGE, log_context.output[0])

    def test_call_recovery_data_dump(self):
        CITIES = {
            "Florianópolis": {
                "nearest": {
                    "lat": -27.625,
                    "lon": -48.875
                }
            }
        }
        mock_data_series = np.array([(1, 20) for _ in range(100)])
        consolidator = Consolidator(
            CITIES, {"Histórico": "test"}, ["ACCESS"], self.SAMPLE_SOURCE_DIR, True)
        EXPECTED_FILE_PATH = Path(consolidator.source_dir, "ACCESS-pr-hist.nc")
        EXPECTED_FILE_PATH.touch()
        generator = consolidator.generate_precipitation_dataset()
        for_recovery = {}
        with (
                patch("agents.extractors.NetCDFExtractor.__init__") as extractor_init,
                patch("agents.consolidator.Consolidator._set_final_state"),
                patch(
                    "agents.extractors.NetCDFExtractor.extract_precipitation"
                    ) as precipitation_mock):
            extractor_init.return_value = None
            precipitation_mock.return_value = mock_data_series
            result_data, result_meta = next(generator)
            for_recovery["Florianópolis"] = {"data": result_data, "metadata": result_meta}
        with (
                self.assertRaises(StopIteration),
                patch("agents.consolidator.Consolidator._dump_recovery_data") as dump_mock):
            next(generator)
            dump_mock.assert_called_once_with("ACCESS", "Histórico", for_recovery)

    def test_no_call_recovery_data_dump(self):
        CITIES = {
            "Florianópolis": {
                "nearest": {
                    "lat": -27.625,
                    "lon": -48.875
                }
            }
        }
        mock_data_series = np.array([(1, 20) for _ in range(100)])
        consolidator = Consolidator(
            CITIES, {"Histórico": "test"}, ["ACCESS"], self.SAMPLE_SOURCE_DIR)
        EXPECTED_FILE_PATH = Path(consolidator.source_dir, "ACCESS-pr-hist.nc")
        EXPECTED_FILE_PATH.touch()
        generator = consolidator.generate_precipitation_dataset()
        for_recovery = {}
        with (
                patch("agents.extractors.NetCDFExtractor.__init__") as extractor_init,
                patch("agents.consolidator.Consolidator._set_final_state"),
                patch(
                    "agents.extractors.NetCDFExtractor.extract_precipitation"
                    ) as precipitation_mock):
            extractor_init.return_value = None
            precipitation_mock.return_value = mock_data_series
            result_data, result_meta = next(generator)
            for_recovery["Florianópolis"] = {"data": result_data, "metadata": result_meta}
        with (
                self.assertRaises(StopIteration),
                patch("agents.consolidator.Consolidator._dump_recovery_data") as dump_mock):
            next(generator)
            dump_mock.assert_not_called()

    def tearDown(self):
        shutil.rmtree(self.SAMPLE_SOURCE_DIR, ignore_errors=True)
        shutil.rmtree(self.EXPECTED_TEMP_DIR, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
