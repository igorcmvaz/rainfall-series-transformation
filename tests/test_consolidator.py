import json
import logging
import shutil
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd

from calculator import estimate_combinations
from consolidator import Consolidator, logger
from constants import CLIMATE_MODELS, INPUT_FILENAME_FORMAT, SSP_SCENARIOS
from tests.stub_netCDF4 import NetCDFStubGenerator
from tests.stub_precipitation import SAMPLE_JSON_PATH
from tests.test_validators import SAMPLE_CITIES_PATH


class TestConsolidatorInternalFunctions(unittest.TestCase):

    SAMPLE_SOURCE_DIR = Path(__file__).parent / "temp"
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
            f"[{ORIGINAL_PROCESSED_COUNT+1}/{self.expected_total}] Error during processing."
            f" Details: test=foo, bar=zii, el=psy")

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
            f"[{ORIGINAL_PROCESSED_COUNT+1}/{self.expected_total}] Completed processing."
            f" Details: test=foo, bar=2, el=None")

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

        with patch("extractor.NetCDFExtractor.extract_precipitation") as precipitation_mock:
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
                "consolidator.Consolidator.generate_precipitation_indices"
                ) as generator_mock:
            generator_mock.return_value = (MOCK_DATAFRAME for _ in range(5))
            result = consolidator.consolidate_dataset()
        self.assertTrue((result == EXPECTED_OUTPUT).all().values)


if __name__ == '__main__':
    unittest.main()
