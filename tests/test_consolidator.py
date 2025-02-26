import json
import logging
import shutil
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from calculator import estimate_combinations
from consolidator import Consolidator, logger
from constants import CLIMATE_MODELS, INPUT_FILENAME_FORMAT, SSP_SCENARIOS
from tests.stub_netCDF4 import NetCDFStubGenerator
from tests.test_validators import SAMPLE_CITIES_PATH


class TestConsolidatorInternalFunctions(unittest.TestCase):

    SAMPLE_SOURCE_DIR = Path(__file__).parent / "temp"
    LATITUDES = [-34.125 + 0.25*step for step in range(6)]
    LONGITUDES = [-74.125 + 0.25*step for step in range(6)]

    def setUp(self):
        with open(SAMPLE_CITIES_PATH) as file:
            self.cities = json.load(file)
        self.models = CLIMATE_MODELS[:5]
        self.scenarios = {
            name: periods[0] for name, periods in SSP_SCENARIOS.items()
        }
        self.consolidator = Consolidator(
            self.cities, self.scenarios, self.models, self.SAMPLE_SOURCE_DIR)
        self.expected_total = estimate_combinations(
            self.models, self.scenarios, self.cities)

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
            for scenario_name in self.scenarios:
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

    SAMPLE_SOURCE_DIR = Path(__file__).parent
    SAMPLE_NC_PATH = Path(__file__).parent / "ACCESS-CM2-pr-hist.nc"

    def setUp(self):
        pass

    def test_generate_sample_precipitation_dataset(self):
        pass

    # TODO: patch extract_precipitation (at least once, return empty dataset)
    def test_fail_to_generate_dataset(self):
        pass

    # TODO: patch generate_precipitation_dataset (return different stuff to concat)
    def test_consolidate_dataset(self):
        pass


if __name__ == '__main__':
    unittest.main()
