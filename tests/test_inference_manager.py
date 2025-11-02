"""Test suite for InferenceManager class (smoke test)."""
import unittest
from inference.profiler import Profiler
from inference.inference_manager import InferenceManager
import os

class TestInferenceManager(unittest.TestCase):
    def test_init(self):
        config_path = "configs/config.yaml"
        if not os.path.exists(config_path):
            self.skipTest("Config file not found.")
        profiler = Profiler()
        manager = InferenceManager(config_path, profiler)
        self.assertIsNotNone(manager)

if __name__ == "__main__":
    unittest.main()
