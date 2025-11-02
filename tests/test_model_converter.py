"""Test suite for ModelConverter class (smoke test)."""
import unittest
import os
from inference.model_converter import ModelConverter

class TestModelConverter(unittest.TestCase):
    def test_init(self):
        config_path = "configs/config.yaml"
        if not os.path.exists(config_path):
            self.skipTest("Config file not found.")
        converter = ModelConverter(config_path)
        self.assertIsNotNone(converter)
        self.assertEqual(converter.config_path, config_path)
        self.assertTrue(os.path.exists(converter.output_dir))

if __name__ == "__main__":
    unittest.main()
