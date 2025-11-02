"""Test suite for main.py (smoke test)."""
import unittest
import importlib.util
import os

class TestMain(unittest.TestCase):
    def test_import(self):
        path = os.path.join(os.path.dirname(__file__), "..", "main.py")
        spec = importlib.util.spec_from_file_location("main", path)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            self.fail(f"main.py import failed: {e}")

if __name__ == "__main__":
    unittest.main()
