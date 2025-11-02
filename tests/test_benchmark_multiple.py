"""Test suite for benchmark_multiple.py (smoke test)."""
import unittest
import importlib.util
import os

class TestBenchmarkMultiple(unittest.TestCase):
    def test_import(self):
        path = os.path.join(os.path.dirname(__file__), "..", "benchmark_multiple.py")
        spec = importlib.util.spec_from_file_location("benchmark_multiple", path)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            self.fail(f"benchmark_multiple.py import failed: {e}")

if __name__ == "__main__":
    unittest.main()
