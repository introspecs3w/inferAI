"""Test suite for Profiler class."""
import unittest
from inference.profiler import Profiler

def dummy_func(x):
    return x * 2

class TestProfiler(unittest.TestCase):
    def test_profile(self):
        profiler = Profiler()
        result = profiler.profile(dummy_func, 3)
        self.assertEqual(result, 6)
        self.assertTrue(len(profiler.records) > 0)

if __name__ == "__main__":
    unittest.main()
