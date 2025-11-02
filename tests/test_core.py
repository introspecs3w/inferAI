"""Test suite for inferAI project."""
import unittest
from inference.core import greet

class TestCore(unittest.TestCase):
    def test_greet_valid(self):
        self.assertEqual(greet("Alice"), "Hello, Alice!")
        self.assertEqual(greet("Bob"), "Hello, Bob!")

    def test_greet_empty(self):
        with self.assertRaises(ValueError):
            greet("")

if __name__ == "__main__":
    unittest.main()
