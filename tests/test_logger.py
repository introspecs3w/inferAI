"""Test suite for logger setup."""
import unittest
from inference.logger import setup_logger

class TestLogger(unittest.TestCase):
    def test_logger_creation(self):
        logger = setup_logger("test_logger")
        self.assertIsNotNone(logger)
        logger.info("Logger test message")

if __name__ == "__main__":
    unittest.main()
