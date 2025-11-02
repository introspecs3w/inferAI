"""Test suite for Quantizer class."""
import unittest
from inference.quantizer import Quantizer

class TestQuantizer(unittest.TestCase):
    def test_quantizer_init(self):
        quant = Quantizer(mode="dynamic")
        self.assertEqual(quant.mode, "dynamic")
        self.assertTrue(quant.per_channel)

if __name__ == "__main__":
    unittest.main()
