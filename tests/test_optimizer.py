"""Test suite for Optimizer class."""
import unittest
import tempfile
import os
from inference.optimizer import Optimizer

class TestOptimizer(unittest.TestCase):
    def test_optimizer_init(self):
        opt = Optimizer(level=1)
        self.assertEqual(opt.level, 1)

    def test_create_session_options(self):
        opts = Optimizer.create_session_options(level=1, intra_threads=2)
        self.assertEqual(opts.intra_op_num_threads, 2)

if __name__ == "__main__":
    unittest.main()
