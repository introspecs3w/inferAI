"""Test suite for backend_factory (smoke test)."""
import unittest
from inference.backend_factory import BackendConfig, IntBackend

class DummyBackend(IntBackend):
    def load(self):
        return True
    def run_inference(self, inputs):
        return inputs

class TestBackendFactory(unittest.TestCase):
    def test_backend_config(self):
        config = BackendConfig()
        config.model_name = "test"
        config.model_type = "onnx"
        config.provider = "CPU"
        config.optimization_level = 1
        config.quantized = False
        self.assertEqual(config.model_name, "test")

    def test_int_backend(self):
        config = BackendConfig()
        backend = DummyBackend("dummy", config)
        self.assertEqual(backend.backend_name, "dummy")
        self.assertEqual(backend.config, config)
        self.assertTrue(backend.load())
        self.assertEqual(backend.run_inference(42), 42)

if __name__ == "__main__":
    unittest.main()
