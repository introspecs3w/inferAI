import sys
import yaml
from types import ModuleType, SimpleNamespace

# Prevent heavy optional imports at module import time
fake_transformers = ModuleType("transformers")
class _DummyTok:
    @staticmethod
    def from_pretrained(path):
        return SimpleNamespace()

fake_transformers.AutoTokenizer = _DummyTok
fake_transformers.AutoImageProcessor = _DummyTok
sys.modules["transformers"] = fake_transformers

fake_opt = ModuleType("optimum.onnxruntime")
fake_opt.ORTModelForSequenceClassification = lambda *a, **k: SimpleNamespace()
fake_opt.ORTModelForImageClassification = lambda *a, **k: SimpleNamespace()
sys.modules["optimum.onnxruntime"] = fake_opt

from inference.inference_manager import InferenceManager


class FakeBackend:
    def __init__(self, config, profiler):
        self.config = config
        self.profiler = profiler

    def load(self):
        # simulate load
        self.loaded = True

    def run_inference(self, inputs):
        # return a deterministic mock response
        return {"predicted_class_ids": [42], "probabilities": [0.99], "input_texts": inputs}


class FakeProfiler:
    def profile(self, func, *args, **kwargs):
        # simply call the function (we don't have a real session.run here)
        return func(*args, **kwargs)

    def report(self):
        return {"mean_ms": 0.0}

    def clear(self):
        pass


def test_inference_manager_uses_backends(tmp_path, monkeypatch):
    # Create a minimal config with one onnx backend entry
    cfg = {"models": [{"name": "fake-model", "type": "text", "backend": "onnx"}]}
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    # Monkeypatch BackendFactory.get_backend to return our FakeBackend instance
    def fake_get_backend(name, config, profiler):
        return FakeBackend(config, profiler)

    monkeypatch.setattr("inference.inference_manager.BackendFactory.get_backend", fake_get_backend)

    profiler = FakeProfiler()
    manager = InferenceManager(str(cfg_path), profiler=profiler)

    # Ensure backend was registered in manager
    assert "fake-model" in manager.backends

    # Exercise run_inference
    resp = manager.run_inference("fake-model", ["hello world"])
    assert resp["predicted_class_ids"] == [42]
    assert resp["input_texts"] == ["hello world"]
