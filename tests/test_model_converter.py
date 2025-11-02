import os
import sys
import yaml
from types import SimpleNamespace, ModuleType

# Create lightweight fake modules for optional heavy deps so imports won't fail in CI
fake_transformers = ModuleType("transformers")
class _DummyTok:
    @staticmethod
    def from_pretrained(name):
        return SimpleNamespace(save_pretrained=lambda path: None)

fake_transformers.AutoTokenizer = _DummyTok
fake_transformers.AutoImageProcessor = _DummyTok
sys.modules["transformers"] = fake_transformers

fake_opt = ModuleType("optimum.onnxruntime")
class _DummyORT:
    @staticmethod
    def from_pretrained(name, export=True):
        return SimpleNamespace(save_pretrained=lambda path: None)

fake_opt.ORTModelForSequenceClassification = _DummyORT
fake_opt.ORTModelForImageClassification = _DummyORT
sys.modules["optimum.onnxruntime"] = fake_opt

from inference.model_converter import ModelConverter


def test_convert_text_model(tmp_path, monkeypatch):
    # Create a temporary config file with one text model
    cfg = {"models": [{"name": "distilbert-base-uncased", "type": "text", "quantized": True, "optimization_level": 1}]}
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    # Prepare dummy model and tokenizer objects
    class DummyModel:
        def save_pretrained(self, path):
            os.makedirs(path, exist_ok=True)
            # create a minimal model file to emulate an ONNX artifact
            with open(os.path.join(path, "model.onnx"), "wb") as f:
                f.write(b"dummy")

    class DummyTokenizer:
        def save_pretrained(self, path):
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, "tokenizer.json"), "w") as f:
                f.write("{}")

    # Monkeypatch transformer and optimum loader methods used in ModelConverter
    monkeypatch.setattr("inference.model_converter.ORTModelForSequenceClassification.from_pretrained", lambda name, export=True: DummyModel())
    monkeypatch.setattr("inference.model_converter.AutoTokenizer.from_pretrained", lambda name: DummyTokenizer())

    # Prevent heavy optimizer/quantizer operations - just create expected dirs
    def fake_optimize(src, dst):
        os.makedirs(dst, exist_ok=True)
        # create an optimized model file
        with open(os.path.join(dst, "model_optimized.onnx"), "wb") as f:
            f.write(b"opt")

    def fake_quantize(src, dst):
        os.makedirs(dst, exist_ok=True)
        with open(os.path.join(dst, "model_quantized.onnx"), "wb") as f:
            f.write(b"q")

    monkeypatch.setattr("inference.model_converter.Optimizer.optimize", fake_optimize)
    monkeypatch.setattr("inference.model_converter.Quantizer.quantize", fake_quantize)

    # Instantiate converter and force its output_dir to a temp directory
    converter = ModelConverter(str(cfg_path))
    converter.output_dir = str(tmp_path / "models")

    # Run conversion - should use our monkeypatched routines and finish quickly
    converter.convert_all_models()

    # Assert expected files/dirs created
    model_path = os.path.join(converter.output_dir, "distilbert-base-uncased")
    assert os.path.isdir(model_path)
    assert os.path.isfile(os.path.join(model_path, "model.onnx"))
    optimized_dir = os.path.join(model_path, "_optimized")
    assert os.path.isdir(optimized_dir)
    assert os.path.isfile(os.path.join(optimized_dir, "model_optimized.onnx"))
    quant_dir = os.path.join(optimized_dir, "_quantized")
    assert os.path.isdir(quant_dir)
    assert os.path.isfile(os.path.join(quant_dir, "model_quantized.onnx"))
