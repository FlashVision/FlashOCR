"""
Export to ONNX
============================
Export a trained FlashOCR model to ONNX format for deployment.
"""
from flashocr import Exporter

exporter = Exporter(model_path="workspace/my_model/best.pth")

exporter.export(
    output="model.onnx",
    simplify=True,
    input_size=(32, 128),
    opset_version=17,
)

print("ONNX model exported successfully!")
print(f"Output: model.onnx")
