# Quick Start

## Train a model

```python
from flashocr import Trainer

trainer = Trainer(
    model_size="m",
    train_images="data/train",
    train_labels="data/train/labels.tsv",
    val_images="data/val",
    val_labels="data/val/labels.tsv",
    charset="0123456789abcdefghijklmnopqrstuvwxyz",
    epochs=100,
    device="cuda",
)
trainer.train()
```

## Run inference

```python
from flashocr import Predictor

predictor = Predictor(model_path="workspace/best.pth", device="cuda")
text, confidence = predictor.recognize_image("photo.jpg")
print(f"Recognized: '{text}' ({confidence:.2f})")
```

## Export to ONNX

```python
from flashocr import Exporter

exporter = Exporter(model_path="workspace/best.pth")
exporter.export(output="model.onnx", simplify=True)
```

## CLI

```bash
flashocr train --config configs/flashocr_m_coco.yaml --device cuda
flashocr predict --model best.pth --source image.jpg
flashocr export --model best.pth --output model.onnx
```
