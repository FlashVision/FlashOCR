<p align="center">
  <img src="assets/logo.png" width="200" alt="FlashOCR Logo">
</p>

<h1 align="center">FlashOCR</h1>

<p align="center">
  <a href="https://pypi.org/project/flashocr/"><img src="https://img.shields.io/pypi/v/flashocr?color=blue&logo=pypi&logoColor=white" alt="PyPI"></a>
  <a href="https://github.com/FlashVision/FlashOCR/actions"><img src="https://img.shields.io/github/actions/workflow/status/FlashVision/FlashOCR/ci.yml?logo=github" alt="CI"></a>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Python-3.8+-3776ab?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/ONNX-Export-005CED?logo=onnx&logoColor=white" alt="ONNX">
  <img src="https://img.shields.io/badge/LoRA-Fine_Tuning-ff6b6b" alt="LoRA">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

<p align="center">
  <b>Ultra-lightweight real-time text recognition with LoRA fine-tuning, knowledge distillation, and edge deployment</b>
</p>

<p align="center">
  <a href="#installation">Install</a> •
  <a href="#usage">Usage</a> •
  <a href="#models">Models</a> •
  <a href="#solutions">Solutions</a> •
  <a href="#training">Training</a> •
  <a href="#lora-fine-tuning">LoRA</a> •
  <a href="#knowledge-distillation">KD</a> •
  <a href="#onnx-export">ONNX</a> •
  <a href="#contributing">Contributing</a>
</p>

---

## What is FlashOCR?

FlashOCR is an end-to-end text recognition (OCR) framework built for **speed and efficiency**. Based on a CRNN architecture with a **ShuffleNetV2** backbone, **BiLSTM** sequence decoder, and **CTC** loss, it delivers real-time text recognition with models as small as ~0.4M parameters (~0.8 MB FP16).

It ships as a `pip`-installable Python package with a CLI, a high-level Python API, and built-in solutions for license plate reading, document scanning, and receipt parsing — similar to how you'd use Ultralytics YOLO.

```bash
pip install -e .
flashocr train --train-images data/train --train-labels data/train/labels.tsv \
               --val-images data/val --val-labels data/val/labels.tsv --device cuda
flashocr predict --model best.pth --source image.jpg
```

---

## Installation

### pip (recommended)

```bash
pip install flashocr

# With all extras (analytics, ONNX export)
pip install "flashocr[all]"
```

### From source (for development)

```bash
git clone https://github.com/FlashVision/FlashOCR.git
cd FlashOCR
pip install -e ".[all]"
```

### Optional extras

```bash
pip install -e ".[export]"      # ONNX export support
pip install -e ".[solutions]"   # PlateReader, DocumentScanner, ReceiptParser
pip install -e ".[analytics]"   # Benchmarking, profiling, plots
pip install -e ".[all]"         # Everything
```

### Verify installation

```bash
flashocr check       # runs full health check
flashocr settings    # shows Python, PyTorch, CUDA, GPU info
flashocr version     # prints version
```

---

## Usage

### Python API

```python
from flashocr import Trainer, Predictor, Exporter

# Train a model
trainer = Trainer(
    model_size="m",
    train_images="data/train",
    train_labels="data/train/labels.tsv",
    val_images="data/val",
    val_labels="data/val/labels.tsv",
    charset="0123456789abcdefghijklmnopqrstuvwxyz",
    epochs=100,
    device="cuda",
    use_lora=True,
)
trainer.train()

# Run inference
predictor = Predictor(model_path="workspace/best.pth", device="cuda")
text, confidence = predictor.recognize_image("photo.jpg")

# Export to ONNX
exporter = Exporter(model_path="workspace/best.pth")
exporter.export(output="model.onnx", simplify=True)
```

### CLI

```bash
# Train
flashocr train --model-size m --epochs 100 --device cuda \
  --train-images data/train --train-labels data/train/labels.tsv \
  --val-images data/val --val-labels data/val/labels.tsv

# Predict
flashocr predict --model best.pth --source image.jpg

# Validate
flashocr val --model best.pth --val-images data/val --val-labels data/val/labels.tsv

# Export
flashocr export --model best.pth --output model.onnx --simplify
```

---

## Models

| Model | Params | FP16 Size | Accuracy (IIIT5k) | GPU Latency |
|-------|--------|-----------|-------------------|-------------|
| **FlashOCR-m-0.5x** | ~0.4M | ~0.8 MB | — | 0.9 ms |
| **FlashOCR-m** | ~1.5M | ~3 MB | 85.2% | 2.1 ms |
| **FlashOCR-m-1.5x** | ~3M | ~6 MB | 88.7% | 3.4 ms |

### Config-driven Training (Model Zoo)

Pick a config and train — no code changes needed:

```bash
flashocr train --config configs/flashocr_m_coco.yaml --device cuda
flashocr train --config configs/flashocr_m_lora.yaml --device cuda
flashocr train --config configs/flashocr_m_kd.yaml --device cuda
```

Available configs in [`configs/`](configs/):
| Config | Description |
|--------|-------------|
| `flashocr_m_coco.yaml` | Standard FlashOCR-m training |
| `flashocr_m05x.yaml` | Ultra-light for edge deployment |
| `flashocr_m15x.yaml` | Larger model for better accuracy |
| `flashocr_m_lora.yaml` | LoRA fine-tuning on custom data |
| `flashocr_m_kd.yaml` | Knowledge distillation |

---

## Solutions

Built-in high-level applications for real-world use cases:

```python
from flashocr import Predictor
from flashocr.solutions import PlateReader, DocumentScanner, ReceiptParser

predictor = Predictor(model_path="best.pth")

# Read license plates
plate_reader = PlateReader(predictor)
results = plate_reader.process_image("car_photo.jpg")

# Scan documents
scanner = DocumentScanner(predictor)
text = scanner.process_image("document.jpg")

# Extract text from pre-detected line crops
lines = scanner.process_crops("document.jpg", boxes=[(10, 50, 400, 80), ...])

# Parse receipts
parser = ReceiptParser(predictor)
data = parser.process_image("receipt.jpg")
print(data["line_items"], data["total"])
```

| Solution | Description |
|----------|-------------|
| **PlateReader** | Read license plates from vehicle images |
| **DocumentScanner** | Extract text lines from document images |
| **ReceiptParser** | Parse structured data (line items, total, tax) from receipt images |

---

## Training

### Standard Training

```bash
flashocr train --model-size m --epochs 100 --batch-size 64 --device cuda \
  --train-images data/train --train-labels data/train/labels.tsv \
  --val-images data/val --val-labels data/val/labels.tsv
```

### LoRA / QLoRA Fine-Tuning

Parameter-efficient — freeze backbone, train only low-rank adapters:

```bash
# LoRA (6 variants: standard, dora, lora_plus, adalora, ortho, lora_fa)
flashocr train --lora --model-size m --epochs 50 --device cuda \
  --train-images data/train --train-labels data/train/labels.tsv \
  --val-images data/val --val-labels data/val/labels.tsv

# QLoRA (quantized base weights + LoRA)
flashocr train --qlora --model-size m --epochs 50 --device cuda \
  --train-images data/train --train-labels data/train/labels.tsv \
  --val-images data/val --val-labels data/val/labels.tsv
```

```python
from flashocr import Trainer

trainer = Trainer(
    model_size="m",
    use_lora=True,
    lora_rank=8,
    lora_alpha=16.0,
    lora_variant="dora",
    train_images="data/train",
    train_labels="data/train/labels.tsv",
    val_images="data/val",
    val_labels="data/val/labels.tsv",
)
trainer.train()
```

---

## LoRA Fine-Tuning

| Variant | Description |
|---------|-------------|
| standard | Classic LoRA |
| dora | Weight-decomposed LoRA (recommended) |
| lora_plus | Differentiated learning rates for A/B |
| adalora | Adaptive rank allocation |
| ortho | Orthogonal regularization |
| lora_fa | Frozen-A LoRA |

---

## Knowledge Distillation

Train a compact student model from a larger teacher:

```python
from flashocr import Trainer

trainer = Trainer(
    model_size="m-0.5x",
    use_kd=True,
    kd_teacher_checkpoint="workspace/teacher/best.pth",
    kd_teacher_model_size="m-1.5x",
    kd_temperature=4.0,
    train_images="data/train",
    train_labels="data/train/labels.tsv",
    val_images="data/val",
    val_labels="data/val/labels.tsv",
)
trainer.train()
```

---

## ONNX Export

```python
from flashocr import Exporter

exporter = Exporter(model_path="workspace/best.pth")
exporter.export(output="model.onnx", simplify=True)
```

```bash
flashocr export --model best.pth --output model.onnx --simplify
```

---

## Analytics

```python
from flashocr.analytics import Benchmark, Profiler

# Benchmark model speed
bench = Benchmark(model_path="best.pth", device="cuda")
results = bench.run()  # {'avg_ms': 2.1, 'fps': 476.3, 'min_ms': ..., 'max_ms': ..., ...}

# Profile layer-by-layer
profiler = Profiler(model_path="best.pth")
print(profiler.summary())             # per-layer timing table
print(profiler.parameter_count())     # parameter breakdown
print(profiler.memory_report())       # GPU memory usage

# Plot training curves
from flashocr.analytics import plot_training_curves, plot_error_rates
plot_training_curves(log, keys=["loss", "cer", "wer"], save_path="curves.png")
plot_error_rates(epochs, cer_values, wer_values, save_path="error_rates.png")
```

---

## Examples

Ready-to-run scripts in the [`examples/`](examples/) folder:

| Script | What it does |
|--------|--------------|
| `predict_text.py` | Recognize text in a single image |
| `train_custom_dataset.py` | Train on your own dataset |
| `train_with_lora.py` | LoRA fine-tuning (DoRA variant) |
| `export_onnx.py` | Export to ONNX for deployment |
| `benchmark_model.py` | Measure FPS and latency |

```bash
cd examples
python predict_text.py
python train_custom_dataset.py
```

---

## Project Structure

```
FlashOCR/
├── flashocr/                  # Main package (pip install -e .)
│   ├── __init__.py            # Public API
│   ├── cli.py                 # CLI entry point (flashocr command)
│   ├── registry.py            # Pluggable component registry
│   ├── cfg/                   # Configuration + YAML loading
│   ├── data/                  # Datasets, loaders, transforms
│   ├── engine/                # Trainer, Validator, Predictor, Exporter, Callbacks
│   ├── models/                # ShuffleNetV2, CNN encoder, BiLSTM decoder, LoRA
│   ├── losses/                # CTC, Attention, KD losses
│   ├── nn/                    # Neural network building blocks
│   ├── utils/                 # Metrics, visualization, checkpoint
│   ├── solutions/             # PlateReader, DocumentScanner, ReceiptParser
│   └── analytics/             # Benchmark, Profiler, plots
├── configs/                   # YAML configs for model zoo (pick & train)
├── examples/                  # Ready-to-run example scripts
├── tests/                     # Unit tests (pytest)
├── docker/                    # Dockerfile + docker-compose
├── docs/                      # Wiki documentation
├── pyproject.toml             # Package configuration
├── CONTRIBUTING.md            # How to contribute
├── CHANGELOG.md               # Version history
└── LICENSE                    # MIT
```

---

## Training Callbacks

Extend the training loop without modifying source code:

```python
from flashocr import Trainer
from flashocr.engine.callbacks import EarlyStopping, CSVLogger, TensorBoardCallback

trainer = Trainer(
    model_size="m",
    train_images="data/train",
    train_labels="data/train/labels.tsv",
    val_images="data/val",
    val_labels="data/val/labels.tsv",
)

trainer.add_callback(EarlyStopping(patience=20, metric="val_accuracy"))
trainer.add_callback(CSVLogger("metrics.csv"))
trainer.add_callback(TensorBoardCallback("runs/exp1"))

trainer.train()
```

Built-in callbacks: `EarlyStopping`, `CSVLogger`, `TensorBoardCallback`, `LRSchedulerCallback`.

---

## Docker

```bash
# Build
docker build -t flashocr -f docker/Dockerfile .

# Run inference
docker run --gpus all -v $(pwd)/data:/app/data flashocr predict --model best.pth --source data/test.jpg

# Or use docker-compose
cd docker && docker compose up
```

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
git clone https://github.com/FlashVision/FlashOCR.git
cd FlashOCR
pip install -e ".[dev,all]"
ruff check flashocr/
flashocr check
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <a href="https://github.com/FlashVision/FlashOCR">
    <b>FlashVision</b>
  </a>
  — Open-source lightweight AI
</p>
