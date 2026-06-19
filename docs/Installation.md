# Installation

## From PyPI

```bash
pip install flashocr
```

## With extras

```bash
pip install "flashocr[all]"       # Everything
pip install "flashocr[export]"    # ONNX export
pip install "flashocr[analytics]" # Benchmarking, plots
```

## From source

```bash
git clone https://github.com/FlashVision/FlashOCR.git
cd FlashOCR
pip install -e ".[all]"
```

## Docker

```bash
docker build -t flashocr -f docker/Dockerfile .
docker run --gpus all flashocr version
```

## Verify

```bash
flashocr check
flashocr version
```

## Requirements

- Python >= 3.8
- PyTorch >= 2.0
- OpenCV >= 4.8
