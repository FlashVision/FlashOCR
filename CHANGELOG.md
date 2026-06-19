# Changelog

All notable changes to FlashOCR will be documented in this file.

## [1.0.0] — 2026-06-19

### Added
- **Package structure** — `pip install` from GitHub or PyPI
- **CLI** — `flashocr train`, `predict`, `val`, `export`, `check`, `settings`, `version`
- **Python API** — `Trainer`, `Predictor`, `Exporter`, `Validator`
- **Models** — FlashOCR-m-0.5x, FlashOCR-m, FlashOCR-m-1.5x
- **CRNN Architecture** — ShuffleNetV2 backbone + BiLSTM decoder + CTC loss
- **Attention decoder** — GRU-based alternative to CTC
- **LoRA fine-tuning** — 6 variants (standard, dora, lora_plus, adalora, ortho, lora_fa)
- **QLoRA** — INT8/NF4 quantized base weights + LoRA
- **Knowledge Distillation** — teacher-student training
- **Solutions** — PlateReader, DocumentScanner, ReceiptParser
- **Analytics** — Benchmark, Profiler
- **ONNX export** — with simplification support
- **Mixed precision** — AMP (FP16) training
- **Multi-GPU** — DataParallel support
- **CI/CD** — GitHub Actions (lint + test, auto-publish to PyPI)
- **Examples** — runnable example scripts

### Architecture
- ShuffleNetV2 backbone (0.5x, 1.0x, 1.5x)
- CNN encoder with adaptive pooling
- BiLSTM decoder with CTC
- Attention decoder (alternative)
