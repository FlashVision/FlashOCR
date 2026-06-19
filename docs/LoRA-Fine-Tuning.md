# LoRA Fine-Tuning

## Overview

LoRA (Low-Rank Adaptation) freezes the backbone and trains only small low-rank adapters, reducing memory and training time significantly.

## Variants

| Variant | Description |
|---------|-------------|
| standard | Classic LoRA |
| dora | Weight-decomposed LoRA |
| lora_plus | Differentiated learning rates |
| adalora | Adaptive rank allocation |
| ortho | Orthogonal regularization |
| lora_fa | Frozen-A LoRA |

## Usage

```bash
flashocr train --config configs/flashocr_m_lora.yaml --device cuda
```

### Python API

```python
from flashocr import Trainer

trainer = Trainer(
    model_size="m",
    train_images="data/custom/train",
    train_labels="data/custom/train/labels.tsv",
    val_images="data/custom/val",
    val_labels="data/custom/val/labels.tsv",
    pretrained=True,
    use_lora=True,
    lora_rank=8,
    lora_alpha=16.0,
    lora_variant="dora",
)
trainer.train()
```

## QLoRA

Quantized LoRA reduces memory further by quantizing frozen weights:

```python
trainer = Trainer(
    model_size="m",
    pretrained=True,
    use_qlora=True,
    qlora_dtype="nf4",
    lora_rank=8,
)
trainer.train()
```

## Knowledge Distillation

Train a small student from a larger teacher:

```bash
flashocr train --config configs/flashocr_m_kd.yaml --device cuda
```

```python
from flashocr import Trainer

trainer = Trainer(
    model_size="m-0.5x",
    use_kd=True,
    kd_teacher_checkpoint="workspace/teacher/best.pth",
    kd_teacher_model_size="m-1.5x",
    kd_temperature=4.0,
)
trainer.train()
```
