# Training

## Standard Training

```bash
flashocr train --config configs/flashocr_m_coco.yaml --device cuda
```

## Config-driven Training

```bash
flashocr train --config configs/flashocr_m_coco.yaml
flashocr train --config configs/flashocr_m_lora.yaml
flashocr train --config configs/flashocr_m_kd.yaml
```

## Training Options

| Flag | Description | Default |
|------|-------------|---------|
| `--config` | YAML config file | — |
| `--model-size` | Model variant (m, m-0.5x, m-1.5x) | m |
| `--epochs` | Training epochs | 100 |
| `--batch-size` | Batch size | 64 |
| `--lr` | Learning rate | 0.001 |
| `--device` | Device (cuda/cpu) | cuda |
| `--amp` | Mixed precision training | false |
| `--multi-gpu` | DataParallel | false |
| `--pretrained` | Load pretrained weights | false |

## Data Format

FlashOCR expects a TSV label file with tab-separated image paths and labels:

```
data/
├── train/
│   ├── images/
│   │   ├── img_001.jpg
│   │   └── img_002.jpg
│   └── labels.tsv
└── val/
    ├── images/
    └── labels.tsv
```

**labels.tsv** format:
```
img_001.jpg	hello
img_002.jpg	world
img_003.jpg	12345
```

## Mixed Precision & Multi-GPU

```bash
flashocr train --config configs/flashocr_m_coco.yaml --amp --multi-gpu
```
