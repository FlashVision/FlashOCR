# FAQ

## How fast is FlashOCR?

FlashOCR-m achieves 400+ FPS on GPU and 60+ FPS on edge devices with only 1.5M parameters for text recognition.

## What character sets are supported?

Any custom charset can be specified. Default is alphanumeric (`0-9a-z`). You can include uppercase, special characters, or non-Latin scripts.

## Can I use my own backbone?

Yes, use the registry system to register custom backbones:

```python
from flashocr.registry import BACKBONES

@BACKBONES.register("my_backbone")
class MyBackbone(nn.Module):
    ...
```

## How to export for mobile?

```bash
flashocr export --model best.pth --output model.onnx --simplify
```

Then convert ONNX to TFLite, CoreML, or NCNN as needed.

## What's the difference between CTC and Attention decoder?

- **CTC**: Faster inference, simpler training, best for fixed-charset recognition
- **Attention**: Handles variable-length outputs better, more accurate for complex scripts

## What's the difference between LoRA variants?

- **standard**: Classic low-rank adapters
- **dora**: Better generalization via weight decomposition
- **adalora**: Automatically adjusts rank per layer
- **lora_plus**: Different LR for A and B matrices

## What data format does FlashOCR expect?

A TSV file with tab-separated image paths and text labels:
```
image_001.jpg	hello
image_002.jpg	world
```
