# Models

## Available Variants

| Model | Params | FP16 Size | Accuracy (IIIT5k) | GPU Latency |
|-------|--------|-----------|-------------------|-------------|
| FlashOCR-m-0.5x | 0.6M | ~1.5 MB | — | 1.2 ms |
| FlashOCR-m | 1.5M | ~3.0 MB | 85.2% | 2.1 ms |
| FlashOCR-m-1.5x | 3.2M | ~6.5 MB | 88.7% | 3.4 ms |

## Architecture

- **Backbone**: ShuffleNetV2 (width multiplier configurable: 0.5x, 1.0x, 1.5x)
- **Encoder**: CNN with adaptive pooling (height → 1)
- **Decoder**: BiLSTM + CTC (default) or GRU-based Attention
- **Loss**: CTC Loss (connectionist temporal classification)

## Decoder Types

| Decoder | Description | Best For |
|---------|-------------|----------|
| CTC | BiLSTM + CTC loss | Fast inference, fixed-length outputs |
| Attention | GRU + Bahdanau attention | Variable-length, complex scripts |

## Pretrained Weights

Pretrained weights are auto-downloaded on first use:

```python
trainer = Trainer(model_size="m", pretrained=True)
```
