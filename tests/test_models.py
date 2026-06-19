"""Unit tests for FlashOCR models."""

import torch

from flashocr.models import build_model


def test_model_forward_m05x():
    model = build_model(backbone_size="0.5x", hidden_size=128, num_layers=2, num_classes=37)
    x = torch.randn(2, 3, 32, 128)
    out = model(x)
    assert out.ndim == 3
    assert out.shape[0] == 2
    assert out.shape[2] == 37


def test_model_forward_m():
    model = build_model(backbone_size="1.0x", hidden_size=256, num_layers=2, num_classes=37)
    x = torch.randn(2, 3, 32, 128)
    out = model(x)
    assert out.ndim == 3
    assert out.shape[0] == 2
    assert out.shape[2] == 37


def test_model_forward_m15x():
    model = build_model(backbone_size="1.5x", hidden_size=384, num_layers=2, num_classes=37)
    x = torch.randn(2, 3, 32, 128)
    out = model(x)
    assert out.ndim == 3
    assert out.shape[0] == 2
    assert out.shape[2] == 37


def test_lora_reduces_trainable_params():
    model = build_model(backbone_size="1.0x", hidden_size=256, num_layers=2, num_classes=37)
    total_params = sum(p.numel() for p in model.parameters())
    model.apply_lora(rank=8, alpha=16.0)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_params < total_params * 0.15


def test_model_relative_sizes():
    m05 = build_model(backbone_size="0.5x", hidden_size=128, num_layers=2, num_classes=37)
    m10 = build_model(backbone_size="1.0x", hidden_size=256, num_layers=2, num_classes=37)
    m15 = build_model(backbone_size="1.5x", hidden_size=384, num_layers=2, num_classes=37)

    p05 = sum(p.numel() for p in m05.parameters())
    p10 = sum(p.numel() for p in m10.parameters())
    p15 = sum(p.numel() for p in m15.parameters())

    assert p05 < p10 < p15
