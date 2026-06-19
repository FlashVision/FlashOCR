"""Unit tests for FlashOCR models."""

import torch

from flashocr.models.recognizer import FlashOCR
from flashocr.models.lora import apply_lora


CHARSET = "0123456789abcdefghijklmnopqrstuvwxyz"


def _make_model(backbone_size="1.0x", hidden_size=256, num_layers=2):
    return FlashOCR(
        charset=CHARSET,
        input_size=(32, 128),
        backbone_size=backbone_size,
        encoder_out_channels=hidden_size,
        decoder_type="ctc",
        hidden_size=hidden_size,
        num_layers=num_layers,
        bidirectional=True,
        dropout=0.1,
        pretrained=False,
    )


def test_model_forward_m05x():
    model = _make_model(backbone_size="0.5x", hidden_size=128, num_layers=2)
    model.eval()
    x = torch.randn(2, 3, 32, 128)
    with torch.no_grad():
        out = model(x)
    assert "log_probs" in out
    log_probs = out["log_probs"]
    assert log_probs.ndim == 3
    assert log_probs.shape[1] == 2  # batch


def test_model_forward_m():
    model = _make_model(backbone_size="1.0x", hidden_size=256, num_layers=2)
    model.eval()
    x = torch.randn(2, 3, 32, 128)
    with torch.no_grad():
        out = model(x)
    assert "log_probs" in out
    log_probs = out["log_probs"]
    assert log_probs.ndim == 3
    assert log_probs.shape[1] == 2


def test_model_forward_m15x():
    model = _make_model(backbone_size="1.5x", hidden_size=384, num_layers=2)
    model.eval()
    x = torch.randn(2, 3, 32, 128)
    with torch.no_grad():
        out = model(x)
    assert "log_probs" in out
    log_probs = out["log_probs"]
    assert log_probs.ndim == 3
    assert log_probs.shape[1] == 2


def test_lora_reduces_trainable_params():
    model = _make_model(backbone_size="1.0x", hidden_size=256, num_layers=2)
    total_params = sum(p.numel() for p in model.parameters())
    model = apply_lora(model, rank=8, alpha=16.0)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_params < total_params


def test_model_relative_sizes():
    m05 = _make_model(backbone_size="0.5x", hidden_size=128, num_layers=2)
    m10 = _make_model(backbone_size="1.0x", hidden_size=256, num_layers=2)
    m15 = _make_model(backbone_size="1.5x", hidden_size=384, num_layers=2)

    p05 = sum(p.numel() for p in m05.parameters())
    p10 = sum(p.numel() for p in m10.parameters())
    p15 = sum(p.numel() for p in m15.parameters())

    assert p05 < p10 < p15
