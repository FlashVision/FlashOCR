"""Tests for TrOCR and multilingual components."""

import torch

CHARSET = "0123456789abcdefghijklmnopqrstuvwxyz"


def test_trocr_forward():
    from flashocr.models.recognition.trocr import TrOCR

    model = TrOCR(
        charset=CHARSET,
        input_size=(32, 128),
        embed_dim=64,
        num_heads=4,
        encoder_layers=2,
        decoder_layers=2,
        max_seq_len=16,
        patch_size=4,
    )
    model.eval()

    x = torch.randn(2, 3, 32, 128)
    with torch.no_grad():
        out = model(x)
    assert "logits" in out
    assert out["logits"].ndim == 3


def test_trocr_with_targets():
    from flashocr.models.recognition.trocr import TrOCR

    model = TrOCR(
        charset=CHARSET,
        input_size=(32, 128),
        embed_dim=64,
        num_heads=4,
        encoder_layers=2,
        decoder_layers=2,
        max_seq_len=16,
        patch_size=4,
    )
    model.train()

    x = torch.randn(2, 3, 32, 128)
    targets = torch.randint(0, model.vocab_size, (2, 10))
    out = model(x, targets=targets)
    assert "loss" in out
    assert out["loss"].requires_grad


def test_trocr_predict():
    from flashocr.models.recognition.trocr import TrOCR

    model = TrOCR(
        charset=CHARSET,
        input_size=(32, 128),
        embed_dim=64,
        num_heads=4,
        encoder_layers=2,
        decoder_layers=2,
        max_seq_len=8,
        patch_size=4,
    )

    x = torch.randn(2, 3, 32, 128)
    results = model.predict(x)
    assert len(results) == 2
    for text, conf in results:
        assert isinstance(text, str)
        assert isinstance(conf, float)


def test_trocr_model_info():
    from flashocr.models.recognition.trocr import TrOCR

    model = TrOCR(charset=CHARSET, embed_dim=64, num_heads=4, encoder_layers=2, decoder_layers=2, patch_size=4)
    info = model.get_model_info()
    assert info["name"] == "TrOCR"
    assert info["total_params"] > 0


def test_trocr_registered():
    from flashocr.registry import DECODERS

    assert "TrOCR" in DECODERS


def test_language_detection():
    from flashocr.models.recognition.multilingual import detect_language, detect_languages

    assert detect_language("hello world") == "en"
    assert detect_language("مرحبا") == "ar"
    assert detect_language("नमस्ते") == "hi"
    assert detect_language("こんにちは") == "ja"
    assert detect_language("안녕하세요") == "ko"

    langs = detect_languages("hello world 你好")
    assert "en" in langs


def test_build_multilingual_charset():
    from flashocr.models.recognition.multilingual import build_multilingual_charset

    charset = build_multilingual_charset(["en", "ar"])
    assert len(charset) > 50
    assert "a" in charset
    assert "0" in charset

    charset_multi = build_multilingual_charset(["en", "hi", "ja", "ko"])
    assert len(charset_multi) > len(charset)


def test_multilingual_ctc_decoder():
    from flashocr.models.recognition.multilingual import MultilingualCTCDecoder

    decoder = MultilingualCTCDecoder(
        in_channels=64,
        hidden_size=32,
        num_classes=100,
        num_layers=1,
    )

    x = torch.randn(2, 10, 64)
    lang_ids = torch.tensor([0, 1])
    log_probs = decoder(x, lang_id=lang_ids)
    assert log_probs.shape == (10, 2, 100)


def test_multilingual_decoder_registered():
    from flashocr.registry import DECODERS

    assert "MultilingualCTCDecoder" in DECODERS


def test_language_adapter():
    from flashocr.models.recognition.multilingual import LanguageAdapter

    adapter = LanguageAdapter(embed_dim=64, num_languages=5, bottleneck_dim=16)
    x = torch.randn(2, 10, 64)
    lang_ids = torch.tensor([0, 3])
    out = adapter(x, lang_ids)
    assert out.shape == x.shape
