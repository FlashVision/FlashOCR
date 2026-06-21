"""Comprehensive test suite for FlashOCR."""

import subprocess
import sys

import pytest
import torch

B, C, H, W = 2, 3, 32, 128
NUM_CLASSES = 37
CHARSET = "0123456789abcdefghijklmnopqrstuvwxyz"


@pytest.fixture
def dummy_input():
    return torch.randn(B, C, H, W)


# ===================================================================
# 1. MODEL ARCHITECTURES
# ===================================================================


class TestFlashOCR:
    def test_forward_ctc(self, dummy_input):
        from flashocr.models.recognizer import FlashOCR

        model = FlashOCR(
            charset=CHARSET,
            input_size=(H, W),
            backbone_size="0.5x",
            encoder_out_channels=64,
            decoder_type="ctc",
            hidden_size=64,
            num_layers=1,
            pretrained=False,
        )
        model.eval()
        with torch.no_grad():
            out = model(dummy_input)
        assert "log_probs" in out
        assert out["log_probs"].dim() == 3

    def test_forward_attention(self, dummy_input):
        from flashocr.models.recognizer import FlashOCR

        model = FlashOCR(
            charset=CHARSET,
            input_size=(H, W),
            backbone_size="0.5x",
            encoder_out_channels=64,
            decoder_type="attention",
            hidden_size=64,
            num_layers=1,
            pretrained=False,
        )
        model.eval()
        with torch.no_grad():
            out = model(dummy_input)
        assert "logits" in out

    def test_ctc_training_loss(self, dummy_input):
        from flashocr.models.recognizer import FlashOCR

        model = FlashOCR(
            charset=CHARSET,
            input_size=(H, W),
            backbone_size="0.5x",
            encoder_out_channels=64,
            decoder_type="ctc",
            hidden_size=64,
            num_layers=1,
            pretrained=False,
        )
        model.train()
        targets = torch.randint(1, len(CHARSET) + 1, (B * 5,))
        target_lengths = torch.full((B,), 5, dtype=torch.long)
        out = model(dummy_input, targets=targets, target_lengths=target_lengths)
        assert "loss" in out
        assert torch.isfinite(out["loss"])

    def test_gradient_flow(self):
        from flashocr.models.recognizer import FlashOCR

        model = FlashOCR(
            charset="abc",
            input_size=(32, 64),
            backbone_size="0.5x",
            encoder_out_channels=32,
            decoder_type="ctc",
            hidden_size=32,
            num_layers=1,
            pretrained=False,
        )
        x = torch.randn(1, 3, 32, 64, requires_grad=True)
        out = model(x)
        out["log_probs"].sum().backward()
        assert x.grad is not None

    def test_model_info(self):
        from flashocr.models.recognizer import FlashOCR

        model = FlashOCR(
            charset=CHARSET, input_size=(H, W), backbone_size="0.5x", encoder_out_channels=64, pretrained=False
        )
        info = model.get_model_info()
        assert info["name"] == "FlashOCR"
        assert info["total_params"] > 0

    def test_predict(self):
        from flashocr.models.recognizer import FlashOCR

        model = FlashOCR(
            charset=CHARSET,
            input_size=(32, 64),
            backbone_size="0.5x",
            encoder_out_channels=64,
            decoder_type="ctc",
            hidden_size=64,
            num_layers=1,
            pretrained=False,
        )
        results = model.predict(torch.randn(1, 3, 32, 64))
        assert len(results) == 1
        text, conf = results[0]
        assert isinstance(text, str)
        assert isinstance(conf, float)

    def test_build_model(self):
        from flashocr.cfg import get_config
        from flashocr.models.recognizer import build_model

        cfg = get_config(model_size="m-0.5x", input_height=32, input_width=64)
        cfg.model.backbone_pretrained = False
        model = build_model(cfg)
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(1, 3, 32, 64))
        assert "log_probs" in out


class TestTrOCR:
    def test_forward(self):
        from flashocr.models.recognition.trocr import TrOCR

        model = TrOCR(
            charset=CHARSET,
            input_size=(32, 64),
            patch_size=4,
            embed_dim=64,
            num_heads=4,
            encoder_layers=1,
            decoder_layers=1,
            max_seq_len=20,
        )
        model.eval()
        x = torch.randn(1, 3, 32, 64)
        with torch.no_grad():
            out = model(x)
        assert "logits" in out

    def test_param_count(self):
        from flashocr.models.recognition.trocr import TrOCR

        model = TrOCR(
            charset="abc",
            input_size=(32, 64),
            patch_size=4,
            embed_dim=32,
            num_heads=4,
            encoder_layers=1,
            decoder_layers=1,
            max_seq_len=10,
        )
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert total > 0
        assert trainable > 0
        assert trainable == total


class TestMultilingual:
    def test_detect_language(self):
        from flashocr.models.recognition.multilingual import detect_language

        lang = detect_language("hello")
        assert isinstance(lang, str)

    def test_build_charset(self):
        from flashocr.models.recognition.multilingual import build_multilingual_charset

        charset = build_multilingual_charset(["en"])
        assert len(charset) > 0

    def test_multilingual_ctc_decoder(self):
        from flashocr.models.recognition.multilingual import MultilingualCTCDecoder

        dec = MultilingualCTCDecoder(
            in_channels=64, hidden_size=32, num_classes=100, num_layers=1, num_languages=3, adapter_dim=16
        )
        x = torch.randn(1, 10, 64)
        out = dec(x)
        assert out.dim() == 3


# ===================================================================
# 2. LOSSES
# ===================================================================


class TestLosses:
    def test_ctc_loss(self):
        from flashocr.losses import CTCLoss

        loss_fn = CTCLoss()
        logits = torch.randn(10, 2, 37)
        targets = torch.randint(1, 37, (2, 5))
        target_lengths = torch.full((2,), 5, dtype=torch.long)
        loss = loss_fn(logits, targets, target_lengths)
        assert torch.isfinite(loss)

    def test_attention_loss(self):
        from flashocr.losses import AttentionLoss

        loss_fn = AttentionLoss()
        logits = torch.randn(2, 10, 37)
        targets = torch.randint(0, 37, (2, 10))
        loss = loss_fn(logits, targets)
        assert torch.isfinite(loss)

    def test_kd_loss(self):
        from flashocr.losses import LogitDistillationLoss

        loss_fn = LogitDistillationLoss()
        student = torch.randn(2, 10, 37)
        teacher = torch.randn(2, 10, 37)
        result = loss_fn(student, teacher)
        assert isinstance(result, dict)

    def test_loss_gradient(self):
        from flashocr.losses import CTCLoss

        loss_fn = CTCLoss()
        logits = torch.randn(10, 2, 37, requires_grad=True)
        targets = torch.randint(1, 37, (2, 5))
        target_lengths = torch.full((2,), 5, dtype=torch.long)
        loss = loss_fn(logits, targets, target_lengths)
        loss.backward()
        assert logits.grad is not None


# ===================================================================
# 3. REGISTRY
# ===================================================================


class TestRegistry:
    def test_registries_exist(self):
        from flashocr.registry import BACKBONES, DECODERS

        assert BACKBONES is not None
        assert DECODERS is not None

    def test_registry_operations(self):
        from flashocr.registry import Registry

        reg = Registry("test")

        @reg.register("A")
        class A:
            pass

        assert "A" in reg
        assert len(reg) == 1


# ===================================================================
# 4. CLI
# ===================================================================


class TestCLI:
    def test_version(self):
        result = subprocess.run(
            [sys.executable, "-m", "flashocr.cli", "version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "FlashOCR" in result.stdout

    def test_no_command(self):
        result = subprocess.run(
            [sys.executable, "-m", "flashocr.cli"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0


# ===================================================================
# 5. ENGINE
# ===================================================================


class TestEngine:
    def test_imports(self):
        from flashocr.engine.exporter import Exporter
        from flashocr.engine.trainer import Trainer

        assert Trainer is not None
        assert Exporter is not None


# ===================================================================
# 6. DOCUMENT PROCESSING
# ===================================================================


class TestDocument:
    def test_table_extractor(self):
        from flashocr.document import TableExtractor

        ext = TableExtractor()
        assert ext is not None

    def test_layout_analyzer(self):
        from flashocr.document import LayoutAnalyzer

        analyzer = LayoutAnalyzer()
        assert analyzer is not None

    def test_pdf_processor(self):
        from flashocr.document import PDFProcessor

        proc = PDFProcessor()
        assert proc is not None


# ===================================================================
# 7. ENCODER / DECODER
# ===================================================================


class TestEncoderDecoder:
    def test_cnn_encoder(self):
        from flashocr.models.encoder import CNNEncoder

        enc = CNNEncoder(in_channels=192, out_channels=64)
        feats = [torch.randn(1, 48, 8, 32), torch.randn(1, 96, 4, 16), torch.randn(1, 192, 2, 8)]
        out = enc(feats)
        assert out.dim() == 3

    def test_ctc_decoder(self):
        from flashocr.models.decoder import CTCDecoder

        dec = CTCDecoder(in_channels=64, hidden_size=32, num_classes=37, num_layers=1)
        x = torch.randn(1, 10, 64)
        out = dec(x)
        assert out.dim() == 3

    def test_attention_decoder(self):
        from flashocr.models.decoder import AttentionDecoder

        dec = AttentionDecoder(in_channels=64, hidden_size=32, num_classes=39)
        x = torch.randn(1, 10, 64)
        out = dec(x)
        assert out.dim() == 3


# ===================================================================
# 8. SOLUTIONS
# ===================================================================


class TestSolutions:
    def test_plate_reader(self):
        from flashocr.solutions import PlateReader

        assert PlateReader is not None

    def test_document_scanner(self):
        from flashocr.solutions import DocumentScanner

        assert DocumentScanner is not None

    def test_receipt_parser(self):
        from flashocr.solutions import ReceiptParser

        assert ReceiptParser is not None


# ===================================================================
# 9. CONFIG
# ===================================================================


class TestConfig:
    def test_get_config(self):
        from flashocr.cfg import get_config

        cfg = get_config(model_size="m", input_height=32, input_width=128)
        assert cfg.model.backbone_size == "1.0x"
        assert cfg.model.input_size == (32, 128)

    def test_config_variants(self):
        from flashocr.cfg import get_config

        for size in ["m-0.5x", "m", "m-1.5x"]:
            cfg = get_config(model_size=size)
            assert cfg.model.charset == CHARSET


# ===================================================================
# 10. LoRA
# ===================================================================


class TestLoRA:
    def test_apply_lora(self):
        from flashocr.models.lora import apply_lora
        from flashocr.models.recognizer import FlashOCR

        model = FlashOCR(
            charset="abc", input_size=(32, 64), backbone_size="0.5x", encoder_out_channels=32, pretrained=False
        )
        model = apply_lora(model, rank=4, alpha=8.0)
        trainable = sum(1 for p in model.parameters() if p.requires_grad)
        assert trainable > 0


# ===================================================================
# 11. EDGE CASES
# ===================================================================


class TestEdgeCases:
    def test_empty_predict(self):
        from flashocr.models.recognizer import FlashOCR

        model = FlashOCR(
            charset=CHARSET, input_size=(32, 64), backbone_size="0.5x", encoder_out_channels=64, pretrained=False
        )
        results = model.predict(torch.randn(1, 3, 32, 64))
        assert len(results) == 1

    def test_wrong_channels(self):
        from flashocr.models.recognizer import FlashOCR

        model = FlashOCR(
            charset="abc", input_size=(32, 64), backbone_size="0.5x", encoder_out_channels=32, pretrained=False
        )
        with pytest.raises(RuntimeError):
            model(torch.randn(1, 1, 32, 64))


# ===================================================================
# 12. INTEGRATION
# ===================================================================


class TestIntegration:
    def test_full_pipeline(self):
        from flashocr.models.recognizer import FlashOCR

        model = FlashOCR(
            charset="abc",
            input_size=(32, 64),
            backbone_size="0.5x",
            encoder_out_channels=32,
            decoder_type="ctc",
            hidden_size=32,
            num_layers=1,
            pretrained=False,
        )
        model.train()
        x = torch.randn(2, 3, 32, 64)
        targets = torch.randint(1, 4, (2 * 3,))
        target_lengths = torch.full((2,), 3, dtype=torch.long)

        out = model(x, targets=targets, target_lengths=target_lengths)
        loss = out["loss"]
        assert torch.isfinite(loss)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        results = model.predict(x)
        assert len(results) == 2


# ===================================================================
# 13. ANALYTICS
# ===================================================================


class TestAnalytics:
    def test_imports(self):
        from flashocr.analytics import Benchmark, Profiler

        assert Benchmark is not None
        assert Profiler is not None
