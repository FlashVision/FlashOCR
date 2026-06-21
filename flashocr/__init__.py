"""FlashOCR — Ultra-lightweight text recognition (OCR) framework."""

__version__ = "1.0.0"

from flashocr.models.recognizer import FlashOCR
from flashocr.models.recognition import TrOCR, MultilingualCTCDecoder, detect_language
from flashocr.models.lora import apply_lora, apply_qlora, merge_lora_weights
from flashocr.engine.trainer import Trainer
from flashocr.engine.validator import Validator
from flashocr.engine.predictor import Predictor
from flashocr.engine.exporter import Exporter
from flashocr.cfg import get_config
from flashocr.analytics import Benchmark, Profiler
from flashocr.document import TableExtractor, LayoutAnalyzer, PDFProcessor

__all__ = [
    "FlashOCR", "TrOCR", "MultilingualCTCDecoder", "detect_language",
    "Trainer", "Validator", "Predictor", "Exporter",
    "apply_lora", "apply_qlora", "merge_lora_weights", "get_config",
    "Benchmark", "Profiler",
    "TableExtractor", "LayoutAnalyzer", "PDFProcessor",
    "__version__",
]
