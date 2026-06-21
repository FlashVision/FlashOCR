"""Multi-language text recognition support.

Extends FlashOCR with language detection and expanded character sets for
Arabic, Devanagari, Japanese, Korean, and many other scripts.
"""

import logging
import unicodedata
from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from flashocr.registry import DECODERS

logger = logging.getLogger(__name__)


LANGUAGE_CHARSETS: Dict[str, str] = {
    "en": (
        "0123456789"
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        " !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    ),
    "zh": "".join(chr(c) for c in range(0x4E00, 0x9FFF + 1)),
    "ar": "".join(chr(c) for c in range(0x0600, 0x06FF + 1)),
    "hi": "".join(chr(c) for c in range(0x0900, 0x097F + 1)),
    "ja": (
        "".join(chr(c) for c in range(0x3040, 0x309F + 1))  # Hiragana
        + "".join(chr(c) for c in range(0x30A0, 0x30FF + 1))  # Katakana
    ),
    "ko": "".join(chr(c) for c in range(0xAC00, 0xD7AF + 1)),
    "th": "".join(chr(c) for c in range(0x0E00, 0x0E7F + 1)),
    "bn": "".join(chr(c) for c in range(0x0980, 0x09FF + 1)),
    "ta": "".join(chr(c) for c in range(0x0B80, 0x0BFF + 1)),
    "te": "".join(chr(c) for c in range(0x0C00, 0x0C7F + 1)),
    "ru": "".join(chr(c) for c in range(0x0400, 0x04FF + 1)),
    "el": "".join(chr(c) for c in range(0x0370, 0x03FF + 1)),
}

SCRIPT_RANGES: List[Tuple[int, int, str]] = [
    (0x0600, 0x06FF, "ar"),
    (0x0900, 0x097F, "hi"),
    (0x0980, 0x09FF, "bn"),
    (0x0B80, 0x0BFF, "ta"),
    (0x0C00, 0x0C7F, "te"),
    (0x0E00, 0x0E7F, "th"),
    (0x3040, 0x309F, "ja"),
    (0x30A0, 0x30FF, "ja"),
    (0x4E00, 0x9FFF, "zh"),
    (0xAC00, 0xD7AF, "ko"),
    (0x0400, 0x04FF, "ru"),
    (0x0370, 0x03FF, "el"),
    (0x0041, 0x007A, "en"),
    (0x0030, 0x0039, "en"),
]


def detect_language(text: str) -> str:
    """Detect the dominant script/language of a text string.

    Returns:
        ISO 639-1 language code (e.g. ``"en"``, ``"ar"``, ``"hi"``).
    """
    if not text:
        return "en"

    counts: Counter = Counter()
    for ch in text:
        cp = ord(ch)
        for lo, hi, lang in SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[lang] += 1
                break
        else:
            cat = unicodedata.category(ch)
            if cat.startswith("L"):
                counts["en"] += 1

    if not counts:
        return "en"

    return counts.most_common(1)[0][0]


def detect_languages(text: str, min_ratio: float = 0.1) -> List[str]:
    """Detect all scripts present in a text string.

    Args:
        text: Input text.
        min_ratio: Minimum fraction of characters for a language to be included.

    Returns:
        List of language codes sorted by frequency.
    """
    if not text:
        return ["en"]

    counts: Counter = Counter()
    total = 0
    for ch in text:
        cp = ord(ch)
        for lo, hi, lang in SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[lang] += 1
                total += 1
                break

    if total == 0:
        return ["en"]

    return [lang for lang, cnt in counts.most_common() if cnt / total >= min_ratio]


def build_multilingual_charset(
    languages: List[str],
    include_digits: bool = True,
    include_punctuation: bool = True,
    max_chars: int = 8000,
) -> str:
    """Build a combined charset for multiple languages.

    Args:
        languages: List of language codes.
        include_digits: Include ASCII digits.
        include_punctuation: Include common punctuation.
        max_chars: Maximum charset size.

    Returns:
        Combined charset string with unique characters.
    """
    chars = set()

    if include_digits:
        chars.update("0123456789")
    if include_punctuation:
        chars.update(" !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")

    for lang in languages:
        lang_chars = LANGUAGE_CHARSETS.get(lang, "")
        chars.update(lang_chars)

    sorted_chars = sorted(chars, key=ord)
    if len(sorted_chars) > max_chars:
        sorted_chars = sorted_chars[:max_chars]

    return "".join(sorted_chars)


class LanguageAdapter(nn.Module):
    """Per-language adapter layer inserted into a recognition model.

    Provides a lightweight bottleneck transform conditioned on the
    detected language, allowing a single model to handle multiple scripts.

    Args:
        embed_dim: Feature dimension.
        num_languages: Number of supported languages.
        bottleneck_dim: Adapter bottleneck dimension.
    """

    def __init__(
        self,
        embed_dim: int,
        num_languages: int = len(LANGUAGE_CHARSETS),
        bottleneck_dim: int = 64,
    ):
        super().__init__()
        self.num_languages = num_languages
        self.down = nn.Linear(embed_dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, embed_dim)
        self.lang_embed = nn.Embedding(num_languages, bottleneck_dim)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        lang_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply language-conditioned adapter.

        Args:
            x: (B, T, D) feature tensor.
            lang_id: (B,) language index tensor. If None, uses index 0 (English).

        Returns:
            Adapted feature tensor (B, T, D).
        """
        residual = x
        h = self.down(x)
        h = self.act(h)

        if lang_id is not None:
            lang_bias = self.lang_embed(lang_id).unsqueeze(1)  # (B, 1, bottleneck)
            h = h + lang_bias

        h = self.up(h)
        return self.norm(residual + h)


LANG_TO_IDX: Dict[str, int] = {lang: i for i, lang in enumerate(LANGUAGE_CHARSETS.keys())}


def get_lang_id(lang_code: str) -> int:
    return LANG_TO_IDX.get(lang_code, 0)


def get_lang_ids(lang_codes: List[str], device: str = "cpu") -> torch.Tensor:
    return torch.tensor([get_lang_id(lc) for lc in lang_codes], device=device)


@DECODERS.register("MultilingualCTCDecoder")
class MultilingualCTCDecoder(nn.Module):
    """CTC decoder with language adapter for multilingual recognition.

    Args:
        in_channels: Input feature dimension.
        hidden_size: LSTM hidden dimension.
        num_classes: Number of output classes.
        num_layers: Number of LSTM layers.
        dropout: Dropout probability.
        bidirectional: Whether to use bidirectional LSTM.
        num_languages: Number of supported languages.
        adapter_dim: Language adapter bottleneck dimension.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_size: int = 256,
        num_classes: int = 8003,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
        num_languages: int = len(LANGUAGE_CHARSETS),
        adapter_dim: int = 64,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.lang_adapter = LanguageAdapter(
            embed_dim=in_channels,
            num_languages=num_languages,
            bottleneck_dim=adapter_dim,
        )

        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_out = hidden_size * 2 if bidirectional else hidden_size
        self.projection = nn.Linear(lstm_out, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        lang_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, T, in_channels) input features.
            lang_id: (B,) language indices.

        Returns:
            Log-probabilities (T, B, num_classes) in CTC format.
        """
        x = self.lang_adapter(x, lang_id)
        output, _ = self.lstm(x)
        logits = self.projection(output)
        log_probs = torch.nn.functional.log_softmax(logits, dim=2)
        return log_probs.permute(1, 0, 2)

    def decode(
        self,
        log_probs: torch.Tensor,
        charset: str,
    ) -> List[str]:
        """Greedy CTC decoding."""
        predictions = log_probs.argmax(dim=2).permute(1, 0)  # (B, T)
        results = []
        blank_idx = 0
        for pred in predictions:
            chars = []
            prev = -1
            for idx in pred.tolist():
                if idx != prev and idx != blank_idx:
                    char_idx = idx - 1
                    if 0 <= char_idx < len(charset):
                        chars.append(charset[char_idx])
                prev = idx
            results.append("".join(chars))
        return results
