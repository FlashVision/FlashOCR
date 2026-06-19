"""ReceiptParser — extract structured data from receipt images."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

_PRICE_PATTERN = re.compile(r"\$?\d+[.,]\d{2}")
_TOTAL_KEYWORDS = {"total", "grand total", "amount due", "balance due", "sum"}
_TAX_KEYWORDS = {"tax", "vat", "gst", "hst"}


class ReceiptParser:
    """Parse structured data from receipt images.

    Wraps a :class:`~flashocr.engine.predictor.Predictor` to recognise
    text from receipt line crops and extract structured fields such as
    line items, totals, and tax.

    Parameters
    ----------
    predictor
        An initialised ``Predictor`` instance.
    min_confidence : float
        Discard lines whose recognition confidence is below this value.
    """

    def __init__(
        self,
        predictor: Any,
        min_confidence: float = 0.3,
    ):
        self.predictor = predictor
        self.min_confidence = min_confidence

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_image(
        self,
        image: Union[str, Path, np.ndarray],
    ) -> Dict[str, Any]:
        """Recognise and parse a single receipt image.

        The image is expected to be a single text-line crop or full
        receipt.  For a full receipt, use :meth:`process_crops` with
        pre-detected line bounding boxes.

        Parameters
        ----------
        image : str | Path | np.ndarray
            File path or HWC uint8 numpy array.

        Returns
        -------
        dict
            ``{"raw_text": str, "line_items": list, "total": str|None,
              "tax": str|None, "confidence": float}``
        """
        preds = self._run_predictor(image)
        if not preds:
            return self._empty_result()

        text, confidence = preds[0]
        lines = [text] if text.strip() else []
        return self._parse_lines(lines, confidence)

    def process_crops(
        self,
        image: Union[str, Path, np.ndarray],
        boxes: Sequence[Tuple[int, int, int, int]],
    ) -> Dict[str, Any]:
        """Parse receipt from cropped text-line regions.

        Parameters
        ----------
        image : str | Path | np.ndarray
            Full receipt image (path or HWC numpy array).
        boxes : sequence of (x1, y1, x2, y2)
            Bounding boxes for each text line, **sorted top-to-bottom**.

        Returns
        -------
        dict
            Same structure as :meth:`process_image`.
        """
        img = self._load_image(image)
        lines: List[str] = []
        confidences: List[float] = []

        for box in boxes:
            x1, y1, x2, y2 = box
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(img.shape[1], x2)
            y2 = min(img.shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = img[y1:y2, x1:x2]
            preds = self._run_predictor(crop)
            if preds:
                text, conf = preds[0]
                if conf >= self.min_confidence and text.strip():
                    lines.append(text.strip())
                    confidences.append(conf)

        avg_conf = float(np.mean(confidences)) if confidences else 0.0
        return self._parse_lines(lines, avg_conf)

    def process_directory(
        self,
        dir_path: Union[str, Path],
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
    ) -> List[Dict[str, Any]]:
        """Parse all receipt images in a directory.

        Returns
        -------
        list[dict]
            Each dict includes ``"file"`` alongside parsed fields.
        """
        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        image_files = sorted(
            p for p in dir_path.iterdir()
            if p.suffix.lower() in extensions
        )
        logger.info("ReceiptParser: found %d images in %s", len(image_files), dir_path)

        results: List[Dict[str, Any]] = []
        for img_path in image_files:
            parsed = self.process_image(img_path)
            parsed["file"] = str(img_path.name)
            results.append(parsed)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_lines(self, lines: List[str], confidence: float) -> Dict[str, Any]:
        """Extract structured fields from recognised text lines."""
        raw_text = "\n".join(lines)
        line_items: List[Dict[str, Any]] = []
        total: Optional[str] = None
        tax: Optional[str] = None

        for line in lines:
            lower = line.lower().strip()

            if any(kw in lower for kw in _TOTAL_KEYWORDS):
                price = _PRICE_PATTERN.search(line)
                if price:
                    total = price.group()
                continue

            if any(kw in lower for kw in _TAX_KEYWORDS):
                price = _PRICE_PATTERN.search(line)
                if price:
                    tax = price.group()
                continue

            price_match = _PRICE_PATTERN.search(line)
            if price_match:
                description = line[:price_match.start()].strip().rstrip(".-:")
                line_items.append({
                    "description": description if description else line,
                    "price": price_match.group(),
                })
            elif line.strip():
                line_items.append({
                    "description": line.strip(),
                    "price": None,
                })

        return {
            "raw_text": raw_text,
            "line_items": line_items,
            "total": total,
            "tax": tax,
            "confidence": round(confidence, 4),
        }

    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        return {
            "raw_text": "",
            "line_items": [],
            "total": None,
            "tax": None,
            "confidence": 0.0,
        }

    def _run_predictor(self, image: Union[str, Path, np.ndarray]) -> List[Tuple[str, float]]:
        result = self.predictor.predict(image)
        if isinstance(result, tuple) and len(result) == 2:
            return [result]
        if isinstance(result, list):
            return [(r["text"], r["confidence"]) for r in result]
        return []

    @staticmethod
    def _load_image(image: Union[str, Path, np.ndarray]) -> np.ndarray:
        if isinstance(image, np.ndarray):
            return image
        import cv2
        img = cv2.imread(str(image))
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {image}")
        return img
