"""PlateReader — license plate text recognition solution."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

_PLATE_PATTERN = re.compile(r"^[A-Z0-9]{2,10}$", re.IGNORECASE)


class PlateReader:
    """Recognize text from license plate crops.

    Wraps a :class:`~flashocr.engine.predictor.Predictor` to provide a
    high-level API for reading license plates from images or directories.

    Parameters
    ----------
    predictor
        An initialised ``Predictor`` instance.
    min_confidence : float
        Discard predictions below this confidence threshold.
    plate_pattern : str | None
        Optional regex that valid plates must match.  The default accepts
        2-10 alphanumeric characters.
    """

    def __init__(
        self,
        predictor: Any,
        min_confidence: float = 0.4,
        plate_pattern: Optional[str] = None,
    ):
        self.predictor = predictor
        self.min_confidence = min_confidence
        self._pattern = re.compile(plate_pattern) if plate_pattern else _PLATE_PATTERN

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_image(
        self,
        image: Union[str, Path, np.ndarray],
    ) -> List[Dict[str, Any]]:
        """Recognise plate text from a single image (or crop).

        Parameters
        ----------
        image : str | Path | np.ndarray
            File path or HWC uint8 numpy array of a plate crop.

        Returns
        -------
        list[dict]
            Each dict: ``{"text": str, "confidence": float, "valid": bool}``.
        """
        results = self._run_predictor(image)
        return self._postprocess(results)

    def process_crops(
        self,
        crops: Sequence[Union[str, Path, np.ndarray]],
    ) -> List[Dict[str, Any]]:
        """Recognise plate text from multiple pre-cropped plate images.

        Parameters
        ----------
        crops : sequence of images
            Each element is a file path or HWC numpy array.

        Returns
        -------
        list[dict]
            One result dict per crop.
        """
        all_results: List[Dict[str, Any]] = []
        for crop in crops:
            all_results.extend(self.process_image(crop))
        return all_results

    def process_directory(
        self,
        dir_path: Union[str, Path],
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
    ) -> List[Dict[str, Any]]:
        """Run plate recognition on every image in a directory.

        Parameters
        ----------
        dir_path : str | Path
            Directory containing plate crop images.
        extensions : tuple of str
            Recognised image file extensions.

        Returns
        -------
        list[dict]
            Each dict includes ``"file"`` with the source filename.
        """
        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        image_files = sorted(
            p for p in dir_path.iterdir()
            if p.suffix.lower() in extensions
        )
        logger.info("PlateReader: found %d images in %s", len(image_files), dir_path)

        results: List[Dict[str, Any]] = []
        for img_path in image_files:
            preds = self.process_image(img_path)
            for pred in preds:
                pred["file"] = str(img_path.name)
            results.extend(preds)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_predictor(self, image: Union[str, Path, np.ndarray]) -> List[Tuple[str, float]]:
        """Run the predictor and return ``[(text, confidence), ...]``."""
        result = self.predictor.predict(image)
        if isinstance(result, tuple) and len(result) == 2:
            return [result]
        if isinstance(result, list):
            return [(r["text"], r["confidence"]) for r in result]
        return []

    def _postprocess(self, results: List[Tuple[str, float]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for text, confidence in results:
            cleaned = self._clean_plate(text)
            valid = bool(self._pattern.match(cleaned)) and confidence >= self.min_confidence
            out.append({
                "text": cleaned,
                "confidence": round(confidence, 4),
                "valid": valid,
            })
        return out

    @staticmethod
    def _clean_plate(text: str) -> str:
        """Normalise plate text: uppercase, strip whitespace and special chars."""
        text = text.upper().strip()
        text = re.sub(r"[^A-Z0-9]", "", text)
        return text
