"""DocumentScanner — extract text from document images."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class DocumentScanner:
    """Extract text from document images using a FlashOCR predictor.

    Accepts full document images or pre-cropped text-line images.  When
    bounding boxes are supplied via :meth:`process_crops`, each crop is
    recognised independently and results are returned in order.

    Parameters
    ----------
    predictor
        An initialised ``Predictor`` instance.
    min_confidence : float
        Drop lines whose recognition confidence is below this value.
    strip_whitespace : bool
        Strip leading/trailing whitespace from each recognised line.
    """

    def __init__(
        self,
        predictor: Any,
        min_confidence: float = 0.3,
        strip_whitespace: bool = True,
    ):
        self.predictor = predictor
        self.min_confidence = min_confidence
        self.strip_whitespace = strip_whitespace

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_image(
        self,
        image: Union[str, Path, np.ndarray],
    ) -> str:
        """Recognise text from a single document image.

        Parameters
        ----------
        image : str | Path | np.ndarray
            File path or HWC uint8 numpy array.

        Returns
        -------
        str
            Recognised text (single prediction from the model).
        """
        result = self._run_predictor(image)
        if not result:
            return ""
        text, confidence = result[0]
        if confidence < self.min_confidence:
            return ""
        return text.strip() if self.strip_whitespace else text

    def process_crops(
        self,
        image: Union[str, Path, np.ndarray],
        boxes: Sequence[Tuple[int, int, int, int]],
    ) -> List[Dict[str, Any]]:
        """Recognise text from cropped regions of a document.

        Parameters
        ----------
        image : str | Path | np.ndarray
            Full document image (path or HWC numpy array).
        boxes : sequence of (x1, y1, x2, y2)
            Bounding boxes for each text region to recognise.

        Returns
        -------
        list[dict]
            Each dict: ``{"text": str, "confidence": float, "box": tuple}``.
        """
        img = self._load_image(image)
        results: List[Dict[str, Any]] = []

        for box in boxes:
            x1, y1, x2, y2 = box
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(img.shape[1], x2)
            y2 = min(img.shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                results.append({"text": "", "confidence": 0.0, "box": box})
                continue

            crop = img[y1:y2, x1:x2]
            preds = self._run_predictor(crop)
            if preds:
                text, conf = preds[0]
                if self.strip_whitespace:
                    text = text.strip()
            else:
                text, conf = "", 0.0

            results.append({"text": text, "confidence": round(conf, 4), "box": box})

        return results

    def process_directory(
        self,
        dir_path: Union[str, Path],
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff"),
    ) -> List[Dict[str, Any]]:
        """Scan all images in a directory.

        Returns
        -------
        list[dict]
            Each dict: ``{"file": str, "text": str, "confidence": float}``.
        """
        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        image_files = sorted(
            p for p in dir_path.iterdir()
            if p.suffix.lower() in extensions
        )
        logger.info("DocumentScanner: found %d images in %s", len(image_files), dir_path)

        results: List[Dict[str, Any]] = []
        for img_path in image_files:
            preds = self._run_predictor(img_path)
            if preds:
                text, conf = preds[0]
            else:
                text, conf = "", 0.0
            results.append({
                "file": str(img_path.name),
                "text": text.strip() if self.strip_whitespace else text,
                "confidence": round(conf, 4),
            })
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
