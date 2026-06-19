"""
OCR visualization utilities.
"""

import numpy as np
from typing import List, Tuple, Optional

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def _ensure_cv2():
    if not HAS_CV2:
        raise ImportError(
            "OpenCV is required for visualization. Install with: pip install opencv-python"
        )


def draw_text_on_image(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int] = (10, 30),
    font_scale: float = 0.8,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    bg_color: Optional[Tuple[int, int, int]] = (0, 0, 0),
) -> np.ndarray:
    """Draw text on an image with optional background rectangle.

    Args:
        image: BGR image (H, W, 3).
        text: Text string to draw.
        position: ``(x, y)`` top-left corner for the text.
        font_scale: Font scale factor.
        color: Text colour in BGR.
        thickness: Text thickness.
        bg_color: If not ``None``, draw a filled rectangle behind the text.

    Returns:
        Image with text drawn on it.
    """
    _ensure_cv2()
    img = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    if bg_color is not None:
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x, y = position
        cv2.rectangle(
            img,
            (x - 2, y - th - 4),
            (x + tw + 2, y + baseline + 2),
            bg_color,
            cv2.FILLED,
        )

    cv2.putText(img, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
    return img


def draw_ocr_results(
    image: np.ndarray,
    results: List[dict],
    font_scale: float = 0.6,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Draw recognised text annotations on (or beside) an image.

    Each entry in *results* should be a dict with at least a ``"text"`` key
    and optionally ``"confidence"`` and ``"bbox"`` (x1, y1, x2, y2).

    Args:
        image: BGR image.
        results: List of recognition result dicts.
        font_scale: Font scale.
        color: Text colour in BGR.

    Returns:
        Annotated image.
    """
    _ensure_cv2()
    img = image.copy()
    h, w = img.shape[:2]

    for i, res in enumerate(results):
        text = res.get("text", "")
        conf = res.get("confidence", None)
        bbox = res.get("bbox", None)

        label = text
        if conf is not None:
            label = f"{text} ({conf:.2f})"

        if bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            pos = (x1, max(y1 - 5, 15))
        else:
            pos = (10, 25 + i * 30)

        img = draw_text_on_image(img, label, position=pos, font_scale=font_scale, color=color)

    return img


def make_ocr_panel(
    image: np.ndarray,
    predicted_text: str,
    ground_truth_text: str = "",
    panel_height: int = 60,
    font_scale: float = 0.7,
) -> np.ndarray:
    """Create an image with a text panel below showing predicted and GT text.

    Args:
        image: BGR image.
        predicted_text: Model prediction.
        ground_truth_text: Ground-truth text (optional).
        panel_height: Height of the text panel in pixels.
        font_scale: Font scale for the panel text.

    Returns:
        Combined image with text panel appended at the bottom.
    """
    _ensure_cv2()
    h, w = image.shape[:2]
    panel = np.zeros((panel_height, w, 3), dtype=np.uint8)

    pred_label = f"Pred: {predicted_text}"
    panel = draw_text_on_image(
        panel, pred_label, position=(5, 22),
        font_scale=font_scale, color=(0, 255, 0), bg_color=None,
    )

    if ground_truth_text:
        gt_label = f"  GT: {ground_truth_text}"
        match = predicted_text == ground_truth_text
        gt_color = (0, 255, 0) if match else (0, 0, 255)
        panel = draw_text_on_image(
            panel, gt_label, position=(5, 48),
            font_scale=font_scale, color=gt_color, bg_color=None,
        )

    return np.vstack([image, panel])
