"""
OCR-specific evaluation metrics.

Provides Character Error Rate (CER), Word Error Rate (WER), exact-match
accuracy, and normalised edit distance for text recognition evaluation.
"""

from collections import defaultdict
from typing import List


def _edit_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _edit_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insert = prev_row[j + 1] + 1
            delete = curr_row[j] + 1
            substitute = prev_row[j] + (0 if c1 == c2 else 1)
            curr_row.append(min(insert, delete, substitute))
        prev_row = curr_row

    return prev_row[-1]


def compute_cer(predictions: List[str], targets: List[str]) -> float:
    """Compute Character Error Rate.

    CER = (total edit distance) / (total target characters).

    Args:
        predictions: List of predicted strings.
        targets: List of ground-truth strings.

    Returns:
        CER as a float in ``[0, ∞)``.  A CER > 1.0 is possible when
        predictions are much longer than targets.
    """
    assert len(predictions) == len(targets), "Prediction/target length mismatch"

    total_dist = 0
    total_chars = 0
    for pred, gt in zip(predictions, targets):
        total_dist += _edit_distance(pred, gt)
        total_chars += max(len(gt), 1)

    return total_dist / max(total_chars, 1)


def compute_wer(predictions: List[str], targets: List[str]) -> float:
    """Compute Word Error Rate.

    WER = (total word-level edit distance) / (total target words).

    Args:
        predictions: List of predicted strings.
        targets: List of ground-truth strings.

    Returns:
        WER as a float in ``[0, ∞)``.
    """
    assert len(predictions) == len(targets), "Prediction/target length mismatch"

    total_dist = 0
    total_words = 0
    for pred, gt in zip(predictions, targets):
        pred_words = pred.strip().split()
        gt_words = gt.strip().split()
        total_dist += _edit_distance(pred_words, gt_words)
        total_words += max(len(gt_words), 1)

    return total_dist / max(total_words, 1)


def compute_accuracy(predictions: List[str], targets: List[str]) -> float:
    """Compute exact-match accuracy.

    Args:
        predictions: List of predicted strings.
        targets: List of ground-truth strings.

    Returns:
        Fraction of predictions that exactly match the target in ``[0, 1]``.
    """
    assert len(predictions) == len(targets), "Prediction/target length mismatch"
    if len(predictions) == 0:
        return 0.0
    correct = sum(p == t for p, t in zip(predictions, targets))
    return correct / len(predictions)


def compute_normalized_edit_distance(
    predictions: List[str], targets: List[str]
) -> float:
    """Compute mean normalised edit distance (1 − NED).

    For each sample: ``NED = edit_distance / max(len(pred), len(gt), 1)``.
    Returns ``1 − mean(NED)`` so higher is better.

    Args:
        predictions: List of predicted strings.
        targets: List of ground-truth strings.

    Returns:
        Mean (1 − NED) in ``[0, 1]``.
    """
    assert len(predictions) == len(targets), "Prediction/target length mismatch"
    if len(predictions) == 0:
        return 0.0

    total_ned = 0.0
    for pred, gt in zip(predictions, targets):
        dist = _edit_distance(pred, gt)
        max_len = max(len(pred), len(gt), 1)
        total_ned += dist / max_len

    mean_ned = total_ned / len(predictions)
    return 1.0 - mean_ned


class MeterBuffer:
    """Track multiple named metrics as running averages.

    Same pattern as FlashDet's MeterBuffer for consistency across the
    FlashVision ecosystem.
    """

    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self._meters: dict = defaultdict(lambda: {"values": [], "sum": 0.0, "count": 0})

    def update(self, values: dict = None, **kwargs):
        """Update meters with new values."""
        if values is not None:
            kwargs.update(values)
        for k, v in kwargs.items():
            meter = self._meters[k]
            meter["values"].append(v)
            meter["sum"] += v
            meter["count"] += 1
            if len(meter["values"]) > self.window_size:
                old = meter["values"].pop(0)
                meter["sum"] -= old
                meter["count"] -= 1

    def get_avg(self, name: str) -> float:
        meter = self._meters.get(name)
        if meter is None or meter["count"] == 0:
            return 0.0
        return meter["sum"] / meter["count"]

    def get_latest(self, name: str) -> float:
        meter = self._meters.get(name)
        if meter is None or not meter["values"]:
            return 0.0
        return meter["values"][-1]

    def clear(self):
        self._meters.clear()

    def __contains__(self, key: str) -> bool:
        return key in self._meters

    def __str__(self) -> str:
        parts = []
        for name, meter in self._meters.items():
            avg = meter["sum"] / max(meter["count"], 1)
            parts.append(f"{name}: {avg:.4f}")
        return "  ".join(parts)
