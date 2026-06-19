"""
Cross-entropy loss for attention-based OCR decoders.
"""

import torch
import torch.nn as nn


class AttentionLoss(nn.Module):
    """Masked cross-entropy loss for attention decoder output.

    Ignores positions where the target equals *pad_idx* and optionally
    applies label smoothing.

    Args:
        pad_idx: Index of the padding token to ignore (default ``0``).
        label_smoothing: Label smoothing factor in ``[0, 1)`` (default ``0.0``).
        loss_weight: Scalar multiplier applied to the final loss.
    """

    def __init__(
        self,
        pad_idx: int = 0,
        label_smoothing: float = 0.0,
        loss_weight: float = 1.0,
    ):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(
            ignore_index=pad_idx,
            label_smoothing=label_smoothing,
        )
        self.loss_weight = loss_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute masked cross-entropy.

        Args:
            logits: ``[B, T, C]`` decoder output logits.
            targets: ``[B, T]`` ground-truth token indices.

        Returns:
            Scalar loss value.
        """
        B, T, C = logits.size()
        loss = self.ce(logits.reshape(B * T, C), targets.reshape(B * T))
        return loss * self.loss_weight
