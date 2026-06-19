"""
CTC loss wrapper for OCR sequence prediction.
"""

import torch
import torch.nn as nn


class CTCLoss(nn.Module):
    """Connectionist Temporal Classification loss with sensible defaults.

    Wraps :class:`torch.nn.CTCLoss` and automatically computes input / target
    lengths from the batch so callers only need to pass logits and labels.

    Args:
        blank: Index of the CTC blank token (default ``0``).
        reduction: ``"mean"`` or ``"sum"``.
        zero_infinity: Clamp infinite losses to zero (recommended).
        loss_weight: Scalar multiplier applied to the final loss.
    """

    def __init__(
        self,
        blank: int = 0,
        reduction: str = "mean",
        zero_infinity: bool = True,
        loss_weight: float = 1.0,
    ):
        super().__init__()
        self.ctc = nn.CTCLoss(blank=blank, reduction=reduction, zero_infinity=zero_infinity)
        self.loss_weight = loss_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CTC loss.

        Args:
            logits: ``[T, B, C]`` model output (log-probabilities or raw
                logits — log-softmax is applied internally).
            targets: ``[B, S]`` padded target sequences.
            target_lengths: ``[B]`` true lengths of each target.

        Returns:
            Scalar loss value.
        """
        log_probs = logits.log_softmax(dim=2)
        T, B, _ = log_probs.size()
        input_lengths = torch.full((B,), T, dtype=torch.long, device=logits.device)

        loss = self.ctc(log_probs, targets, input_lengths, target_lengths)
        return loss * self.loss_weight
