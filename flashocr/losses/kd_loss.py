"""
Knowledge Distillation losses for FlashOCR.

Implements distillation losses inspired by torchtune's KD recipe, adapted
for sequence-to-sequence text recognition models.

Supported distillation modes:
  - **Logit KD**: KL-divergence between teacher and student per-timestep
    classification logits (soft targets).
  - **Feature KD**: L2 alignment between teacher and student encoder
    features after a lightweight adapter projection.
  - **Combined**: Both logit and feature KD with configurable weighting.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LogitDistillationLoss(nn.Module):
    """Per-timestep logit distillation via KL-divergence.

    Matches the teacher's soft character distribution at every time step
    using temperature-scaled KL divergence — the same formulation used in
    Hinton et al. (2015) and torchtune's KD recipe.

    Args:
        temperature: Softmax temperature for KL divergence.
        loss_weight: Scalar multiplier for the loss.
    """

    def __init__(self, temperature: float = 4.0, loss_weight: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.loss_weight = loss_weight

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> dict:
        """Compute logit-level KD loss.

        Args:
            student_logits: ``[B, T, C]`` or ``[T, B, C]`` student output.
            teacher_logits: Same shape, detached teacher output.
            mask: Optional ``[B, T]`` boolean mask (``True`` = valid position).

        Returns:
            Dict with ``kd_logit_loss``.
        """
        if student_logits.dim() == 3 and student_logits.size(0) != teacher_logits.size(0):
            student_logits = student_logits.permute(1, 0, 2)
            teacher_logits = teacher_logits.permute(1, 0, 2)

        T = self.temperature
        s_log = F.log_softmax(student_logits / T, dim=-1)
        t_prob = F.softmax(teacher_logits / T, dim=-1)

        kl = F.kl_div(s_log, t_prob, reduction="none").sum(dim=-1)

        if mask is not None:
            kl = kl * mask.float()
            loss = kl.sum() / mask.float().sum().clamp(min=1.0)
        else:
            loss = kl.mean()

        loss = loss * (T * T) * self.loss_weight

        return {"kd_logit_loss": loss}


class FeatureDistillationLoss(nn.Module):
    """Encoder feature distillation via normalised L2 alignment.

    Aligns student encoder features to teacher encoder features using an
    optional 1×1 conv adapter (when channel counts differ).

    Args:
        student_channels: Student encoder output channels.
        teacher_channels: Teacher encoder output channels.
        loss_weight: Scalar multiplier for the loss.
    """

    def __init__(
        self,
        student_channels: int = 256,
        teacher_channels: int = 512,
        loss_weight: float = 0.5,
    ):
        super().__init__()
        self.loss_weight = loss_weight

        if student_channels != teacher_channels:
            self.adapter = nn.Conv2d(
                student_channels, teacher_channels, 1, bias=False
            )
        else:
            self.adapter = nn.Identity()

    def forward(
        self,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Compute feature-level KD loss.

        Args:
            student_feat: ``[B, C_s, H, W]`` student encoder feature map.
            teacher_feat: ``[B, C_t, H, W]`` teacher encoder feature map
                (detached).

        Returns:
            Scalar feature distillation loss.
        """
        s_feat = self.adapter(student_feat)

        if s_feat.shape[2:] != teacher_feat.shape[2:]:
            s_feat = F.adaptive_avg_pool2d(s_feat, teacher_feat.shape[2:])

        s_norm = F.normalize(s_feat, dim=1)
        t_norm = F.normalize(teacher_feat, dim=1)

        return self.loss_weight * F.mse_loss(s_norm, t_norm)


class KnowledgeDistillationLoss(nn.Module):
    """Combined knowledge distillation loss for OCR.

    Combines logit-level and feature-level distillation into a single
    module, inspired by torchtune's KD training recipe adapted for
    FlashOCR.

    Args:
        temperature: KL divergence temperature.
        logit_weight: Weight for the logit KD component.
        feature_weight: Weight for the feature KD component.
        student_channels: Student encoder channels.
        teacher_channels: Teacher encoder channels.
    """

    def __init__(
        self,
        temperature: float = 4.0,
        logit_weight: float = 1.0,
        feature_weight: float = 0.5,
        student_channels: int = 256,
        teacher_channels: int = 512,
    ):
        super().__init__()
        self.logit_loss = LogitDistillationLoss(
            temperature=temperature,
            loss_weight=logit_weight,
        )
        self.feature_loss = FeatureDistillationLoss(
            student_channels=student_channels,
            teacher_channels=teacher_channels,
            loss_weight=feature_weight,
        )
        self.logit_weight = logit_weight
        self.feature_weight = feature_weight

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_feat: torch.Tensor = None,
        teacher_feat: torch.Tensor = None,
        mask: torch.Tensor = None,
    ) -> dict:
        """Compute the combined KD loss.

        Args:
            student_logits: Student decoder / CTC logits.
            teacher_logits: Teacher decoder / CTC logits (detached).
            student_feat: Optional student encoder feature map.
            teacher_feat: Optional teacher encoder feature map (detached).
            mask: Optional valid-position mask for the logit loss.

        Returns:
            Dict with all loss components and the combined ``kd_loss``.
        """
        result = {}

        if self.logit_weight > 0:
            logit_res = self.logit_loss(student_logits, teacher_logits, mask)
            result.update(logit_res)
        else:
            result["kd_logit_loss"] = torch.tensor(
                0.0, device=student_logits.device
            )

        if (
            self.feature_weight > 0
            and student_feat is not None
            and teacher_feat is not None
        ):
            feat_loss = self.feature_loss(student_feat, teacher_feat)
            result["kd_feature_loss"] = feat_loss
        else:
            feat_loss = torch.tensor(0.0, device=student_logits.device)
            result["kd_feature_loss"] = feat_loss

        result["kd_loss"] = result["kd_logit_loss"] + feat_loss
        return result
