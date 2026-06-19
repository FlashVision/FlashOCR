"""FlashOCR Validator — compute CER, WER, and accuracy on a validation set."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from flashocr.cfg import get_config
from flashocr.models import FlashOCR, build_model
from flashocr.data import create_dataloader

logger = logging.getLogger(__name__)


def _ctc_greedy_decode(log_probs: torch.Tensor, charset: str) -> List[str]:
    """Greedy-decode CTC output into strings."""
    preds = log_probs.argmax(dim=2).permute(1, 0)
    texts = []
    for seq in preds:
        chars = []
        prev = -1
        for idx in seq.tolist():
            if idx != 0 and idx != prev:
                if 1 <= idx <= len(charset):
                    chars.append(charset[idx - 1])
            prev = idx
        texts.append("".join(chars))
    return texts


def _char_error_rate(pred: str, target: str) -> float:
    if len(target) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    n, m = len(pred), len(target)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, m + 1):
            cost = 0 if pred[i - 1] == target[j - 1] else 1
            dp[j] = min(prev[j] + 1, dp[j - 1] + 1, prev[j - 1] + cost)
    return dp[m] / m


def _word_error_rate(pred: str, target: str) -> float:
    pred_words = pred.split()
    target_words = target.split()
    if len(target_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0
    n, m = len(pred_words), len(target_words)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, m + 1):
            cost = 0 if pred_words[i - 1] == target_words[j - 1] else 1
            dp[j] = min(prev[j] + 1, dp[j - 1] + 1, prev[j - 1] + cost)
    return dp[m] / m


class Validator:
    """Validate a FlashOCR model on a dataset with CER/WER computation.

    Example::

        from flashocr import Validator

        val = Validator(model_path="workspace/checkpoint_best.pth")
        results = val.validate()
        print(f"CER: {results['cer']:.4f}")
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[nn.Module] = None,
        device: str = "cuda",
        batch_size: int = 64,
        workers: int = 4,
        val_images: Optional[str] = None,
        val_labels: Optional[str] = None,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.workers = workers

        cfg = get_config()
        self.val_images = val_images or cfg.data.val_images
        self.val_labels = val_labels or cfg.data.val_labels
        self.charset = cfg.model.charset
        self.input_size = cfg.model.input_size

        if model is not None:
            self.model = model.to(self.device)
        elif model_path is not None:
            self.model, self.charset, self.input_size = self._load_model(model_path, cfg)
        else:
            raise ValueError("Either model_path or model must be provided")

    def _load_model(self, model_path: str, cfg) -> Tuple[nn.Module, str, tuple]:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        backbone_size = cfg.model.backbone_size
        hidden_size = cfg.model.hidden_size
        charset = cfg.model.charset
        input_size = cfg.model.input_size

        if "config" in checkpoint:
            ckpt_cfg = checkpoint["config"]
            backbone_size = ckpt_cfg.get("backbone_size", backbone_size)
            hidden_size = ckpt_cfg.get("hidden_size", hidden_size)
            input_size = ckpt_cfg.get("input_size", input_size)
            if "charset" in ckpt_cfg and ckpt_cfg["charset"]:
                charset = ckpt_cfg["charset"]

        num_classes = len(charset) + 1

        model = FlashOCR(
            num_classes=num_classes,
            input_size=input_size,
            backbone_size=backbone_size,
            hidden_size=hidden_size,
        )

        if "model_state_dict" in checkpoint:
            sd = checkpoint["model_state_dict"]
            if checkpoint.get("half", False):
                sd = {k: v.float() if v.is_floating_point() else v for k, v in sd.items()}
            model.load_state_dict(sd, strict=False)
        elif "state_dict" in checkpoint:
            sd = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
            model.load_state_dict(sd, strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

        model = model.to(self.device).eval()
        return model, charset, input_size

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation and return CER, WER, accuracy, and loss.

        Returns:
            Dict with keys: ``cer``, ``wer``, ``accuracy``, ``val_loss``.
        """
        self.model.eval()

        val_loader = create_dataloader(
            img_dir=self.val_images,
            label_file=self.val_labels,
            charset=self.charset,
            batch_size=self.batch_size,
            input_size=self.input_size,
            num_workers=self.workers,
            is_train=False,
        )

        ctc_loss_fn = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
        loss_sum = 0.0
        loss_count = 0
        total_cer = 0.0
        total_wer = 0.0
        total_correct = 0
        total_samples = 0

        for batch in val_loader:
            images = batch["images"].to(self.device)
            targets = batch["targets"].to(self.device)
            target_lengths = batch["target_lengths"].to(self.device)
            raw_labels = batch["raw_labels"]

            log_probs = self.model(images)
            T, N, C = log_probs.shape
            input_lengths = torch.full((N,), T, dtype=torch.long, device=self.device)

            try:
                loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)
                loss_sum += loss.item() * N
                loss_count += N
            except Exception:
                pass

            pred_texts = _ctc_greedy_decode(log_probs, self.charset)

            for pred, gt in zip(pred_texts, raw_labels):
                pred_lower = pred.lower()
                gt_lower = gt.lower()
                total_cer += _char_error_rate(pred_lower, gt_lower)
                total_wer += _word_error_rate(pred_lower, gt_lower)
                if pred_lower == gt_lower:
                    total_correct += 1
                total_samples += 1

        cer = total_cer / max(total_samples, 1)
        wer = total_wer / max(total_samples, 1)
        accuracy = total_correct / max(total_samples, 1)
        val_loss = loss_sum / max(loss_count, 1)

        result = {
            "cer": cer,
            "wer": wer,
            "accuracy": accuracy,
            "val_loss": val_loss,
        }

        logger.info(
            f"Validation: CER={cer:.4f}, WER={wer:.4f}, "
            f"Acc={accuracy:.4f}, Loss={val_loss:.4f}"
        )

        return result
