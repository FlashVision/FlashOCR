"""FlashOCR Trainer — wraps the full training loop into a reusable class."""

import os
import copy
import math
import logging
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn

from flashocr.cfg import get_config
from flashocr.models import build_model
from flashocr.models.lora import (
    apply_lora, apply_qlora, merge_lora_weights, get_lora_state_dict,
)
from flashocr.data import create_dataloader

logger = logging.getLogger(__name__)


def _setup_logger(name: str, save_dir: str) -> logging.Logger:
    """Configure a logger with console and file handlers."""
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    log.propagate = False
    log.handlers = []

    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    import sys
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    log.addHandler(ch)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(os.path.join(save_dir, f"train_{ts}.log"))
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        log.addHandler(fh)

    return log


class _AverageMeter:
    """Compute and store running average."""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0


class ModelEMA:
    """Exponential Moving Average of model weights with adaptive decay warmup."""

    def __init__(self, model: nn.Module, decay: float = 0.9998, warmup: int = 2000):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.target_decay = decay
        self.warmup = warmup
        self.num_updates = 0
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @property
    def decay(self):
        return min(self.target_decay,
                   (1 + self.num_updates) / (self.warmup + self.num_updates))

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.num_updates += 1
        d = self.decay
        for ema_p, model_p in zip(self.ema.parameters(), model.parameters()):
            ema_p.data.mul_(d).add_(model_p.data, alpha=1.0 - d)
        for ema_b, model_b in zip(self.ema.buffers(), model.buffers()):
            ema_b.copy_(model_b)

    def state_dict(self):
        return {
            "ema_state": self.ema.state_dict(),
            "target_decay": self.target_decay,
            "warmup": self.warmup,
            "num_updates": self.num_updates,
        }

    def load_state_dict(self, state: dict):
        self.ema.load_state_dict(state["ema_state"], strict=False)
        self.target_decay = state.get("target_decay",
                                      state.get("decay", self.target_decay))
        self.warmup = state.get("warmup", self.warmup)
        self.num_updates = state.get("num_updates", 0)


MODEL_SIZE_MAP = {
    "m": {"backbone": "1.0x", "hidden_size": 256},
    "m-1.5x": {"backbone": "1.5x", "hidden_size": 384},
    "m-0.5x": {"backbone": "0.5x", "hidden_size": 128},
}


def _ctc_greedy_decode(log_probs: torch.Tensor, charset: str) -> List[str]:
    """Greedy-decode CTC output into strings.

    Args:
        log_probs: ``(T, N, C)`` CTC log-probabilities.
        charset: character set (index 0 is blank).

    Returns:
        List of decoded strings, one per batch element.
    """
    preds = log_probs.argmax(dim=2).permute(1, 0)  # (N, T)
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
    """Compute character error rate (edit distance / target length)."""
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
    """Compute word error rate."""
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


class Trainer:
    """High-level trainer for FlashOCR.

    Example::

        from flashocr import Trainer

        trainer = Trainer(
            epochs=100,
            batch_size=64,
            model_size="m",
            train_images="data/train",
            train_labels="data/train/labels.tsv",
            val_images="data/val",
            val_labels="data/val/labels.tsv",
            lora=True,
            amp=True,
        )
        trainer.train()
    """

    def __init__(
        self,
        # Basic training
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 0.001,
        workers: int = 4,
        save_dir: str = "workspace/ocr_experiment",
        resume: Optional[str] = None,
        device: str = "cuda",
        warmup_epochs: int = 5,
        patience: int = 50,
        # Model
        model_size: str = "m",
        input_height: int = 32,
        input_width: int = 128,
        charset: str = "0123456789abcdefghijklmnopqrstuvwxyz",
        # Data
        train_images: Optional[str] = None,
        train_labels: Optional[str] = None,
        val_images: Optional[str] = None,
        val_labels: Optional[str] = None,
        # Performance
        amp: bool = False,
        multi_gpu: bool = False,
        grad_accum: int = 1,
        # torchtune optimizations
        activation_checkpointing: bool = False,
        activation_offloading: bool = False,
        optimizer_in_bwd: bool = False,
        use_8bit_optimizer: bool = False,
        compile: bool = False,
        chunked_loss: bool = False,
        # LoRA
        lora: bool = False,
        lora_variant: str = "standard",
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        lora_targets: Optional[List[str]] = None,
        qlora: bool = False,
        qlora_dtype: str = "int8",
        # Config override
        config: Any = None,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.workers = workers
        self.save_dir = save_dir
        self.resume = resume
        self.warmup_epochs = warmup_epochs
        self.patience = patience
        self.model_size = model_size
        self.input_size = (input_height, input_width)
        self.charset = charset
        self.train_images = train_images
        self.train_labels = train_labels
        self.val_images = val_images
        self.val_labels = val_labels
        self.amp = amp
        self.multi_gpu = multi_gpu
        self.grad_accum = max(1, grad_accum)
        self.activation_checkpointing = activation_checkpointing
        self.activation_offloading = activation_offloading
        self.optimizer_in_bwd = optimizer_in_bwd
        self.use_8bit_optimizer = use_8bit_optimizer
        self.compile = compile
        self.chunked_loss = chunked_loss
        self.lora = lora
        self.lora_variant = lora_variant
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_targets = lora_targets or ["backbone", "encoder"]
        self.qlora = qlora
        self.qlora_dtype = qlora_dtype

        self._config = config or get_config(
            model_size=self.model_size,
            charset=self.charset,
            input_height=self.input_size[0],
            input_width=self.input_size[1],
        )
        self._model_cfg = MODEL_SIZE_MAP[self.model_size]

        if torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")
            if device not in ("cpu", ""):
                logger.warning("CUDA unavailable; falling back to CPU.")

        os.makedirs(self.save_dir, exist_ok=True)
        self._logger = _setup_logger("FlashOCR", self.save_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, float]:
        """Run the full training loop. Returns dict with best_cer and best_loss."""
        cfg = self._config
        charset = self.charset

        if self.train_images:
            cfg.data.train_images = self.train_images
        if self.train_labels:
            cfg.data.train_labels = self.train_labels
        if self.val_images:
            cfg.data.val_images = self.val_images
        if self.val_labels:
            cfg.data.val_labels = self.val_labels

        num_classes = len(charset) + 1  # +1 for CTC blank

        self._logger.info("=" * 60)
        self._logger.info("FlashOCR Training")
        self._logger.info("=" * 60)
        self._logger.info(f"Device: {self.device}")
        self._logger.info(f"Model: {self.model_size}, Input: {self.input_size}")
        self._logger.info(f"Epochs: {self.epochs}, Batch: {self.batch_size}, LR: {self.lr}")
        self._logger.info(f"Charset ({len(charset)} chars): {charset[:40]}{'...' if len(charset) > 40 else ''}")

        # Data loaders
        train_loader = create_dataloader(
            img_dir=cfg.data.train_images,
            label_file=cfg.data.train_labels,
            charset=charset,
            batch_size=self.batch_size,
            input_size=self.input_size,
            num_workers=self.workers,
            is_train=True,
        )
        val_loader = create_dataloader(
            img_dir=cfg.data.val_images,
            label_file=cfg.data.val_labels,
            charset=charset,
            batch_size=self.batch_size,
            input_size=self.input_size,
            num_workers=self.workers,
            is_train=False,
        )

        # Build model
        model = build_model(cfg).to(self.device)

        # Apply LoRA / QLoRA
        model = self._apply_lora(model)

        # Load pretrained if specified
        self._load_pretrained(model)

        # AMP
        scaler = None
        if self.amp and self.device.type == "cuda":
            scaler = torch.amp.GradScaler("cuda", enabled=True)
            self._logger.info("AMP enabled")

        # Multi-GPU
        use_multi_gpu = self.multi_gpu and torch.cuda.device_count() > 1
        if use_multi_gpu:
            model = nn.DataParallel(model)

        raw_model = model.module if use_multi_gpu else model

        # torchtune optimizations
        if self.activation_checkpointing:
            try:
                for module in raw_model.modules():
                    if hasattr(module, "gradient_checkpointing"):
                        module.gradient_checkpointing = True
                self._logger.info("Activation checkpointing enabled")
            except Exception as e:
                self._logger.warning(f"Activation checkpointing failed: {e}")

        if self.compile:
            try:
                raw_model = torch.compile(raw_model)
                if not use_multi_gpu:
                    model = raw_model
                self._logger.info("torch.compile enabled")
            except Exception as e:
                self._logger.warning(f"torch.compile failed: {e}")

        # Optimizer
        if self.use_8bit_optimizer:
            try:
                import bitsandbytes as bnb
                optimizer = bnb.optim.AdamW8bit(
                    model.parameters(), lr=self.lr, weight_decay=cfg.train.weight_decay,
                )
                self._logger.info("8-bit AdamW optimizer enabled")
            except ImportError:
                self._logger.warning("bitsandbytes not installed, using standard AdamW")
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=self.lr, weight_decay=cfg.train.weight_decay,
                )
        else:
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=self.lr, weight_decay=cfg.train.weight_decay,
            )

        # LR schedule (cosine with warmup)
        eta_min = 0.00005
        eta_min_factor = eta_min / self.lr

        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return (epoch + 1) / self.warmup_epochs
            progress = (epoch - self.warmup_epochs) / max(self.epochs - self.warmup_epochs, 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return eta_min_factor + (1.0 - eta_min_factor) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # EMA
        ema = ModelEMA(raw_model, decay=0.9998, warmup=2000)

        # CTC loss
        ctc_loss_fn = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

        # Resume
        start_epoch = 0
        best_loss = float("inf")
        best_cer = float("inf")

        if self.resume:
            ckpt = torch.load(self.resume, map_location=self.device, weights_only=False)
            if "model_state_dict" in ckpt:
                raw_model.load_state_dict(ckpt["model_state_dict"], strict=False)
            if "optimizer_state_dict" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                except (ValueError, KeyError):
                    self._logger.warning("Optimizer state skipped (mismatch)")
            if "scheduler_state_dict" in ckpt and scheduler is not None:
                try:
                    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                except (ValueError, KeyError):
                    pass
            start_epoch = ckpt.get("epoch", 0) + 1
            best_loss = ckpt.get("loss", float("inf"))
            best_cer = ckpt.get("metrics", {}).get("cer", float("inf"))
            if "ema_state_dict" in ckpt:
                ema.load_state_dict(ckpt["ema_state_dict"])
            else:
                ema = ModelEMA(raw_model, decay=0.9998, warmup=2000)
            self._logger.info(f"Resumed from epoch {start_epoch}")

        model_config = {
            "charset": charset,
            "input_size": self.input_size,
            "backbone_size": self._model_cfg["backbone"],
            "hidden_size": self._model_cfg["hidden_size"],
            "num_classes": num_classes,
        }

        # Training loop
        self._logger.info("\nStarting training...")
        epochs_without_improvement = 0

        for epoch in range(start_epoch, self.epochs):
            current_lr = optimizer.param_groups[0]["lr"]
            self._logger.info(f"\nEpoch {epoch + 1}/{self.epochs} (lr={current_lr:.6f})")

            train_losses = self._train_one_epoch(
                model, train_loader, optimizer, ctc_loss_fn, charset,
                epoch + 1, ema, scaler,
            )

            # Validate
            if (epoch + 1) % cfg.train.val_interval == 0:
                val_metrics = self._validate(
                    raw_model, val_loader, ctc_loss_fn, charset, ema,
                )
                val_loss = val_metrics["val_loss"]
                cer = val_metrics["cer"]

                if val_loss < best_loss:
                    best_loss = val_loss

                if cer < best_cer:
                    best_cer = cer
                    epochs_without_improvement = 0

                    self._save_checkpoint(
                        raw_model, optimizer, epoch, val_loss,
                        os.path.join(self.save_dir, "checkpoint_best.pth"),
                        scheduler=scheduler, config=model_config,
                        ema=ema, metrics=val_metrics,
                    )
                    self._save_inference_weights(
                        ema.ema,
                        os.path.join(self.save_dir, "model_best_inference.pth"),
                        config=model_config,
                    )
                    self._logger.info(f"  Best model saved (CER: {best_cer:.4f})")
                else:
                    epochs_without_improvement += cfg.train.val_interval

                if self.patience > 0 and epochs_without_improvement >= self.patience:
                    self._logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

            # Save latest
            self._save_checkpoint(
                raw_model, optimizer, epoch, train_losses["loss"],
                os.path.join(self.save_dir, "checkpoint_last.pth"),
                scheduler=scheduler, config=model_config, ema=ema,
            )
            self._save_inference_weights(
                ema.ema,
                os.path.join(self.save_dir, "model_last_inference.pth"),
                config=model_config,
            )

            if scheduler is not None:
                scheduler.step()

        # Final save
        if self.lora or self.qlora:
            lora_path = os.path.join(self.save_dir, "lora_adapters.pth")
            torch.save(get_lora_state_dict(ema.ema), lora_path)
            merge_lora_weights(ema.ema)

        self._save_inference_weights(
            ema.ema,
            os.path.join(self.save_dir, "model_final_inference.pth"),
            config=model_config,
        )
        self._save_inference_weights(
            ema.ema,
            os.path.join(self.save_dir, "model_final_fp16.pth"),
            config=model_config, half=True,
        )

        self._logger.info("=" * 60)
        self._logger.info("Training Complete!")
        self._logger.info(f"Best CER: {best_cer:.4f}  |  Best Loss: {best_loss:.4f}")
        self._logger.info("=" * 60)

        return {"best_cer": best_cer, "best_loss": best_loss}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_lora(self, model: nn.Module) -> nn.Module:
        if self.qlora:
            model = apply_qlora(
                model, rank=self.lora_rank, alpha=self.lora_alpha,
                dropout=self.lora_dropout, target_modules=self.lora_targets,
                quant_dtype=self.qlora_dtype, variant=self.lora_variant,
            )
            self._logger.info(f"QLoRA applied (rank={self.lora_rank})")
        elif self.lora:
            model = apply_lora(
                model, rank=self.lora_rank, alpha=self.lora_alpha,
                dropout=self.lora_dropout, target_modules=self.lora_targets,
                variant=self.lora_variant,
            )
            self._logger.info(f"LoRA applied (rank={self.lora_rank})")
        return model

    def _load_pretrained(self, model: nn.Module):
        if self.resume:
            return
        # No COCO pretrained for OCR, but support loading a custom checkpoint
        finetune = getattr(self._config.train, "resume", None)
        if finetune and os.path.isfile(finetune):
            ckpt = torch.load(finetune, map_location=self.device, weights_only=False)
            src_sd = ckpt.get("model_state_dict", ckpt)
            src_sd = {k: v.float() if v.is_floating_point() else v for k, v in src_sd.items()}
            model.load_state_dict(src_sd, strict=False)
            self._logger.info(f"Pretrained weights loaded from: {finetune}")

    def _train_one_epoch(self, model, dataloader, optimizer, ctc_loss_fn,
                         charset, epoch, ema, scaler):
        model.train()
        use_amp = scaler is not None
        loss_meter = _AverageMeter("Loss")
        raw_model = model.module if hasattr(model, "module") else model

        for batch_idx, batch in enumerate(dataloader):
            images = batch["images"].to(self.device)
            targets = batch["targets"].to(self.device)
            target_lengths = batch["target_lengths"].to(self.device)

            with torch.amp.autocast(self.device.type, enabled=use_amp):
                log_probs = model(images)  # (T, N, C)
                T, N, C = log_probs.shape
                input_lengths = torch.full((N,), T, dtype=torch.long, device=self.device)
                loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)
                loss = loss / self.grad_accum

            if torch.isnan(loss):
                continue

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % self.grad_accum == 0 or (batch_idx + 1) == len(dataloader):
                if scaler:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(raw_model.parameters(), 5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(raw_model.parameters(), 5.0)
                    optimizer.step()
                optimizer.zero_grad()

                if ema is not None:
                    ema.update(raw_model)

            loss_meter.update(loss.item() * self.grad_accum)

            if (batch_idx + 1) % 10 == 0:
                self._logger.info(
                    f"  [{batch_idx+1}/{len(dataloader)}] Loss: {loss_meter.avg:.4f}"
                )

        return {"loss": loss_meter.avg}

    @torch.no_grad()
    def _validate(self, model, dataloader, ctc_loss_fn, charset, ema):
        eval_model = ema.ema if ema is not None else model
        eval_model.eval()

        loss_meter = _AverageMeter("Loss")
        total_cer = 0.0
        total_wer = 0.0
        total_correct = 0
        total_samples = 0

        for batch in dataloader:
            images = batch["images"].to(self.device)
            targets = batch["targets"].to(self.device)
            target_lengths = batch["target_lengths"].to(self.device)
            raw_labels = batch["raw_labels"]

            log_probs = eval_model(images)
            T, N, C = log_probs.shape
            input_lengths = torch.full((N,), T, dtype=torch.long, device=self.device)
            loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)
            loss_meter.update(loss.item())

            pred_texts = _ctc_greedy_decode(log_probs, charset)

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

        self._logger.info(
            f"  Val Loss: {loss_meter.avg:.4f} | CER: {cer:.4f} | "
            f"WER: {wer:.4f} | Acc: {accuracy:.4f}"
        )

        model.train()
        return {
            "val_loss": loss_meter.avg,
            "cer": cer,
            "wer": wer,
            "accuracy": accuracy,
        }

    @staticmethod
    def _save_checkpoint(model, optimizer, epoch, loss, save_path,
                         scheduler=None, config=None, ema=None, metrics=None):
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "metrics": metrics or {},
        }
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        if ema is not None:
            checkpoint["ema_state_dict"] = ema.state_dict()
        if config is not None:
            checkpoint["config"] = config
        torch.save(checkpoint, save_path)

    @staticmethod
    def _save_inference_weights(model, save_path, config=None, half=False):
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        state_dict = model.state_dict()
        if half:
            state_dict = {
                k: v.half() if v.dtype == torch.float32 else v
                for k, v in state_dict.items()
            }
        checkpoint = {"model_state_dict": state_dict, "half": half}
        if config is not None:
            checkpoint["config"] = config
        torch.save(checkpoint, save_path)
