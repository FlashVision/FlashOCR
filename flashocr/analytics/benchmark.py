"""Benchmark — measure FlashOCR model inference speed and resource usage."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


class Benchmark:
    """Benchmark FlashOCR model speed and resource usage.

    Parameters
    ----------
    model_path : str | Path
        Path to a saved FlashOCR checkpoint (``.pth`` / ``.onnx``).
    device : str
        ``"cuda"`` or ``"cpu"``.
    input_size : tuple[int, int]
        Network input resolution ``(height, width)``.
    num_warmup : int
        Number of warmup iterations before timing.
    num_runs : int
        Number of timed iterations.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        device: str = "cuda",
        input_size: Union[tuple, List[int]] = (32, 128),
        num_warmup: int = 10,
        num_runs: int = 100,
    ):
        self.model_path = Path(model_path)
        self.device = device
        self.input_size = tuple(input_size)
        self.num_warmup = num_warmup
        self.num_runs = num_runs

        self._model: Optional[Any] = None
        self._is_onnx: bool = self.model_path.suffix.lower() == ".onnx"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, float]:
        """Run a speed benchmark.

        Returns
        -------
        dict
            ``{"avg_ms": …, "fps": …, "min_ms": …, "max_ms": …, "std_ms": …,
              "params": …, "model_size_mb": …}``
        """
        model = self._load_model()
        dummy = self._make_dummy_input()

        if self._is_onnx:
            return self._bench_onnx(model, dummy)
        return self._bench_pytorch(model, dummy)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self):
        if self._model is not None:
            return self._model

        if self._is_onnx:
            import onnxruntime as ort
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if self.device == "cuda"
                else ["CPUExecutionProvider"]
            )
            self._model = ort.InferenceSession(str(self.model_path), providers=providers)
        else:
            import torch
            from flashocr.cfg import get_config
            from flashocr.models import FlashOCR

            checkpoint = torch.load(str(self.model_path), map_location=self.device, weights_only=False)
            cfg = get_config()

            backbone_size = cfg.model.backbone_size
            hidden_size = cfg.model.hidden_size
            charset = cfg.model.charset

            if "config" in checkpoint:
                ckpt_cfg = checkpoint["config"]
                backbone_size = ckpt_cfg.get("backbone_size", backbone_size)
                hidden_size = ckpt_cfg.get("hidden_size", hidden_size)
                if "charset" in ckpt_cfg:
                    charset = ckpt_cfg["charset"]

            num_classes = len(charset) + 1
            model = FlashOCR(
                num_classes=num_classes,
                input_size=self.input_size,
                backbone_size=backbone_size,
                hidden_size=hidden_size,
            )

            if "model_state_dict" in checkpoint:
                sd = checkpoint["model_state_dict"]
                if checkpoint.get("half", False):
                    sd = {k: v.float() if v.is_floating_point() else v for k, v in sd.items()}
                model.load_state_dict(sd, strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)

            self._model = model.to(self.device).eval()

        return self._model

    def _make_dummy_input(self) -> np.ndarray:
        return np.random.rand(1, 3, *self.input_size).astype(np.float32)

    def _bench_pytorch(self, model, dummy: np.ndarray) -> Dict[str, float]:
        import torch

        tensor = torch.from_numpy(dummy).to(self.device)

        with torch.no_grad():
            for _ in range(self.num_warmup):
                model(tensor)
        if self.device == "cuda":
            torch.cuda.synchronize()

        latencies = []
        with torch.no_grad():
            for _ in range(self.num_runs):
                if self.device == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                model(tensor)
                if self.device == "cuda":
                    torch.cuda.synchronize()
                latencies.append((time.perf_counter() - start) * 1000)

        latencies_np = np.array(latencies)
        params = sum(p.numel() for p in model.parameters()) if hasattr(model, "parameters") else 0
        size_mb = self.model_path.stat().st_size / (1024 * 1024)

        return {
            "avg_ms": round(float(latencies_np.mean()), 3),
            "fps": round(1000.0 / float(latencies_np.mean()), 2),
            "min_ms": round(float(latencies_np.min()), 3),
            "max_ms": round(float(latencies_np.max()), 3),
            "std_ms": round(float(latencies_np.std()), 3),
            "params": params,
            "model_size_mb": round(size_mb, 2),
        }

    def _bench_onnx(self, session, dummy: np.ndarray) -> Dict[str, float]:
        input_name = session.get_inputs()[0].name

        for _ in range(self.num_warmup):
            session.run(None, {input_name: dummy})

        latencies = []
        for _ in range(self.num_runs):
            start = time.perf_counter()
            session.run(None, {input_name: dummy})
            latencies.append((time.perf_counter() - start) * 1000)

        latencies_np = np.array(latencies)
        size_mb = self.model_path.stat().st_size / (1024 * 1024)

        return {
            "avg_ms": round(float(latencies_np.mean()), 3),
            "fps": round(1000.0 / float(latencies_np.mean()), 2),
            "min_ms": round(float(latencies_np.min()), 3),
            "max_ms": round(float(latencies_np.max()), 3),
            "std_ms": round(float(latencies_np.std()), 3),
            "params": 0,
            "model_size_mb": round(size_mb, 2),
        }
