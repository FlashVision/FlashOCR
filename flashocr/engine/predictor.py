"""FlashOCR Predictor — inference on images for text recognition."""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from flashocr.cfg import get_config
from flashocr.models import FlashOCR
from flashocr.data.transforms import InferenceTransform

logger = logging.getLogger(__name__)


def _ctc_greedy_decode(log_probs: torch.Tensor, charset: str) -> List[Tuple[str, float]]:
    """Greedy-decode CTC output into (text, confidence) pairs.

    Args:
        log_probs: ``(T, N, C)`` CTC log-probabilities.
        charset: character set (index 0 is CTC blank).

    Returns:
        List of ``(decoded_text, confidence)`` tuples, one per batch element.
    """
    probs = log_probs.exp()
    preds = log_probs.argmax(dim=2).permute(1, 0)  # (N, T)
    max_probs = probs.max(dim=2).values.permute(1, 0)  # (N, T)

    results = []
    for seq, seq_probs in zip(preds, max_probs):
        chars = []
        char_confs = []
        prev = -1
        for idx, prob in zip(seq.tolist(), seq_probs.tolist()):
            if idx != 0 and idx != prev:
                if 1 <= idx <= len(charset):
                    chars.append(charset[idx - 1])
                    char_confs.append(prob)
            prev = idx

        text = "".join(chars)
        confidence = float(np.mean(char_confs)) if char_confs else 0.0
        results.append((text, confidence))

    return results


class Predictor:
    """High-level inference wrapper for FlashOCR.

    Example::

        from flashocr import Predictor

        pred = Predictor(model_path="workspace/model_best_inference.pth")
        text, conf = pred.recognize(cv2.imread("word.jpg"))
        print(f"{text} ({conf:.2f})")
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        charset: Optional[str] = None,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        cfg = get_config()
        backbone_size = cfg.model.backbone_size
        hidden_size = cfg.model.hidden_size
        input_size = cfg.model.input_size
        model_charset = cfg.model.charset

        if "config" in checkpoint:
            ckpt_cfg = checkpoint["config"]
            backbone_size = ckpt_cfg.get("backbone_size", backbone_size)
            hidden_size = ckpt_cfg.get("hidden_size", hidden_size)
            input_size = ckpt_cfg.get("input_size", input_size)
            if "charset" in ckpt_cfg and ckpt_cfg["charset"]:
                model_charset = ckpt_cfg["charset"]

        self.charset = charset or model_charset
        self.input_size = input_size if isinstance(input_size, tuple) else (input_size, input_size)
        num_classes = len(self.charset) + 1

        self.model = FlashOCR(
            num_classes=num_classes,
            input_size=self.input_size,
            backbone_size=backbone_size,
            hidden_size=hidden_size,
        )

        if "model_state_dict" in checkpoint:
            sd = checkpoint["model_state_dict"]
            if checkpoint.get("half", False):
                sd = {k: v.float() if v.is_floating_point() else v for k, v in sd.items()}
            self.model.load_state_dict(sd, strict=False)
        elif "state_dict" in checkpoint:
            sd = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
            self.model.load_state_dict(sd, strict=False)
        else:
            self.model.load_state_dict(checkpoint, strict=False)

        self.model = self.model.to(self.device).eval()
        self.transform = InferenceTransform(input_size=self.input_size)

        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model loaded: {total_params:,} params, charset={len(self.charset)} chars")

    @torch.no_grad()
    def recognize(self, image: np.ndarray) -> Tuple[str, float]:
        """Recognize text in a single cropped word image.

        Args:
            image: BGR image (numpy array) containing a single word/text line.

        Returns:
            ``(text, confidence)`` tuple.
        """
        if image is None or image.size == 0:
            return "", 0.0

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image
        tensor = self.transform(rgb)
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
        tensor = tensor.unsqueeze(0).to(self.device)

        log_probs = self.model(tensor)  # (T, 1, C)
        results = _ctc_greedy_decode(log_probs, self.charset)
        return results[0]

    def predict(
        self,
        source: Union[str, np.ndarray, Path],
        output_dir: Optional[str] = None,
    ) -> Union[Tuple[str, float], List[Dict]]:
        """Run recognition on an image path, numpy array, or directory.

        Args:
            source: Path to an image, a directory of images, or a BGR numpy array.
            output_dir: If set, saves results to a text file here.

        Returns:
            For single image: ``(text, confidence)``.
            For directory: list of dicts ``{"file", "text", "confidence"}``.
        """
        if isinstance(source, np.ndarray):
            return self.recognize(source)

        source = str(source)
        if os.path.isdir(source):
            return self.predict_directory(source, output_dir)
        return self.predict_image(source, output_dir)

    def predict_image(
        self,
        image_path: str,
        output_dir: Optional[str] = None,
    ) -> Tuple[str, float]:
        """Recognize text from an image file.

        Args:
            image_path: Path to the image.
            output_dir: If set, appends result to a text file.

        Returns:
            ``(text, confidence)`` tuple.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        text, confidence = self.recognize(image)

        logger.info(f"{Path(image_path).name}: '{text}' ({confidence:.3f})")

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            results_path = os.path.join(output_dir, "results.txt")
            with open(results_path, "a", encoding="utf-8") as f:
                f.write(f"{Path(image_path).name}\t{text}\t{confidence:.4f}\n")

        return text, confidence

    def predict_directory(
        self,
        dir_path: str,
        output_dir: Optional[str] = None,
    ) -> List[Dict]:
        """Recognize text from all images in a directory.

        Args:
            dir_path: Path to directory of images.
            output_dir: If set, saves results to ``results.txt``.

        Returns:
            List of dicts with ``file``, ``text``, ``confidence`` keys.
        """
        results = []
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

        paths = sorted(Path(dir_path).iterdir())
        for path in paths:
            if path.suffix.lower() not in image_extensions:
                continue
            try:
                text, confidence = self.predict_image(str(path), output_dir)
                results.append({
                    "file": str(path),
                    "text": text,
                    "confidence": confidence,
                })
            except Exception as e:
                logger.warning(f"Failed to process {path.name}: {e}")

        logger.info(f"Processed {len(results)} images from {dir_path}")
        return results
