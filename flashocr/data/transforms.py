"""
OCR-specific image transforms for text recognition.

Images are resized to a fixed height (default 32 px) with proportional width,
then right-padded to the target width.
"""

import random
from typing import Dict, Tuple

import numpy as np
import torch
from PIL import Image, ImageFilter


class _ResizePad:
    """Resize an image to a fixed height, keeping aspect ratio, then pad to
    target width."""

    def __init__(self, target_h: int, target_w: int):
        self.target_h = target_h
        self.target_w = target_w

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        ratio = self.target_h / h
        new_w = min(int(w * ratio), self.target_w)
        img = img.resize((new_w, self.target_h), Image.BILINEAR)

        padded = Image.new("RGB", (self.target_w, self.target_h), (0, 0, 0))
        padded.paste(img, (0, 0))
        return padded


class _ToTensorNormalize:
    """Convert PIL image to float tensor and normalise to [-1, 1]."""

    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - np.array(self.MEAN, dtype=np.float32)) / np.array(
            self.STD, dtype=np.float32
        )
        tensor = torch.from_numpy(arr.transpose(2, 0, 1))
        return tensor


class TrainTransform:
    """Training augmentation pipeline for OCR.

    Applies random rotation, brightness/contrast jitter, Gaussian noise,
    then resize-pad and normalise.

    Args:
        input_size: ``(height, width)`` target size (default ``(32, 128)``).
        max_rotation: Maximum rotation in degrees (default ``5``).
        noise_std: Standard deviation of additive Gaussian noise (default ``0.02``).
        brightness_range: ``(low, high)`` multiplicative brightness factor.
        contrast_range: ``(low, high)`` multiplicative contrast factor.
    """

    def __init__(
        self,
        input_size: Tuple[int, int] = (32, 128),
        max_rotation: float = 5.0,
        noise_std: float = 0.02,
        brightness_range: Tuple[float, float] = (0.8, 1.2),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
    ):
        self.target_h, self.target_w = input_size
        self.max_rotation = max_rotation
        self.noise_std = noise_std
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self._resize_pad = _ResizePad(self.target_h, self.target_w)
        self._to_tensor = _ToTensorNormalize()

    def __call__(self, img: Image.Image) -> torch.Tensor:
        # Random rotation
        if self.max_rotation > 0:
            angle = random.uniform(-self.max_rotation, self.max_rotation)
            img = img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=(0, 0, 0))

        # Random brightness
        factor = random.uniform(*self.brightness_range)
        from PIL import ImageEnhance
        img = ImageEnhance.Brightness(img).enhance(factor)

        # Random contrast
        factor = random.uniform(*self.contrast_range)
        img = ImageEnhance.Contrast(img).enhance(factor)

        # Gaussian blur (small probability)
        if random.random() < 0.15:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))

        # Resize and pad
        img = self._resize_pad(img)

        # To tensor + normalise
        tensor = self._to_tensor(img)

        # Additive Gaussian noise
        if self.noise_std > 0:
            noise = torch.randn_like(tensor) * self.noise_std
            tensor = tensor + noise

        return tensor


class ValTransform:
    """Validation / test transform: resize-pad + normalise only.

    Args:
        input_size: ``(height, width)`` target size.
    """

    def __init__(self, input_size: Tuple[int, int] = (32, 128)):
        self.target_h, self.target_w = input_size
        self._resize_pad = _ResizePad(self.target_h, self.target_w)
        self._to_tensor = _ToTensorNormalize()

    def __call__(self, img: Image.Image) -> torch.Tensor:
        img = self._resize_pad(img)
        return self._to_tensor(img)


class InferenceTransform:
    """Inference transform that also returns metadata about the original image.

    Args:
        input_size: ``(height, width)`` target size.

    Returns:
        ``(tensor, meta)`` where *meta* is a dict with ``original_size``
        ``(w, h)`` and ``scale_ratio``.
    """

    def __init__(self, input_size: Tuple[int, int] = (32, 128)):
        self.target_h, self.target_w = input_size
        self._resize_pad = _ResizePad(self.target_h, self.target_w)
        self._to_tensor = _ToTensorNormalize()

    def __call__(self, img: Image.Image) -> Tuple[torch.Tensor, Dict]:
        orig_w, orig_h = img.size
        ratio = self.target_h / orig_h
        new_w = min(int(orig_w * ratio), self.target_w)

        meta = {
            "original_size": (orig_w, orig_h),
            "scale_ratio": ratio,
            "resized_width": new_w,
        }

        img = self._resize_pad(img)
        tensor = self._to_tensor(img)
        return tensor, meta
