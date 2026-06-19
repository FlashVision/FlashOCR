"""
OCR dataset for text recognition training and evaluation.

Expects a TSV label file with one sample per line:
    image_filename<TAB>text_label
"""

import os
import logging
from typing import Dict, List, Optional, Callable, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image

logger = logging.getLogger(__name__)


class OCRDataset(Dataset):
    """Dataset for OCR text recognition.

    Args:
        img_dir: Root directory containing the images.
        label_file: Path to a TSV file with ``image_filename\\ttext_label`` per
            line.
        charset: String of all characters the model can predict.  Characters
            are mapped to indices ``1..len(charset)``; index ``0`` is reserved
            for the CTC blank / padding token.
        transform: Callable that preprocesses a PIL image.
        max_label_length: Maximum number of characters in a label.  Longer
            labels are silently truncated.
    """

    BLANK_IDX = 0

    def __init__(
        self,
        img_dir: str,
        label_file: str,
        charset: str,
        transform: Optional[Callable] = None,
        max_label_length: int = 25,
    ):
        self.img_dir = img_dir
        self.charset = charset
        self.transform = transform
        self.max_label_length = max_label_length

        self.char2idx = {c: i + 1 for i, c in enumerate(charset)}
        self.idx2char = {i + 1: c for i, c in enumerate(charset)}

        self.samples: List[Tuple[str, str]] = []
        self._load_labels(label_file)
        logger.info(
            "OCRDataset: %d samples from %s (charset size %d, max_len %d)",
            len(self.samples), label_file, len(charset), max_label_length,
        )

    def _load_labels(self, label_file: str):
        with open(label_file, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.rstrip("\n\r")
                if not line:
                    continue
                parts = line.split("\t", maxsplit=1)
                if len(parts) != 2:
                    logger.warning("Skipping malformed line %d: %r", lineno, line)
                    continue
                img_name, text = parts
                if not text:
                    continue
                self.samples.append((img_name, text))

    def encode(self, text: str) -> List[int]:
        """Encode *text* into a list of integer indices using the charset."""
        text = text[: self.max_label_length]
        return [self.char2idx.get(c, self.BLANK_IDX) for c in text]

    def decode(self, indices) -> str:
        """Decode integer indices back into a string, stopping at blank/pad."""
        chars = []
        for idx in indices:
            idx = int(idx)
            if idx == self.BLANK_IDX:
                continue
            if idx in self.idx2char:
                chars.append(self.idx2char[idx])
        return "".join(chars)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict]:
        img_name, text = self.samples[index]
        img_path = os.path.join(self.img_dir, img_name)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.error("Failed to load %s: %s", img_path, e)
            image = Image.new("RGB", (128, 32), color=(128, 128, 128))
            text = ""

        if self.transform is not None:
            result = self.transform(image)
            if isinstance(result, tuple):
                image_tensor, _ = result
            else:
                image_tensor = result
        else:
            from torchvision import transforms as T
            image_tensor = T.ToTensor()(image)

        encoded = self.encode(text)

        target = {
            "labels": torch.tensor(encoded, dtype=torch.long),
            "label_lengths": torch.tensor(len(encoded), dtype=torch.long),
            "texts": text,
        }
        return image_tensor, target


def collate_fn(batch: List[Tuple[torch.Tensor, Dict]]) -> Tuple[torch.Tensor, Dict]:
    """Collate variable-length OCR labels into a padded batch.

    Returns:
        images: ``[B, C, H, W]`` tensor.
        targets: Dict with
            - ``labels``: ``[B, max_len]`` padded with ``0``.
            - ``label_lengths``: ``[B]`` true label lengths.
            - ``texts``: List of raw strings.
    """
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)

    labels_list = [t["labels"] for t in targets]
    lengths = torch.stack([t["label_lengths"] for t in targets])
    texts = [t["texts"] for t in targets]

    max_len = max(lab.size(0) for lab in labels_list)
    padded = torch.zeros(len(labels_list), max_len, dtype=torch.long)
    for i, lab in enumerate(labels_list):
        padded[i, : lab.size(0)] = lab

    return images, {
        "labels": padded,
        "label_lengths": lengths,
        "texts": texts,
    }
