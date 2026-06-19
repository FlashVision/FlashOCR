"""
DataLoader utilities for OCR text recognition.
"""

import logging
from typing import Tuple

import torch
from torch.utils.data import DataLoader

from .dataset import OCRDataset, collate_fn
from .transforms import TrainTransform, ValTransform

logger = logging.getLogger(__name__)


def create_dataloader(
    img_dir: str,
    label_file: str,
    charset: str,
    batch_size: int = 64,
    input_size: Tuple[int, int] = (32, 128),
    num_workers: int = 4,
    is_train: bool = True,
    shuffle: bool = None,
    max_label_length: int = 25,
) -> DataLoader:
    """Create a DataLoader for OCR recognition.

    Args:
        img_dir: Directory containing images.
        label_file: Path to TSV label file.
        charset: Character set string.
        batch_size: Batch size.
        input_size: Target image size ``(height, width)``.
        num_workers: Number of data loading workers.
        is_train: Whether this is training data.
        shuffle: Whether to shuffle (defaults to *is_train*).
        max_label_length: Maximum label length.

    Returns:
        DataLoader instance.
    """
    if shuffle is None:
        shuffle = is_train

    if is_train:
        transform = TrainTransform(input_size=input_size)
    else:
        transform = ValTransform(input_size=input_size)

    dataset = OCRDataset(
        img_dir=img_dir,
        label_file=label_file,
        charset=charset,
        transform=transform,
        max_label_length=max_label_length,
    )

    pin = torch.cuda.is_available()

    effective_workers = num_workers
    if num_workers > 0:
        try:
            import multiprocessing
            _lock = multiprocessing.Lock()
            del _lock
        except (PermissionError, OSError, RuntimeError) as e:
            logger.warning(
                "Multiprocessing unavailable (%s), falling back to num_workers=0", e
            )
            effective_workers = 0

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=effective_workers,
        pin_memory=pin and effective_workers > 0,
        collate_fn=collate_fn,
        drop_last=is_train,
    )

    return dataloader


def create_train_val_loaders(
    train_img_dir: str,
    train_label_file: str,
    val_img_dir: str,
    val_label_file: str,
    charset: str,
    batch_size: int = 64,
    input_size: Tuple[int, int] = (32, 128),
    num_workers: int = 4,
    max_label_length: int = 25,
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation DataLoaders.

    Args:
        train_img_dir: Training images directory.
        train_label_file: Training TSV label file.
        val_img_dir: Validation images directory.
        val_label_file: Validation TSV label file.
        charset: Character set string.
        batch_size: Batch size.
        input_size: Target image size ``(height, width)``.
        num_workers: Number of workers.
        max_label_length: Maximum label length.

    Returns:
        ``(train_loader, val_loader)``
    """
    train_loader = create_dataloader(
        img_dir=train_img_dir,
        label_file=train_label_file,
        charset=charset,
        batch_size=batch_size,
        input_size=input_size,
        num_workers=num_workers,
        is_train=True,
        max_label_length=max_label_length,
    )

    val_loader = create_dataloader(
        img_dir=val_img_dir,
        label_file=val_label_file,
        charset=charset,
        batch_size=batch_size,
        input_size=input_size,
        num_workers=num_workers,
        is_train=False,
        max_label_length=max_label_length,
    )

    return train_loader, val_loader
