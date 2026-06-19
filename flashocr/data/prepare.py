"""
Data format conversion utilities for OCR datasets.

Supports conversion from common OCR dataset formats to the simple TSV format
expected by :class:`OCRDataset`:

    image_filename<TAB>text_label

Supported source formats:
  - **ICDAR**: ``gt_*.txt`` files with ``x1,y1,x2,y2,...,text`` per line.
  - **LMDB**: MDB databases with ``image-{key}`` / ``label-{key}`` pairs
    (as used by many STR benchmarks).
"""

import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def detect_dataset_format(data_dir: str) -> str:
    """Auto-detect dataset format in *data_dir*.

    Returns:
        One of ``"icdar"``, ``"lmdb"``, ``"tsv"``, or ``"unknown"``.
    """
    data_dir = Path(data_dir)

    if list(data_dir.glob("gt_*.txt")):
        return "icdar"

    if (data_dir / "data.mdb").exists():
        return "lmdb"

    tsv_files = list(data_dir.glob("*.tsv")) + list(data_dir.glob("labels.txt"))
    if tsv_files:
        return "tsv"

    return "unknown"


def convert_icdar_to_tsv(
    data_dir: str,
    output_file: str,
    img_dir: Optional[str] = None,
    encoding: str = "utf-8-sig",
) -> int:
    """Convert ICDAR ground-truth files to TSV format.

    ICDAR annotation lines are typically::

        x1,y1,x2,y2,x3,y3,x4,y4,text

    We extract only the text portion and pair it with the corresponding word
    image.  If *img_dir* is provided we look for pre-cropped word images
    there; otherwise we expect images named ``word_NNNN.png`` alongside the
    GT files.

    Args:
        data_dir: Directory with ``gt_*.txt`` files.
        output_file: Destination TSV path.
        img_dir: Directory with cropped word images (optional).
        encoding: Text encoding for ICDAR files.

    Returns:
        Number of samples written.
    """
    data_dir = Path(data_dir)
    if img_dir is None:
        img_dir = data_dir

    gt_files = sorted(data_dir.glob("gt_*.txt"))
    if not gt_files:
        raise FileNotFoundError(f"No gt_*.txt files found in {data_dir}")

    samples = []
    for gt_path in gt_files:
        stem = gt_path.stem.replace("gt_", "")
        with open(gt_path, "r", encoding=encoding) as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 9:
                    continue
                text = ",".join(parts[8:]).strip()
                if text.startswith('"') and text.endswith('"'):
                    text = text[1:-1]
                text = text.strip()
                if not text or text == "###":
                    continue

                img_name = f"{stem}_word_{idx:04d}.png"
                img_path = Path(img_dir) / img_name
                if not img_path.exists():
                    img_name = f"{stem}_{idx}.png"
                    img_path = Path(img_dir) / img_name
                    if not img_path.exists():
                        logger.debug("Image not found for %s line %d, skipping", gt_path.name, idx)
                        continue

                samples.append((img_name, text))

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for img_name, text in samples:
            f.write(f"{img_name}\t{text}\n")

    logger.info("ICDAR → TSV: %d samples written to %s", len(samples), output_file)
    return len(samples)


def convert_lmdb_to_tsv(
    lmdb_dir: str,
    output_file: str,
    img_output_dir: str,
    max_samples: Optional[int] = None,
) -> int:
    """Convert an LMDB OCR dataset to TSV + individual image files.

    Many scene-text recognition benchmarks (MJSynth, SynthText, etc.) ship
    as LMDB databases.  This function extracts images and labels into the
    flat format expected by :class:`OCRDataset`.

    Args:
        lmdb_dir: Path to directory containing ``data.mdb``.
        output_file: Destination TSV path.
        img_output_dir: Directory to write extracted images.
        max_samples: Optional cap on the number of samples to extract.

    Returns:
        Number of samples written.
    """
    try:
        import lmdb
    except ImportError:
        raise ImportError(
            "lmdb package is required for LMDB conversion. "
            "Install with: pip install lmdb"
        )

    os.makedirs(img_output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    env = lmdb.open(lmdb_dir, readonly=True, lock=False, readahead=False)
    with env.begin(write=False) as txn:
        num_samples_raw = txn.get(b"num-samples")
        if num_samples_raw is None:
            raise ValueError("LMDB dataset missing 'num-samples' key")
        num_samples = int(num_samples_raw.decode())

        if max_samples is not None:
            num_samples = min(num_samples, max_samples)

        written = 0
        with open(output_file, "w", encoding="utf-8") as f_out:
            for i in range(1, num_samples + 1):
                img_key = f"image-{i:09d}".encode()
                label_key = f"label-{i:09d}".encode()

                img_data = txn.get(img_key)
                label_data = txn.get(label_key)
                if img_data is None or label_data is None:
                    continue

                label = label_data.decode("utf-8").strip()
                if not label:
                    continue

                img_name = f"{i:09d}.png"
                img_path = os.path.join(img_output_dir, img_name)

                with open(img_path, "wb") as f_img:
                    f_img.write(img_data)

                f_out.write(f"{img_name}\t{label}\n")
                written += 1

    env.close()
    logger.info("LMDB → TSV: %d samples written to %s", written, output_file)
    return written


def verify_dataset(
    img_dir: str,
    label_file: str,
    charset: Optional[str] = None,
    check_images: bool = True,
) -> dict:
    """Verify that a dataset directory is valid and report statistics.

    Args:
        img_dir: Image directory.
        label_file: TSV label file path.
        charset: If provided, report characters that appear in labels but
            are absent from the charset.
        check_images: If ``True``, verify that every referenced image exists.

    Returns:
        Dict with ``total``, ``valid``, ``missing_images``,
        ``oov_characters``, ``max_label_length``, ``charset_coverage``.
    """
    if not os.path.isfile(label_file):
        raise FileNotFoundError(f"Label file not found: {label_file}")

    total = 0
    valid = 0
    missing = 0
    max_len = 0
    all_chars: set = set()
    missing_files: list = []

    with open(label_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if not line:
                continue
            parts = line.split("\t", maxsplit=1)
            if len(parts) != 2:
                continue
            img_name, text = parts
            total += 1
            all_chars.update(text)
            max_len = max(max_len, len(text))

            if check_images:
                img_path = os.path.join(img_dir, img_name)
                if os.path.isfile(img_path):
                    valid += 1
                else:
                    missing += 1
                    if len(missing_files) < 10:
                        missing_files.append(img_name)
            else:
                valid += 1

    oov = set()
    if charset is not None:
        charset_set = set(charset)
        oov = all_chars - charset_set

    result = {
        "total": total,
        "valid": valid,
        "missing_images": missing,
        "missing_files_sample": missing_files,
        "unique_characters": len(all_chars),
        "max_label_length": max_len,
        "oov_characters": sorted(oov) if oov else [],
        "charset_coverage": (
            1.0 - len(oov) / max(len(all_chars), 1) if charset else None
        ),
    }

    logger.info(
        "Dataset verify: %d total, %d valid, %d missing images, "
        "max label len %d, %d unique chars",
        total, valid, missing, max_len, len(all_chars),
    )
    if oov:
        logger.warning("OOV characters not in charset: %s", oov)

    return result
