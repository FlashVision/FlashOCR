"""Table extraction from document images.

Provides both line-based (classical) and deep-learning-based table detection,
cell extraction, and structure recognition.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class TableCell:
    """A single cell in a detected table."""

    row: int
    col: int
    row_span: int = 1
    col_span: int = 1
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x1, y1, x2, y2
    text: str = ""
    confidence: float = 0.0


@dataclass
class Table:
    """A detected table with structure and content."""

    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    num_rows: int = 0
    num_cols: int = 0
    cells: List[TableCell] = field(default_factory=list)
    confidence: float = 0.0

    def to_grid(self) -> List[List[str]]:
        """Convert to a 2D text grid."""
        grid = [["" for _ in range(self.num_cols)] for _ in range(self.num_rows)]
        for cell in self.cells:
            if 0 <= cell.row < self.num_rows and 0 <= cell.col < self.num_cols:
                grid[cell.row][cell.col] = cell.text
        return grid

    def to_html(self) -> str:
        """Render as an HTML table."""
        grid = self.to_grid()
        rows_html = []
        for row in grid:
            cells_html = "".join(f"<td>{c}</td>" for c in row)
            rows_html.append(f"<tr>{cells_html}</tr>")
        return f"<table>{''.join(rows_html)}</table>"


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.skip = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride, bias=False), nn.BatchNorm2d(out_ch))
            if in_ch != out_ch or stride != 1
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.skip(x)


class TableDetectionNet(nn.Module):
    """Lightweight CNN for table region detection.

    Outputs a binary segmentation mask where 1 = table region.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.encoder = nn.Sequential(
            _ConvBlock(in_channels, c, stride=2),
            _ConvBlock(c, c * 2, stride=2),
            _ConvBlock(c * 2, c * 4, stride=2),
            _ConvBlock(c * 4, c * 8, stride=2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(c * 8, c * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c * 4, c * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c * 2, c, 4, 2, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c, 1, 4, 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        mask = self.decoder(h)
        mask = F.interpolate(mask, size=x.shape[2:], mode="bilinear", align_corners=False)
        return mask


class TableStructureNet(nn.Module):
    """Predicts row/column separators for a table region.

    Two output heads: horizontal separators (rows) and vertical separators (cols).
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.backbone = nn.Sequential(
            _ConvBlock(in_channels, c, stride=2),
            _ConvBlock(c, c * 2, stride=2),
            _ConvBlock(c * 2, c * 4, stride=2),
        )
        self.row_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, 1)),
        )
        self.col_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),
        )
        self.row_proj = nn.Conv2d(c * 4, 1, 1)
        self.col_proj = nn.Conv2d(c * 4, 1, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(x)
        row_feat = self.row_head(feat)  # (B, C, H', 1)
        col_feat = self.col_head(feat)  # (B, C, 1, W')

        row_logits = self.row_proj(row_feat).squeeze(1).squeeze(-1)  # (B, H')
        col_logits = self.col_proj(col_feat).squeeze(1).squeeze(-2)  # (B, W')
        return row_logits, col_logits


def _line_based_detect_lines(
    gray: np.ndarray,
    min_line_length: int = 50,
    max_line_gap: int = 10,
) -> Tuple[List[Tuple], List[Tuple]]:
    """Detect horizontal and vertical lines using morphological operations.

    Args:
        gray: Grayscale image as uint8 numpy array (H, W).
        min_line_length: Minimum pixel length to count as a line.
        max_line_gap: Maximum gap between line segments to merge.

    Returns:
        (horizontal_lines, vertical_lines) each as lists of (x1, y1, x2, y2).
    """
    _, binary = _threshold_image(gray)

    h, w = binary.shape
    h_kernel_len = max(w // 30, min_line_length // 2)
    v_kernel_len = max(h // 30, min_line_length // 2)

    h_kernel = np.zeros((1, h_kernel_len), dtype=np.uint8)
    h_kernel[0, :] = 1
    h_lines_img = _morph_open(binary, h_kernel)

    v_kernel = np.zeros((v_kernel_len, 1), dtype=np.uint8)
    v_kernel[:, 0] = 1
    v_lines_img = _morph_open(binary, v_kernel)

    h_lines = _extract_line_segments(h_lines_img, axis="horizontal", min_length=min_line_length)
    v_lines = _extract_line_segments(v_lines_img, axis="vertical", min_length=min_line_length)

    h_lines = _merge_close_lines(h_lines, max_gap=max_line_gap, axis="horizontal")
    v_lines = _merge_close_lines(v_lines, max_gap=max_line_gap, axis="vertical")

    return h_lines, v_lines


def _threshold_image(gray: np.ndarray) -> Tuple[float, np.ndarray]:
    threshold = np.mean(gray) * 0.7
    binary = (gray < threshold).astype(np.uint8)
    return threshold, binary


def _morph_open(binary: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    from scipy.ndimage import binary_erosion, binary_dilation
    eroded = binary_erosion(binary, structure=kernel).astype(np.uint8)
    dilated = binary_dilation(eroded, structure=kernel).astype(np.uint8)
    return dilated


def _extract_line_segments(
    line_img: np.ndarray,
    axis: str,
    min_length: int,
) -> List[Tuple[int, int, int, int]]:
    """Extract line segments from a binary line image via connected components."""
    from scipy.ndimage import label
    labeled, n_features = label(line_img)
    segments = []
    for i in range(1, n_features + 1):
        ys, xs = np.where(labeled == i)
        if len(ys) == 0:
            continue
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        length = (x2 - x1) if axis == "horizontal" else (y2 - y1)
        if length >= min_length:
            segments.append((x1, y1, x2, y2))
    return segments


def _merge_close_lines(
    lines: List[Tuple[int, int, int, int]],
    max_gap: int,
    axis: str,
) -> List[Tuple[int, int, int, int]]:
    if not lines:
        return []

    if axis == "horizontal":
        lines = sorted(lines, key=lambda l: (l[1], l[0]))
    else:
        lines = sorted(lines, key=lambda l: (l[0], l[1]))

    merged = [lines[0]]
    for line in lines[1:]:
        prev = merged[-1]
        if axis == "horizontal":
            if abs(line[1] - prev[1]) <= max_gap:
                merged[-1] = (
                    min(prev[0], line[0]),
                    min(prev[1], line[1]),
                    max(prev[2], line[2]),
                    max(prev[3], line[3]),
                )
            else:
                merged.append(line)
        else:
            if abs(line[0] - prev[0]) <= max_gap:
                merged[-1] = (
                    min(prev[0], line[0]),
                    min(prev[1], line[1]),
                    max(prev[2], line[2]),
                    max(prev[3], line[3]),
                )
            else:
                merged.append(line)

    return merged


def _lines_to_cells(
    h_lines: List[Tuple[int, int, int, int]],
    v_lines: List[Tuple[int, int, int, int]],
    table_bbox: Tuple[int, int, int, int],
) -> List[TableCell]:
    """Create cells from the grid of horizontal and vertical separators."""
    h_positions = sorted(set(
        [(l[1] + l[3]) // 2 for l in h_lines]
    ))
    v_positions = sorted(set(
        [(l[0] + l[2]) // 2 for l in v_lines]
    ))

    tx1, ty1, tx2, ty2 = table_bbox
    if not h_positions or h_positions[0] > ty1 + 5:
        h_positions.insert(0, ty1)
    if not h_positions or h_positions[-1] < ty2 - 5:
        h_positions.append(ty2)
    if not v_positions or v_positions[0] > tx1 + 5:
        v_positions.insert(0, tx1)
    if not v_positions or v_positions[-1] < tx2 - 5:
        v_positions.append(tx2)

    cells = []
    for r in range(len(h_positions) - 1):
        for c in range(len(v_positions) - 1):
            cell = TableCell(
                row=r,
                col=c,
                bbox=(v_positions[c], h_positions[r], v_positions[c + 1], h_positions[r + 1]),
            )
            cells.append(cell)

    return cells


class TableExtractor:
    """End-to-end table detection, structure recognition, and cell extraction.

    Supports two modes:
      1. **line-based** (classical): morphological line detection + grid construction.
      2. **deep** (learned): ``TableDetectionNet`` + ``TableStructureNet``.

    Args:
        mode: ``"line"`` or ``"deep"``.
        detection_threshold: Confidence threshold for deep detection.
        device: Torch device for deep models.
    """

    def __init__(
        self,
        mode: str = "line",
        detection_threshold: float = 0.5,
        device: str = "cpu",
    ):
        self.mode = mode
        self.detection_threshold = detection_threshold
        self.device = torch.device(device)

        if mode == "deep":
            self.detection_net = TableDetectionNet().to(self.device).eval()
            self.structure_net = TableStructureNet().to(self.device).eval()
        else:
            self.detection_net = None
            self.structure_net = None

    def extract(
        self,
        image: np.ndarray,
        ocr_fn: Optional[callable] = None,
    ) -> List[Table]:
        """Extract tables from an image.

        Args:
            image: BGR or RGB image as uint8 numpy array (H, W, 3).
            ocr_fn: Optional callable ``(crop_image) -> str`` for cell OCR.

        Returns:
            List of detected ``Table`` objects with structure and optionally text.
        """
        if self.mode == "deep":
            return self._extract_deep(image, ocr_fn)
        return self._extract_line_based(image, ocr_fn)

    def _extract_line_based(
        self,
        image: np.ndarray,
        ocr_fn: Optional[callable],
    ) -> List[Table]:
        if image.ndim == 3:
            gray = np.mean(image, axis=2).astype(np.uint8)
        else:
            gray = image

        h_lines, v_lines = _line_based_detect_lines(gray)

        if len(h_lines) < 2 or len(v_lines) < 2:
            return []

        all_xs = [l[0] for l in h_lines + v_lines] + [l[2] for l in h_lines + v_lines]
        all_ys = [l[1] for l in h_lines + v_lines] + [l[3] for l in h_lines + v_lines]
        table_bbox = (min(all_xs), min(all_ys), max(all_xs), max(all_ys))

        cells = _lines_to_cells(h_lines, v_lines, table_bbox)
        if not cells:
            return []

        num_rows = max(c.row for c in cells) + 1
        num_cols = max(c.col for c in cells) + 1

        if ocr_fn is not None:
            for cell in cells:
                x1, y1, x2, y2 = cell.bbox
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
                if x2 > x1 and y2 > y1:
                    crop = image[y1:y2, x1:x2]
                    cell.text = ocr_fn(crop)

        table = Table(
            bbox=table_bbox,
            num_rows=num_rows,
            num_cols=num_cols,
            cells=cells,
            confidence=1.0,
        )
        return [table]

    @torch.no_grad()
    def _extract_deep(
        self,
        image: np.ndarray,
        ocr_fn: Optional[callable],
    ) -> List[Table]:
        h, w = image.shape[:2]
        img_t = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        img_t = img_t.to(self.device)

        mask = self.detection_net(img_t)
        mask_prob = torch.sigmoid(mask).squeeze(0).squeeze(0).cpu().numpy()
        table_mask = (mask_prob > self.detection_threshold).astype(np.uint8)

        from scipy.ndimage import label as ndlabel
        labeled, n_tables = ndlabel(table_mask)

        tables = []
        for t_id in range(1, n_tables + 1):
            ys, xs = np.where(labeled == t_id)
            if len(ys) < 100:
                continue

            tx1, ty1 = int(xs.min()), int(ys.min())
            tx2, ty2 = int(xs.max()), int(ys.max())

            table_crop = image[ty1:ty2, tx1:tx2]
            crop_t = torch.from_numpy(table_crop).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            crop_t = crop_t.to(self.device)

            row_logits, col_logits = self.structure_net(crop_t)
            row_probs = torch.sigmoid(row_logits[0]).cpu().numpy()
            col_probs = torch.sigmoid(col_logits[0]).cpu().numpy()

            row_seps = _peaks_from_probs(row_probs, ty2 - ty1, threshold=0.3)
            col_seps = _peaks_from_probs(col_probs, tx2 - tx1, threshold=0.3)

            row_seps = [ty1] + [ty1 + r for r in row_seps] + [ty2]
            col_seps = [tx1] + [tx1 + c for c in col_seps] + [tx2]

            cells = []
            for r in range(len(row_seps) - 1):
                for c in range(len(col_seps) - 1):
                    cell = TableCell(
                        row=r, col=c,
                        bbox=(col_seps[c], row_seps[r], col_seps[c + 1], row_seps[r + 1]),
                    )
                    if ocr_fn is not None:
                        cx1, cy1, cx2, cy2 = cell.bbox
                        cx1, cy1 = max(0, cx1), max(0, cy1)
                        cx2, cy2 = min(w, cx2), min(h, cy2)
                        if cx2 > cx1 and cy2 > cy1:
                            cell.text = ocr_fn(image[cy1:cy2, cx1:cx2])
                    cells.append(cell)

            table = Table(
                bbox=(tx1, ty1, tx2, ty2),
                num_rows=len(row_seps) - 1,
                num_cols=len(col_seps) - 1,
                cells=cells,
                confidence=float(mask_prob[ty1:ty2, tx1:tx2].mean()),
            )
            tables.append(table)

        return tables


def _peaks_from_probs(
    probs: np.ndarray,
    original_size: int,
    threshold: float = 0.3,
) -> List[int]:
    """Find peak positions in a 1D probability array, mapped back to original scale."""
    scale = original_size / len(probs)
    peaks = []
    for i in range(1, len(probs) - 1):
        if probs[i] > threshold and probs[i] >= probs[i - 1] and probs[i] >= probs[i + 1]:
            peaks.append(int(i * scale))
    return peaks
