"""Document layout analysis.

Detects and classifies document regions (title, paragraph, table, figure,
list, header, footer, page number, caption) using either a deep-learning
model or rule-based heuristics.
"""

import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LayoutLabel(IntEnum):
    """Document layout region types."""

    TITLE = 0
    PARAGRAPH = 1
    TABLE = 2
    FIGURE = 3
    LIST = 4
    HEADER = 5
    FOOTER = 6
    PAGE_NUMBER = 7
    CAPTION = 8

    @classmethod
    def num_classes(cls) -> int:
        return len(cls)

    @classmethod
    def name_of(cls, idx: int) -> str:
        for member in cls:
            if member.value == idx:
                return member.name.lower()
        return "unknown"


@dataclass
class LayoutRegion:
    """A detected layout region."""

    label: int
    label_name: str
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    text: str = ""


class _DWConv(nn.Module):
    """Depth-wise separable convolution."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.dw(x)))
        x = self.act(self.bn2(self.pw(x)))
        return x


class LayoutBackbone(nn.Module):
    """Lightweight backbone for layout analysis."""

    def __init__(self, in_channels: int = 3, base_ch: int = 32):
        super().__init__()
        c = base_ch
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, 2, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(_DWConv(c, c * 2, stride=2), _DWConv(c * 2, c * 2))
        self.stage2 = nn.Sequential(_DWConv(c * 2, c * 4, stride=2), _DWConv(c * 4, c * 4))
        self.stage3 = nn.Sequential(_DWConv(c * 4, c * 8, stride=2), _DWConv(c * 8, c * 8))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        c1 = self.stage1(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        return [c1, c2, c3]


class LayoutFPN(nn.Module):
    """Simple FPN for multi-scale feature fusion."""

    def __init__(self, in_channels_list: List[int], out_channels: int = 64):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()
        for in_ch in in_channels_list:
            self.lateral_convs.append(nn.Conv2d(in_ch, out_channels, 1))
            self.output_convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]
        for i in range(len(laterals) - 2, -1, -1):
            up = F.interpolate(laterals[i + 1], size=laterals[i].shape[2:], mode="bilinear", align_corners=False)
            laterals[i] = laterals[i] + up
        outputs = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]
        return outputs


class LayoutDetectionHead(nn.Module):
    """Detection head that produces per-class heatmaps + bounding-box regression."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.heatmap = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, 1),
        )
        self.regression = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 4, 1),  # dx, dy, dw, dh
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.heatmap(x), self.regression(x)


class LayoutNet(nn.Module):
    """End-to-end layout analysis network.

    Produces per-class heatmaps and box regression for document regions.
    """

    def __init__(self, num_classes: int = 9, base_ch: int = 32, fpn_ch: int = 64):
        super().__init__()
        self.backbone = LayoutBackbone(base_ch=base_ch)
        ch_list = [base_ch * 2, base_ch * 4, base_ch * 8]
        self.fpn = LayoutFPN(ch_list, out_channels=fpn_ch)
        self.head = LayoutDetectionHead(fpn_ch, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        fpn_features = self.fpn(features)
        fused = fpn_features[0]
        for f in fpn_features[1:]:
            fused = fused + F.interpolate(f, size=fused.shape[2:], mode="bilinear", align_corners=False)
        heatmap, regression = self.head(fused)
        return heatmap, regression


def _nms_detections(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    iou_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-class non-maximum suppression."""
    if len(boxes) == 0:
        return boxes, scores, labels

    keep_boxes, keep_scores, keep_labels = [], [], []

    for cls in np.unique(labels):
        cls_mask = labels == cls
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]

        order = cls_scores.argsort()[::-1]
        cls_boxes = cls_boxes[order]
        cls_scores = cls_scores[order]

        selected = []
        while len(cls_boxes) > 0:
            selected.append(0)
            if len(cls_boxes) == 1:
                break

            ious = _compute_iou_vec(cls_boxes[0], cls_boxes[1:])
            keep = ious < iou_threshold
            cls_boxes = np.concatenate([[cls_boxes[0]], cls_boxes[1:][keep]])
            cls_scores = np.concatenate([[cls_scores[0]], cls_scores[1:][keep]])
            cls_boxes = cls_boxes[1:]
            cls_scores = cls_scores[1:]
            if len(selected) > 0:
                keep_boxes.append(cls_boxes[:0].reshape(0, 4) if len(cls_boxes) == 0 else np.array([]))
                break

        if selected:
            keep_boxes.extend([boxes[cls_mask][order[i]] for i in selected])
            keep_scores.extend([scores[cls_mask][order[i]] for i in selected])
            keep_labels.extend([cls] * len(selected))

    if not keep_boxes:
        return np.empty((0, 4)), np.array([]), np.array([], dtype=int)

    return np.array(keep_boxes), np.array(keep_scores), np.array(keep_labels, dtype=int)


def _compute_iou_vec(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area1 + area2 - inter, 1e-6)


def _heatmap_to_detections(
    heatmap: np.ndarray,
    regression: np.ndarray,
    score_threshold: float,
    image_h: int,
    image_w: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert heatmap + regression outputs to bounding box detections."""
    num_classes, fh, fw = heatmap.shape
    scale_y = image_h / fh
    scale_x = image_w / fw

    boxes, scores, labels = [], [], []

    for cls in range(num_classes):
        cls_map = heatmap[cls]
        for y in range(1, fh - 1):
            for x in range(1, fw - 1):
                score = cls_map[y, x]
                if score < score_threshold:
                    continue
                if score < cls_map[y - 1, x] or score < cls_map[y + 1, x]:
                    continue
                if score < cls_map[y, x - 1] or score < cls_map[y, x + 1]:
                    continue

                dx, dy, dw, dh = regression[:, y, x]
                cx = (x + dx) * scale_x
                cy = (y + dy) * scale_y
                bw = max(np.exp(dw) * scale_x * 4, 10)
                bh = max(np.exp(dh) * scale_y * 4, 10)

                x1 = max(0, cx - bw / 2)
                y1 = max(0, cy - bh / 2)
                x2 = min(image_w, cx + bw / 2)
                y2 = min(image_h, cy + bh / 2)

                boxes.append([x1, y1, x2, y2])
                scores.append(float(score))
                labels.append(cls)

    if not boxes:
        return np.empty((0, 4)), np.array([]), np.array([], dtype=int)

    return np.array(boxes), np.array(scores), np.array(labels, dtype=int)


class LayoutAnalyzer:
    """Document layout analyzer.

    Detects and classifies document regions into titles, paragraphs,
    tables, figures, lists, headers, footers, page numbers, and captions.

    Args:
        model: Pre-trained ``LayoutNet`` or ``None`` to load defaults.
        score_threshold: Minimum detection confidence.
        nms_threshold: IoU threshold for NMS.
        device: Torch device.
    """

    LABELS = [m.name.lower() for m in LayoutLabel]

    def __init__(
        self,
        model: Optional[LayoutNet] = None,
        score_threshold: float = 0.3,
        nms_threshold: float = 0.5,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold

        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = LayoutNet(num_classes=LayoutLabel.num_classes()).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def analyze(self, image: np.ndarray) -> List[LayoutRegion]:
        """Analyse a document image and return detected layout regions.

        Args:
            image: RGB/BGR uint8 numpy array (H, W, 3).

        Returns:
            List of ``LayoutRegion`` detections.
        """
        h, w = image.shape[:2]
        img_t = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        img_t = img_t.to(self.device)

        heatmap, regression = self.model(img_t)
        heatmap = torch.sigmoid(heatmap[0]).cpu().numpy()
        regression = regression[0].cpu().numpy()

        boxes, scores, labels = _heatmap_to_detections(
            heatmap, regression, self.score_threshold, h, w,
        )
        boxes, scores, labels = _nms_detections(boxes, scores, labels, self.nms_threshold)

        regions = []
        for box, score, label in zip(boxes, scores, labels):
            region = LayoutRegion(
                label=int(label),
                label_name=LayoutLabel.name_of(int(label)),
                bbox=tuple(int(v) for v in box),
                confidence=float(score),
            )
            regions.append(region)

        return regions

    def analyze_and_sort(self, image: np.ndarray) -> List[LayoutRegion]:
        """Analyse and sort regions in reading order (top-to-bottom, left-to-right)."""
        regions = self.analyze(image)
        regions.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
        return regions

    def get_regions_by_type(
        self,
        image: np.ndarray,
        region_type: str,
    ) -> List[LayoutRegion]:
        """Get only regions of a specific type."""
        regions = self.analyze(image)
        return [r for r in regions if r.label_name == region_type.lower()]
