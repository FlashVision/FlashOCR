"""
Configuration for FlashOCR Model.

CRNN-based text recognition: ShuffleNetV2 backbone + CNN encoder + BiLSTM/CTC decoder.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class DataConfig:
    """Dataset paths for OCR training.

    Expects a TSV label file where each line is: image_filename\\ttext_label
    """
    train_images: str = "data/ocr/train"
    train_labels: str = "data/ocr/train/labels.tsv"
    val_images: str = "data/ocr/val"
    val_labels: str = "data/ocr/val/labels.tsv"
    test_images: str = "data/ocr/test"
    test_labels: str = "data/ocr/test/labels.tsv"
    num_workers: int = 4


@dataclass
class ModelConfig:
    """Model architecture configuration.

    FlashOCR model specifications:
    - FlashOCR-m:      backbone=1.0x, hidden=256, ~1.5M params
    - FlashOCR-m-1.5x: backbone=1.5x, hidden=384, ~2.8M params
    - FlashOCR-m-0.5x: backbone=0.5x, hidden=128, ~0.6M params
    """
    name: str = "FlashOCR"
    charset: str = "0123456789abcdefghijklmnopqrstuvwxyz"
    input_size: Tuple[int, int] = (32, 128)  # (height, width)

    backbone: str = "ShuffleNetV2"
    backbone_size: str = "1.0x"
    backbone_pretrained: bool = True

    encoder_out_channels: int = 256

    decoder_type: str = "ctc"  # "ctc" or "attention"
    hidden_size: int = 256
    num_layers: int = 2
    bidirectional: bool = True
    dropout: float = 0.1


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    grad_clip: float = 5.0
    val_interval: int = 1
    save_dir: str = "workspace/ocr_experiment"
    resume: Optional[str] = None

    enable_activation_checkpointing: bool = False
    enable_activation_offloading: bool = False
    optimizer_in_bwd: bool = False
    use_8bit_optimizer: bool = False
    compile_model: bool = False

    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["backbone", "encoder"])

    use_qlora: bool = False
    qlora_quant_dtype: str = "int8"


@dataclass
class AugmentConfig:
    """Data augmentation configuration for OCR."""
    rotation: float = 5.0  # max rotation degrees
    gaussian_noise_std: float = 0.01
    blur_kernel: int = 3
    blur_prob: float = 0.3
    brightness: float = 0.2
    contrast: Tuple[float, float] = (0.8, 1.2)
    scale: Tuple[float, float] = (0.8, 1.2)
    perspective_distortion: float = 0.05
    normalize_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class Config:
    """Top-level configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)


MODEL_SIZE_MAP = {
    "m-0.5x": ("0.5x", 128, 128),   # backbone_size, encoder_out_channels, hidden_size
    "m": ("1.0x", 256, 256),
    "m-1.5x": ("1.5x", 384, 384),
}


def get_config(
    model_size: str = "m",
    charset: str = "0123456789abcdefghijklmnopqrstuvwxyz",
    input_height: int = 32,
    input_width: int = 128,
    **overrides,
) -> Config:
    """Return configuration for a given model size.

    Args:
        model_size: One of "m-0.5x", "m", "m-1.5x".
        charset: Character set for recognition.
        input_height: Input image height.
        input_width: Input image width.
        **overrides: Additional overrides applied to the Config.
    """
    cfg = Config()

    if model_size in MODEL_SIZE_MAP:
        backbone_size, encoder_out, hidden = MODEL_SIZE_MAP[model_size]
        cfg.model.backbone_size = backbone_size
        cfg.model.encoder_out_channels = encoder_out
        cfg.model.hidden_size = hidden

    cfg.model.input_size = (input_height, input_width)
    cfg.model.charset = charset

    for key, value in overrides.items():
        parts = key.split(".")
        obj = cfg
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)

    return cfg


def load_yaml_config(yaml_path: str) -> Config:
    """Load configuration from a YAML file.

    YAML structure mirrors the Config dataclass hierarchy:
        model:
          backbone_size: "1.0x"
          charset: "0123456789abcdefghijklmnopqrstuvwxyz"
          input_size: [32, 128]
        data:
          train_images: data/ocr/train
          train_labels: data/ocr/train/labels.tsv
        train:
          epochs: 100
    """
    import yaml

    with open(yaml_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg = Config()

    if "model" in raw:
        for key, value in raw["model"].items():
            if key == "input_size" and isinstance(value, list):
                value = tuple(value)
            if hasattr(cfg.model, key):
                setattr(cfg.model, key, value)

    if "data" in raw:
        for key, value in raw["data"].items():
            if hasattr(cfg.data, key):
                setattr(cfg.data, key, value)

    if "train" in raw:
        for key, value in raw["train"].items():
            if hasattr(cfg.train, key):
                setattr(cfg.train, key, value)

    if "augment" in raw:
        for key, value in raw["augment"].items():
            if key in ("contrast", "scale") and isinstance(value, list):
                value = tuple(value)
            if hasattr(cfg.augment, key):
                setattr(cfg.augment, key, value)

    # Derive encoder/hidden from backbone_size if not explicitly set
    if "model" in raw and "encoder_out_channels" not in raw["model"]:
        bs = cfg.model.backbone_size
        size_key = {
            "0.5x": "m-0.5x",
            "1.0x": "m",
            "1.5x": "m-1.5x",
        }.get(bs)
        if size_key and size_key in MODEL_SIZE_MAP:
            _, enc_out, hidden = MODEL_SIZE_MAP[size_key]
            cfg.model.encoder_out_channels = enc_out
            if "hidden_size" not in raw.get("model", {}):
                cfg.model.hidden_size = hidden

    return cfg
