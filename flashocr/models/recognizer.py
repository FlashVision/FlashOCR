"""
FlashOCR recognizer.
CRNN-based text recognition with ShuffleNetV2 backbone + CNN encoder + sequence decoder.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import ShuffleNetV2
from .encoder import CNNEncoder
from .decoder import CTCDecoder, AttentionDecoder

logger = logging.getLogger(__name__)

BACKBONE_CHANNELS = {
    "0.5x": [48, 96, 192],
    "1.0x": [116, 232, 464],
    "1.5x": [176, 352, 704],
    "2.0x": [244, 488, 976],
}


class FlashOCR(nn.Module):
    """FlashOCR text recognizer.

    Ultra-lightweight CRNN model for text recognition.
    Architecture: ShuffleNetV2 backbone → CNN encoder → BiLSTM/Attention decoder.

    Args:
        charset: String of characters the model can recognize.
        input_size: (height, width) of input images.
        backbone_size: ShuffleNetV2 variant ("0.5x", "1.0x", "1.5x", "2.0x").
        encoder_out_channels: CNN encoder output channel dimension.
        decoder_type: "ctc" or "attention".
        hidden_size: Decoder LSTM/GRU hidden dimension.
        num_layers: Number of recurrent layers.
        bidirectional: Whether to use bidirectional RNN.
        dropout: Dropout probability.
        pretrained: Whether to load pretrained backbone weights.
    """

    def __init__(
        self,
        charset: str = "0123456789abcdefghijklmnopqrstuvwxyz",
        input_size: Tuple[int, int] = (32, 128),
        backbone_size: str = "1.0x",
        encoder_out_channels: int = 256,
        decoder_type: str = "ctc",
        hidden_size: int = 256,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.1,
        pretrained: bool = True,
    ):
        super().__init__()

        self.charset = charset
        self.input_size = input_size
        self.decoder_type = decoder_type

        # num_classes: charset + blank (for CTC) or charset + blank/SOS/EOS (for attention)
        if decoder_type == "ctc":
            self.num_classes = len(charset) + 1  # +1 for CTC blank at index 0
        else:
            self.num_classes = len(charset) + 3  # +3 for blank, SOS, EOS

        # Backbone
        self.backbone = ShuffleNetV2(
            model_size=backbone_size,
            out_stages=(2, 3, 4),
            pretrained=pretrained,
            activation="ReLU",
        )

        # Encoder — takes last stage features
        backbone_out_ch = BACKBONE_CHANNELS[backbone_size][-1]
        self.encoder = CNNEncoder(
            in_channels=backbone_out_ch,
            out_channels=encoder_out_channels,
        )

        # Decoder
        if decoder_type == "ctc":
            self.decoder = CTCDecoder(
                in_channels=encoder_out_channels,
                hidden_size=hidden_size,
                num_classes=self.num_classes,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=bidirectional,
            )
        elif decoder_type == "attention":
            self.decoder = AttentionDecoder(
                in_channels=encoder_out_channels,
                hidden_size=hidden_size,
                num_classes=self.num_classes,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type}")

        # CTC loss (built-in for convenience)
        self.ctc_loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
    ) -> Dict:
        """Forward pass.

        Args:
            x: Input images (B, 3, H, W).
            targets: Target label indices for training.
                For CTC: (sum of target_lengths,) — concatenated targets.
                For attention: (B, max_target_len).
            target_lengths: Length of each target in the batch (CTC only).

        Returns:
            Dict with 'loss' (if targets given) and/or 'log_probs'/'logits'.
        """
        features = self.backbone(x)
        encoded = self.encoder(features)  # (B, T, C)

        if self.decoder_type == "ctc":
            log_probs = self.decoder(encoded)  # (T, B, num_classes)
            result = {"log_probs": log_probs}

            if targets is not None and target_lengths is not None:
                T = log_probs.size(0)
                B = log_probs.size(1)
                input_lengths = torch.full((B,), T, dtype=torch.long, device=x.device)
                loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
                result["loss"] = loss

            return result

        else:
            # Attention decoder
            teacher_ratio = 1.0 if self.training and targets is not None else 0.0
            logits = self.decoder(encoded, targets=targets, teacher_forcing_ratio=teacher_ratio)
            result = {"logits": logits}

            if targets is not None:
                # Cross entropy loss
                B, T, C = logits.shape
                loss = F.cross_entropy(
                    logits.reshape(B * T, C),
                    targets.reshape(B * T),
                    ignore_index=0,
                )
                result["loss"] = loss

            return result

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> List[Tuple[str, float]]:
        """Run inference and return decoded text with confidence.

        Args:
            x: Input images (B, 3, H, W).

        Returns:
            List of (text, confidence) tuples, one per batch element.
        """
        self.eval()
        features = self.backbone(x)
        encoded = self.encoder(features)

        if self.decoder_type == "ctc":
            log_probs = self.decoder(encoded)  # (T, B, C)
            texts = self.decoder.decode(log_probs, self.charset)

            # Compute confidence as mean probability of non-blank predictions
            probs = log_probs.exp()  # (T, B, C)
            max_probs, max_indices = probs.max(dim=2)  # (T, B)
            max_probs = max_probs.permute(1, 0)  # (B, T)
            max_indices = max_indices.permute(1, 0)  # (B, T)

            confidences = []
            for i in range(max_probs.size(0)):
                non_blank = max_indices[i] != 0
                if non_blank.any():
                    conf = max_probs[i][non_blank].mean().item()
                else:
                    conf = 0.0
                confidences.append(conf)

            return list(zip(texts, confidences))

        else:
            texts = self.decoder.decode(encoded, self.charset)
            # For attention decoder, use softmax probabilities
            logits = self.decoder(encoded, targets=None, teacher_forcing_ratio=0.0)
            probs = F.softmax(logits, dim=2)
            max_probs = probs.max(dim=2).values  # (B, T)

            confidences = []
            predictions = logits.argmax(dim=2)
            for i in range(max_probs.size(0)):
                non_special = predictions[i] > self.decoder.eos_idx
                if non_special.any():
                    conf = max_probs[i][non_special].mean().item()
                else:
                    conf = 0.0
                confidences.append(conf)

            return list(zip(texts, confidences))

    def get_model_info(self) -> Dict:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "name": "FlashOCR",
            "charset_size": len(self.charset),
            "num_classes": self.num_classes,
            "input_size": self.input_size,
            "decoder_type": self.decoder_type,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "params_mb": total_params * 4 / (1024 ** 2),
            "params_fp16_mb": total_params * 2 / (1024 ** 2),
        }


def build_model(config) -> FlashOCR:
    """Build FlashOCR model from config.

    Args:
        config: Configuration object with model settings.

    Returns:
        FlashOCR model instance.
    """
    return FlashOCR(
        charset=config.model.charset,
        input_size=config.model.input_size,
        backbone_size=config.model.backbone_size,
        encoder_out_channels=config.model.encoder_out_channels,
        decoder_type=config.model.decoder_type,
        hidden_size=config.model.hidden_size,
        num_layers=config.model.num_layers,
        bidirectional=config.model.bidirectional,
        dropout=config.model.dropout,
        pretrained=config.model.backbone_pretrained,
    )
