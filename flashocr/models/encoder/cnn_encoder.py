"""
CNN Encoder for FlashOCR.

Takes backbone feature maps and produces a sequence of feature vectors
suitable for the sequence decoder (BiLSTM/Attention).
"""

import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """CNN encoder that converts spatial feature maps to a feature sequence.

    Takes the last stage output from the backbone, applies a 1x1 convolution
    to project channels, pools height to 1, and outputs a (batch, width, channels)
    sequence for the decoder.

    Args:
        in_channels: Number of input channels (from backbone stage 4).
        out_channels: Number of output channels for the sequence features.
        use_bn: Whether to use batch normalization after projection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        use_bn: bool = True,
    ):
        super().__init__()
        self.out_channels = out_channels

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=not use_bn),
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        # Additional 3x1 conv to capture vertical context before collapsing height
        layers.append(
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0), bias=False)
        )
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        self.projection = nn.Sequential(*layers)

        # Collapse the height dimension to 1
        self.pool = nn.AdaptiveAvgPool2d((1, None))

    def forward(self, features: list) -> torch.Tensor:
        """Forward pass.

        Args:
            features: List of backbone feature maps. Uses the last one (stage 4).

        Returns:
            Tensor of shape (batch, seq_len, out_channels) where seq_len
            corresponds to the width dimension of the pooled feature map.
        """
        x = features[-1]  # Use last stage features

        x = self.projection(x)  # (B, out_channels, H, W)
        x = self.pool(x)  # (B, out_channels, 1, W)
        x = x.squeeze(2)  # (B, out_channels, W)
        x = x.permute(0, 2, 1)  # (B, W, out_channels) — sequence format

        return x
