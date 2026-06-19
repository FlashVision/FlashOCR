"""
CTC Decoder for FlashOCR.

BiLSTM + linear projection decoder for CTC-based text recognition.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class CTCDecoder(nn.Module):
    """BiLSTM decoder with CTC output for text recognition.

    Args:
        in_channels: Input feature dimension from the encoder.
        hidden_size: LSTM hidden state dimension.
        num_classes: Number of output classes (len(charset) + 1 for CTC blank).
        num_layers: Number of LSTM layers.
        dropout: Dropout between LSTM layers.
        bidirectional: Whether to use bidirectional LSTM.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_size: int = 256,
        num_classes: int = 37,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_out_size = hidden_size * 2 if bidirectional else hidden_size
        self.projection = nn.Linear(lstm_out_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, in_channels).

        Returns:
            Log-softmax output of shape (seq_len, batch, num_classes) for CTC loss.
        """
        output, _ = self.lstm(x)  # (B, T, hidden*2)
        logits = self.projection(output)  # (B, T, num_classes)
        log_probs = F.log_softmax(logits, dim=2)  # (B, T, num_classes)
        # CTC loss expects (T, B, C)
        log_probs = log_probs.permute(1, 0, 2)
        return log_probs

    def decode(self, log_probs: torch.Tensor, charset: str) -> List[str]:
        """Greedy CTC decoding: collapse repeats and remove blanks.

        Args:
            log_probs: Tensor of shape (T, B, C) — output of forward().
            charset: Character set string (blank is index 0).

        Returns:
            List of decoded strings, one per batch element.
        """
        # log_probs: (T, B, C)
        predictions = log_probs.argmax(dim=2)  # (T, B)
        predictions = predictions.permute(1, 0)  # (B, T)

        results = []
        blank_idx = 0

        for pred in predictions:
            chars = []
            prev = -1
            for idx in pred.tolist():
                if idx != prev:
                    if idx != blank_idx:
                        chars.append(charset[idx - 1])
                prev = idx
            results.append("".join(chars))

        return results

    def beam_search_decode(
        self,
        log_probs: torch.Tensor,
        charset: str,
        beam_width: int = 10,
    ) -> List[str]:
        """Beam search CTC decoding.

        Args:
            log_probs: Tensor of shape (T, B, C).
            charset: Character set string.
            beam_width: Number of beams to keep.

        Returns:
            List of decoded strings.
        """
        # log_probs: (T, B, C)
        T, B, C = log_probs.shape
        results = []
        blank_idx = 0

        for b in range(B):
            # beams: list of (prefix, last_char, log_prob)
            beams = [("", blank_idx, 0.0)]

            for t in range(T):
                new_beams = {}
                probs_t = log_probs[t, b].tolist()

                for prefix, last_char, score in beams:
                    for c in range(C):
                        new_score = score + probs_t[c]

                        if c == blank_idx:
                            key = (prefix, blank_idx)
                        elif c == last_char:
                            key = (prefix, c)
                        else:
                            char = charset[c - 1] if c > 0 else ""
                            key = (prefix + char, c)

                        if key not in new_beams or new_beams[key] < new_score:
                            new_beams[key] = new_score

                # Keep top-k beams
                sorted_beams = sorted(new_beams.items(), key=lambda x: x[1], reverse=True)
                beams = [(k[0], k[1], v) for (k, v) in sorted_beams[:beam_width]]

            results.append(beams[0][0] if beams else "")

        return results
