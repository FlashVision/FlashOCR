"""
Attention Decoder for FlashOCR.

GRU-based decoder with Bahdanau attention as an alternative to CTC.
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    """Bahdanau (additive) attention mechanism."""

    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int):
        super().__init__()
        self.encoder_proj = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.decoder_proj = nn.Linear(decoder_dim, attention_dim, bias=False)
        self.energy = nn.Linear(attention_dim, 1, bias=False)

    def forward(
        self, encoder_out: torch.Tensor, decoder_hidden: torch.Tensor
    ) -> tuple:
        """Compute attention weights and context vector.

        Args:
            encoder_out: (B, T, encoder_dim)
            decoder_hidden: (B, decoder_dim)

        Returns:
            context: (B, encoder_dim)
            weights: (B, T)
        """
        enc_proj = self.encoder_proj(encoder_out)  # (B, T, attn_dim)
        dec_proj = self.decoder_proj(decoder_hidden).unsqueeze(1)  # (B, 1, attn_dim)

        energy = torch.tanh(enc_proj + dec_proj)  # (B, T, attn_dim)
        scores = self.energy(energy).squeeze(2)  # (B, T)

        weights = F.softmax(scores, dim=1)  # (B, T)
        context = torch.bmm(weights.unsqueeze(1), encoder_out).squeeze(1)  # (B, enc_dim)

        return context, weights


class AttentionDecoder(nn.Module):
    """GRU-based decoder with Bahdanau attention for text recognition.

    Args:
        in_channels: Encoder output dimension.
        hidden_size: GRU hidden state dimension.
        num_classes: Number of output classes (charset size + special tokens).
        max_length: Maximum decoding length.
        dropout: Dropout probability.
        sos_idx: Start-of-sequence token index.
        eos_idx: End-of-sequence token index.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_size: int = 256,
        num_classes: int = 39,
        max_length: int = 50,
        dropout: float = 0.1,
        sos_idx: int = 1,
        eos_idx: int = 2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.max_length = max_length
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

        self.embedding = nn.Embedding(num_classes, hidden_size)
        self.attention = BahdanauAttention(in_channels, hidden_size, hidden_size)
        self.gru = nn.GRUCell(hidden_size + in_channels, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_size, num_classes)

        self.init_hidden_proj = nn.Linear(in_channels, hidden_size)

    def _init_hidden(self, encoder_out: torch.Tensor) -> torch.Tensor:
        """Initialize decoder hidden state from encoder output."""
        mean_enc = encoder_out.mean(dim=1)  # (B, in_channels)
        return torch.tanh(self.init_hidden_proj(mean_enc))  # (B, hidden)

    def forward(
        self,
        encoder_out: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 1.0,
    ) -> torch.Tensor:
        """Forward pass with optional teacher forcing.

        Args:
            encoder_out: (B, T, in_channels) from the encoder.
            targets: (B, max_len) target token indices for teacher forcing.
            teacher_forcing_ratio: Probability of using teacher forcing.

        Returns:
            outputs: (B, max_length, num_classes) logits at each step.
        """
        batch_size = encoder_out.size(0)
        device = encoder_out.device

        hidden = self._init_hidden(encoder_out)

        if targets is not None:
            max_steps = targets.size(1)
        else:
            max_steps = self.max_length

        outputs = torch.zeros(batch_size, max_steps, self.num_classes, device=device)

        # First input is SOS token
        input_token = torch.full(
            (batch_size,), self.sos_idx, dtype=torch.long, device=device
        )

        for t in range(max_steps):
            embedded = self.embedding(input_token)  # (B, hidden)
            embedded = self.dropout(embedded)

            context, _ = self.attention(encoder_out, hidden)  # (B, in_channels)

            gru_input = torch.cat([embedded, context], dim=1)  # (B, hidden + in_ch)
            hidden = self.gru(gru_input, hidden)  # (B, hidden)

            output = self.output_proj(hidden)  # (B, num_classes)
            outputs[:, t, :] = output

            # Teacher forcing
            if targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
                input_token = targets[:, t]
            else:
                input_token = output.argmax(dim=1)

        return outputs

    def decode(self, encoder_out: torch.Tensor, charset: str) -> List[str]:
        """Greedy decoding at inference time.

        Args:
            encoder_out: (B, T, in_channels) from the encoder.
            charset: Character set (index 0=blank, 1=SOS, 2=EOS, 3+=chars).

        Returns:
            List of decoded strings.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(encoder_out, targets=None, teacher_forcing_ratio=0.0)

        predictions = logits.argmax(dim=2)  # (B, max_length)
        results = []

        for pred in predictions:
            chars = []
            for idx in pred.tolist():
                if idx == self.eos_idx:
                    break
                if idx > self.eos_idx:
                    char_idx = idx - 3  # offset for blank/SOS/EOS
                    if 0 <= char_idx < len(charset):
                        chars.append(charset[char_idx])
            results.append("".join(chars))

        return results
