"""TrOCR — Transformer-based Optical Character Recognition.

Wraps the HuggingFace TrOCR vision-encoder-decoder model for text recognition.
Uses a ViT image encoder and a GPT-2-style autoregressive text decoder.

References:
    Li et al., "TrOCR: Transformer-based Optical Character Recognition
    with Pre-trained Models", AAAI 2023.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashocr.registry import DECODERS

logger = logging.getLogger(__name__)

_HF_AVAILABLE = False
try:
    from transformers import (
        TrOCRProcessor,
        VisionEncoderDecoderModel,
        VisionEncoderDecoderConfig,
    )
    _HF_AVAILABLE = True
except ImportError:
    pass


class ViTEncoder(nn.Module):
    """Lightweight Vision Transformer encoder for TrOCR.

    Patch-based image tokenization with positional embeddings and
    standard transformer encoder blocks.
    """

    def __init__(
        self,
        img_size: Tuple[int, int] = (32, 128),
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        h_patches = img_size[0] // patch_size
        w_patches = img_size[1] // patch_size
        self.num_patches = h_patches * w_patches

        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )
        self.pos_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)  # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.encoder(x)
        x = self.norm(x)
        return x


class TransformerDecoder(nn.Module):
    """Autoregressive transformer decoder for text generation."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        max_seq_len: int = 64,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )
        return mask

    def forward(
        self,
        encoder_out: torch.Tensor,
        tgt_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            encoder_out: (B, S_enc, embed_dim) from the ViT encoder.
            tgt_ids: (B, T) target token indices.

        Returns:
            Logits of shape (B, T, vocab_size).
        """
        T = tgt_ids.size(1)
        tgt = self.token_embed(tgt_ids) + self.pos_embed[:, :T]
        tgt = self.pos_drop(tgt)

        causal_mask = self._causal_mask(T, tgt.device)
        out = self.decoder(tgt, encoder_out, tgt_mask=causal_mask)
        out = self.norm(out)
        logits = self.head(out)
        return logits


@DECODERS.register("TrOCR")
class TrOCR(nn.Module):
    """TrOCR text recognizer — ViT encoder + Transformer decoder.

    Supports two modes:
      1. **Standalone** (default): Uses built-in lightweight ViT encoder and
         autoregressive transformer decoder with configurable sizes.
      2. **HuggingFace**: Wraps a pretrained ``VisionEncoderDecoderModel`` from
         the HuggingFace Hub (set ``hf_model_name``).

    Args:
        charset: String of characters the model can recognise.
        input_size: (height, width) of input images.
        embed_dim: Transformer embedding dimension (standalone mode).
        num_heads: Number of attention heads (standalone mode).
        encoder_layers: Number of ViT encoder layers (standalone mode).
        decoder_layers: Number of decoder layers (standalone mode).
        max_seq_len: Maximum output sequence length.
        patch_size: ViT patch size (standalone mode).
        dropout: Dropout probability.
        hf_model_name: HuggingFace model name to load (e.g.
            ``"microsoft/trocr-base-printed"``).  When set, standalone
            architecture args are ignored.
    """

    SOS_TOKEN = 0
    EOS_TOKEN = 1
    PAD_TOKEN = 2

    def __init__(
        self,
        charset: str = "0123456789abcdefghijklmnopqrstuvwxyz",
        input_size: Tuple[int, int] = (32, 128),
        embed_dim: int = 256,
        num_heads: int = 4,
        encoder_layers: int = 4,
        decoder_layers: int = 4,
        max_seq_len: int = 64,
        patch_size: int = 4,
        dropout: float = 0.1,
        hf_model_name: Optional[str] = None,
    ):
        super().__init__()
        self.charset = charset
        self.input_size = input_size
        self.max_seq_len = max_seq_len

        # vocab: SOS, EOS, PAD, then charset characters
        self.vocab_size = len(charset) + 3
        self._char_to_idx = {c: i + 3 for i, c in enumerate(charset)}
        self._idx_to_char = {i + 3: c for i, c in enumerate(charset)}

        self.hf_mode = hf_model_name is not None and _HF_AVAILABLE

        if self.hf_mode:
            self.hf_model = VisionEncoderDecoderModel.from_pretrained(hf_model_name)
            self.hf_processor = TrOCRProcessor.from_pretrained(hf_model_name)
            self.encoder = None
            self.decoder = None
        else:
            self.hf_model = None
            self.hf_processor = None
            self.encoder = ViTEncoder(
                img_size=input_size,
                patch_size=patch_size,
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=encoder_layers,
                dropout=dropout,
            )
            self.decoder = TransformerDecoder(
                vocab_size=self.vocab_size,
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=decoder_layers,
                max_seq_len=max_seq_len,
                dropout=dropout,
            )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.hf_mode:
            return self.hf_model.encoder(pixel_values=x).last_hidden_state
        return self.encoder(x)

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
    ) -> Dict:
        """Forward pass.

        Args:
            x: Input images (B, 3, H, W).
            targets: (B, T) target token indices (including SOS/EOS).
            target_lengths: Unused, kept for API compatibility.

        Returns:
            Dict with ``'logits'`` and optionally ``'loss'``.
        """
        if self.hf_mode:
            return self._forward_hf(x, targets)

        encoder_out = self.encoder(x)

        if targets is not None:
            decoder_input = targets[:, :-1]
            decoder_target = targets[:, 1:]
            logits = self.decoder(encoder_out, decoder_input)
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                decoder_target.reshape(-1),
                ignore_index=self.PAD_TOKEN,
            )
            return {"logits": logits, "loss": loss}

        # Inference: autoregressive decoding
        logits = self._greedy_decode(encoder_out)
        return {"logits": logits}

    def _forward_hf(self, x: torch.Tensor, targets: Optional[torch.Tensor]) -> Dict:
        if targets is not None:
            outputs = self.hf_model(pixel_values=x, labels=targets)
            return {"logits": outputs.logits, "loss": outputs.loss}
        generated = self.hf_model.generate(x, max_new_tokens=self.max_seq_len)
        return {"generated_ids": generated}

    @torch.no_grad()
    def _greedy_decode(self, encoder_out: torch.Tensor) -> torch.Tensor:
        B = encoder_out.size(0)
        device = encoder_out.device

        sos = torch.full((B, 1), self.SOS_TOKEN, dtype=torch.long, device=device)
        generated = sos
        all_logits = []

        for _ in range(self.max_seq_len):
            logits = self.decoder(encoder_out, generated)
            next_logits = logits[:, -1:]
            all_logits.append(next_logits)
            next_token = next_logits.argmax(dim=-1)
            generated = torch.cat([generated, next_token], dim=1)

            if (next_token == self.EOS_TOKEN).all():
                break

        return torch.cat(all_logits, dim=1)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> List[Tuple[str, float]]:
        """Run inference and return decoded text with confidence.

        Args:
            x: Input images (B, 3, H, W).

        Returns:
            List of (text, confidence) tuples.
        """
        self.eval()

        if self.hf_mode:
            generated = self.hf_model.generate(x, max_new_tokens=self.max_seq_len)
            texts = self.hf_processor.batch_decode(generated, skip_special_tokens=True)
            return [(t, 1.0) for t in texts]

        encoder_out = self.encoder(x)
        logits = self._greedy_decode(encoder_out)  # (B, T, vocab)
        probs = F.softmax(logits, dim=-1)
        max_probs, predictions = probs.max(dim=-1)

        results = []
        for i in range(predictions.size(0)):
            chars = []
            confs = []
            for t in range(predictions.size(1)):
                idx = predictions[i, t].item()
                if idx == self.EOS_TOKEN:
                    break
                if idx == self.PAD_TOKEN or idx == self.SOS_TOKEN:
                    continue
                if idx in self._idx_to_char:
                    chars.append(self._idx_to_char[idx])
                    confs.append(max_probs[i, t].item())

            text = "".join(chars)
            confidence = sum(confs) / len(confs) if confs else 0.0
            results.append((text, confidence))

        return results

    def get_model_info(self) -> Dict:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "name": "TrOCR",
            "charset_size": len(self.charset),
            "vocab_size": self.vocab_size,
            "input_size": self.input_size,
            "max_seq_len": self.max_seq_len,
            "hf_mode": self.hf_mode,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "params_mb": total_params * 4 / (1024 ** 2),
        }
