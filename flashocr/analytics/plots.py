"""Plotting utilities — training curves, CER/WER metrics, and character-level analysis."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Union

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def _get_plt():
    """Lazy-import matplotlib to avoid hard dependency at module level."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ------------------------------------------------------------------
# Training curves
# ------------------------------------------------------------------

def plot_training_curves(
    log: Dict[str, List[float]],
    keys: Optional[Sequence[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Training Curves",
) -> "Figure":
    """Plot one or more scalar metrics from a training log dict.

    Parameters
    ----------
    log : dict[str, list[float]]
        ``{"loss": [...], "cer": [...], "wer": [...], "lr": [...], ...}``
    keys : sequence of str | None
        Which keys to plot.  *None* plots everything.
    save_path : str | Path | None
        If given, save figure to this path.
    title : str
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _get_plt()
    keys = keys or list(log.keys())
    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    axes = axes.flatten()

    for ax, key in zip(axes, keys):
        values = log[key]
        ax.plot(values, linewidth=1.5)
        ax.set_title(key)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

    return fig


# ------------------------------------------------------------------
# CER / WER over epochs
# ------------------------------------------------------------------

def plot_error_rates(
    epochs: Sequence[int],
    cer_values: Sequence[float],
    wer_values: Optional[Sequence[float]] = None,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Error Rates over Training",
) -> "Figure":
    """Plot CER (and optionally WER) over training epochs.

    Parameters
    ----------
    epochs : sequence of int
        Epoch indices.
    cer_values : sequence of float
        Character Error Rate per epoch.
    wer_values : sequence of float | None
        Word Error Rate per epoch (plotted alongside CER when provided).
    save_path : str | Path | None
        Optional file path.
    title : str
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(epochs, cer_values, linewidth=1.5, marker="o", markersize=3, label="CER")
    if wer_values is not None:
        ax.plot(epochs, wer_values, linewidth=1.5, marker="s", markersize=3, label="WER")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Error Rate")
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

    return fig


# ------------------------------------------------------------------
# Loss curves (train + val)
# ------------------------------------------------------------------

def plot_loss_curves(
    train_losses: Sequence[float],
    val_losses: Optional[Sequence[float]] = None,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Loss Curves",
) -> "Figure":
    """Plot training and validation loss curves.

    Parameters
    ----------
    train_losses : sequence of float
        Training loss per epoch.
    val_losses : sequence of float | None
        Validation loss per epoch.
    save_path : str | Path | None
        Optional file path.
    title : str
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(8, 5))

    epochs = list(range(1, len(train_losses) + 1))
    ax.plot(epochs, train_losses, linewidth=1.5, label="Train Loss")
    if val_losses is not None:
        ax.plot(epochs, val_losses, linewidth=1.5, label="Val Loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

    return fig


# ------------------------------------------------------------------
# Character-level accuracy heatmap
# ------------------------------------------------------------------

def plot_character_accuracy(
    char_accuracies: Dict[str, float],
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Per-Character Accuracy",
) -> "Figure":
    """Plot a bar chart of per-character recognition accuracy.

    Parameters
    ----------
    char_accuracies : dict[str, float]
        Mapping from character to accuracy (0-1).
    save_path : str | Path | None
        Optional save path.
    title : str
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _get_plt()

    chars = sorted(char_accuracies.keys())
    accs = [char_accuracies[c] for c in chars]

    fig, ax = plt.subplots(figsize=(max(8, len(chars) * 0.4), 5))
    colors = ["#2ecc71" if a >= 0.9 else "#e67e22" if a >= 0.7 else "#e74c3c" for a in accs]
    ax.bar(range(len(chars)), accs, color=colors, edgecolor="white", width=0.8)

    ax.set_xticks(range(len(chars)))
    ax.set_xticklabels(chars, fontsize=8, rotation=0)
    ax.set_xlabel("Character")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

    return fig


# ------------------------------------------------------------------
# Confusion matrix
# ------------------------------------------------------------------

def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Character Confusion Matrix",
) -> "Figure":
    """Plot a confusion matrix as a heatmap.

    Parameters
    ----------
    matrix : np.ndarray
        Square confusion matrix of shape ``(n_chars, n_chars)``.
    class_names : list[str] | None
        Character labels.  Auto-generated indices when *None*.
    normalize : bool
        Row-normalise the matrix before plotting.
    save_path : str | Path | None
        Optional save path.
    title : str
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _get_plt()
    n = matrix.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n)]

    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        matrix = matrix.astype(np.float64) / row_sums

    fig, ax = plt.subplots(figsize=(max(6, n * 0.6), max(5, n * 0.5)))
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    thresh = matrix.max() / 2
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            text = f"{val:.2f}" if normalize else f"{int(val)}"
            ax.text(
                j, i, text, ha="center", va="center",
                color="white" if val > thresh else "black", fontsize=7,
            )

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

    return fig
