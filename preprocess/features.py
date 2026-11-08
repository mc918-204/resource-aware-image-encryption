"""Feature extraction utilities without fragile skimage dependencies."""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def _to_gray(block: np.ndarray) -> np.ndarray:
    """Convert a block to grayscale float32 in [0, 1]."""
    block_float = block.astype(np.float32)
    if block.dtype == np.uint8:
        block_float = block_float / 255.0

    if block.ndim == 3 and block.shape[2] >= 3:
        r = block_float[:, :, 0]
        g = block_float[:, :, 1]
        b = block_float[:, :, 2]
        return 0.299 * r + 0.587 * g + 0.114 * b
    if block.ndim == 3 and block.shape[2] == 1:
        return block_float[:, :, 0]
    return block_float


def _shannon_entropy(gray_block: np.ndarray) -> float:
    """Compute Shannon entropy on 8-bit quantized grayscale values."""
    quantized = np.clip(gray_block * 255.0, 0, 255).astype(np.uint8)
    histogram = np.bincount(quantized.ravel(), minlength=256).astype(np.float64)
    total = histogram.sum()
    if total <= 0:
        return 0.0
    probabilities = histogram / total
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log2(probabilities)).sum())


def _edge_map(gray_block: np.ndarray) -> np.ndarray:
    """Compute a Sobel-like edge magnitude map using numpy only."""
    padded = np.pad(gray_block, 1, mode="edge")
    gx = (
        padded[:-2, 2:]
        + 2.0 * padded[1:-1, 2:]
        + padded[2:, 2:]
        - padded[:-2, :-2]
        - 2.0 * padded[1:-1, :-2]
        - padded[2:, :-2]
    )
    gy = (
        padded[2:, :-2]
        + 2.0 * padded[2:, 1:-1]
        + padded[2:, 2:]
        - padded[:-2, :-2]
        - 2.0 * padded[:-2, 1:-1]
        - padded[:-2, 2:]
    )
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


def extract_block_features(block: np.ndarray) -> Dict[str, float]:
    """
    Extract deterministic handcrafted block features.

    Features:
    1. mean: Average intensity.
    2. variance: Global variance of pixel values.
    3. entropy: Shannon entropy over 8-bit quantized grayscale values.
    4. edge_strength: Mean edge magnitude.
    5. texture_score: Variance of the edge magnitude map.
    """
    intensity_block = _to_gray(block)
    mean_val = float(np.mean(intensity_block))
    var_val = float(np.var(intensity_block))
    entropy_val = _shannon_entropy(intensity_block)
    edge_map = _edge_map(intensity_block)
    edge_strength = float(np.mean(edge_map))
    texture_score = float(np.var(edge_map))
    return {
        "mean": mean_val,
        "variance": var_val,
        "entropy": entropy_val,
        "edge_strength": edge_strength,
        "texture_score": texture_score,
    }


def extract_features_batch(blocks: np.ndarray) -> List[Dict[str, float]]:
    """Extract features for a batch of blocks."""
    return [extract_block_features(block) for block in blocks]
