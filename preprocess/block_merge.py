import numpy as np
from typing import List, Tuple, Dict, Any, Optional

def merge_blocks(blocks: np.ndarray, positions: Optional[List[Tuple[int, int]]], meta: Dict[str, Any]) -> np.ndarray:
    """
    Merge blocks back into full image.
    
    Args:
        blocks: Array of blocks (N, H, W, C) or (N, H, W).
        positions: List of (row, col) positions (optional if blocks are ordered).
                   Currently assumes blocks are ordered row-major if positions is None or ignored.
                   But we will use meta for dimensions.
        meta: Metadata dict from split_into_blocks.
        
    Returns:
        reconstructed_image: The full image.
    """
    n_rows = meta["n_rows"]
    n_cols = meta["n_cols"]
    block_size = meta["block_size"]
    channels = meta["channels"]
    dtype = np.dtype(meta.get("dtype", "uint8"))
    
    full_h = n_rows * block_size
    full_w = n_cols * block_size
    
    if channels > 1:
        canvas = np.zeros((full_h, full_w, channels), dtype=dtype)
    else:
        canvas = np.zeros((full_h, full_w), dtype=dtype)
        
    # If positions matches blocks length, use them
    if positions is not None and len(positions) == len(blocks):
        for i, (r, c) in enumerate(positions):
            r_start = r * block_size
            r_end = r_start + block_size
            c_start = c * block_size
            c_end = c_start + block_size
            
            canvas[r_start:r_end, c_start:c_end] = blocks[i]
    else:
        # Fallback to simple reshape if positions missing/mismatched (assuming sorted)
        # This mirrors the logic in block_split
        if channels > 1:
             # (N, B, B, C) -> (Rows, Cols, B, B, C)
             reshaped = blocks.reshape(n_rows, n_cols, block_size, block_size, channels)
             # -> (Rows, B, Cols, B, C)
             transposed = reshaped.transpose(0, 2, 1, 3, 4)
             # -> (H, W, C)
             canvas = transposed.reshape(full_h, full_w, channels)
        else:
             reshaped = blocks.reshape(n_rows, n_cols, block_size, block_size)
             transposed = reshaped.transpose(0, 2, 1, 3)
             canvas = transposed.reshape(full_h, full_w)
             
    return canvas


def merge_blocks_to_image(blocks: np.ndarray, image_shape: Tuple[int, ...], block_size: int) -> np.ndarray:
    """Backward-compatible alias used by older smoke tests."""
    meta = {
        "n_rows": image_shape[0] // block_size,
        "n_cols": image_shape[1] // block_size,
        "block_size": block_size,
        "channels": image_shape[2] if len(image_shape) == 3 else 1,
        "dtype": str(blocks.dtype),
    }
    return merge_blocks(blocks, positions=None, meta=meta)
