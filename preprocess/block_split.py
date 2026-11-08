import numpy as np
from typing import Tuple, List, Dict, Any

def split_into_blocks(image: np.ndarray, block_size: int) -> Tuple[np.ndarray, List[Tuple[int, int]], Dict[str, Any]]:
    """
    Split image into non-overlapping blocks.
    
    Args:
        image: Input image (H, W, C) or (H, W). Must be padded.
        block_size: Size of blocks.
        
    Returns:
        blocks: Numpy array of shape (N, block_size, block_size, C) or (N, block_size, block_size).
        positions: List of (row_idx, col_idx) tuples indicating block location.
        meta: Metadata dict with grid information.
    """
    h, w = image.shape[:2]
    
    if h % block_size != 0 or w % block_size != 0:
        raise ValueError(f"Image dimensions ({h}x{w}) not divisible by block_size {block_size}. Please pad first.")
        
    n_rows = h // block_size
    n_cols = w // block_size
    
    # Efficient splitting using reshaping and transposition
    # 1. Reshape to (rows, block_size, cols, block_size, channels)
    if image.ndim == 3:
        c = image.shape[2]
        reshaped = image.reshape(n_rows, block_size, n_cols, block_size, c)
        # 2. Transpose to (rows, cols, block_size, block_size, channels)
        transposed = reshaped.transpose(0, 2, 1, 3, 4)
        # 3. Flatten to (N, block_size, block_size, channels)
        blocks = transposed.reshape(-1, block_size, block_size, c)
    else:
        reshaped = image.reshape(n_rows, block_size, n_cols, block_size)
        transposed = reshaped.transpose(0, 2, 1, 3)
        blocks = transposed.reshape(-1, block_size, block_size)
        
    # Generate positions
    positions = []
    for r in range(n_rows):
        for c in range(n_cols):
            positions.append((r, c))
            
    meta = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "block_size": block_size,
        "channels": image.shape[2] if image.ndim == 3 else 1,
        "dtype": str(image.dtype)
    }
    
    return blocks, positions, meta


def split_image_to_blocks(image: np.ndarray, block_size: int) -> np.ndarray:
    """Backward-compatible alias returning only the block tensor."""
    blocks, _, _ = split_into_blocks(image, block_size)
    return blocks
