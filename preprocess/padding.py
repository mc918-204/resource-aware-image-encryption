import numpy as np
from typing import Tuple, Dict, Any

def pad_image(image: np.ndarray, block_size: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Pad image to be divisible by block_size.
    Uses 'reflect' mode to minimize boundary artifacts.
    
    Args:
        image: Input image (H, W) or (H, W, C).
        block_size: Block size (e.g. 16).
        
    Returns:
        padded_image: The padded image.
        meta: Metadata dict containing 'original_shape' and 'padding'.
    """
    h, w = image.shape[:2]
    
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    
    meta = {
        "original_shape": image.shape,
        "padding": (pad_h, pad_w),
        "block_size": block_size
    }
    
    if pad_h == 0 and pad_w == 0:
        return image.copy(), meta
        
    # Construct padding tuple
    # ((top, bottom), (left, right)) for 2D
    # ((top, bottom), (left, right), (0, 0)) for 3D
    if image.ndim == 3:
        pad_width = ((0, pad_h), (0, pad_w), (0, 0))
    else:
        pad_width = ((0, pad_h), (0, pad_w))
        
    # 'reflect' mode helps with edge features stability compared to constant 0
    padded_image = np.pad(image, pad_width, mode='reflect')
    
    return padded_image, meta

def unpad_image(padded_image: np.ndarray, meta: Dict[str, Any]) -> np.ndarray:
    """
    Restore original image from padded version using metadata.
    
    Args:
        padded_image: The padded image.
        meta: Metadata dict returned by pad_image.
        
    Returns:
        original_image: Cropped image.
    """
    original_shape = meta["original_shape"]
    h, w = original_shape[:2]
    
    return padded_image[:h, :w]
