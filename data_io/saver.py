import numpy as np
from PIL import Image
from pathlib import Path

def save_image(image_data: np.ndarray, save_path: str) -> None:
    """
    Save numpy array as an image file.
    
    Args:
        image_data: Image data (H, W, C) or (H, W).
        save_path: Destination path.
    
    Raises:
        IOError: If saving fails.
    """
    try:
        # Handle grayscale (H, W) -> (H, W) or (H, W, 1) -> (H, W)
        if image_data.ndim == 3 and image_data.shape[2] == 1:
            image_data = image_data.squeeze(axis=2)
            
        img = Image.fromarray(image_data.astype(np.uint8))
        img.save(save_path)
    except Exception as e:
        raise IOError(f"Failed to save image to {save_path}: {e}")
