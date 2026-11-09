import numpy as np
from PIL import Image
from pathlib import Path

def load_image(image_path: str, grayscale: bool = False) -> np.ndarray:
    """
    Load an image from disk.
    
    Args:
        image_path: Path to the image file.
        grayscale: If True, convert to grayscale.
        
    Returns:
        np.ndarray: Image data as numpy array (uint8).
        
    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If the image cannot be opened.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found at: {image_path}")
        
    try:
        with Image.open(path) as img:
            if grayscale:
                img = img.convert('L')
            else:
                img = img.convert('RGB')
            return np.array(img)
    except Exception as e:
        raise IOError(f"Failed to load image: {e}")
