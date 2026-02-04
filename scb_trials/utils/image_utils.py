"""
Image Processing Utilities
Helper functions for working with images.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np


def load_image(path: Union[str, Path]) -> np.ndarray:
    """
    Load an image from file.
    
    Args:
        path: Path to image file
        
    Returns:
        Image as numpy array (BGR format)
    """
    import cv2
    
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not load image: {path}")
    return image


def save_image(
    image: np.ndarray,
    path: Union[str, Path],
    quality: int = 95
) -> str:
    """
    Save an image to file.
    
    Args:
        image: Image as numpy array
        path: Output path
        quality: JPEG quality (1-100)
        
    Returns:
        Saved file path
    """
    import cv2
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    ext = path.suffix.lower()
    if ext in ('.jpg', '.jpeg'):
        cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    else:
        cv2.imwrite(str(path), image)
    
    return str(path)


def crop_region(
    image: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    padding: int = 0
) -> Optional[np.ndarray]:
    """
    Crop a region from an image.
    
    Args:
        image: Source image
        x: Top-left x coordinate
        y: Top-left y coordinate
        width: Region width
        height: Region height
        padding: Pixels to add around the region
        
    Returns:
        Cropped image or None if invalid
    """
    img_height, img_width = image.shape[:2]
    
    # Apply padding
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img_width, x + width + padding)
    y2 = min(img_height, y + height + padding)
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    return image[y1:y2, x1:x2].copy()


def resize_image(
    image: np.ndarray,
    width: Optional[int] = None,
    height: Optional[int] = None,
    max_dim: Optional[int] = None
) -> np.ndarray:
    """
    Resize an image.
    
    Args:
        image: Source image
        width: Target width (maintains aspect if height not specified)
        height: Target height (maintains aspect if width not specified)
        max_dim: Maximum dimension (resizes proportionally)
        
    Returns:
        Resized image
    """
    import cv2
    
    img_height, img_width = image.shape[:2]
    
    if max_dim is not None:
        # Resize to fit within max_dim while maintaining aspect ratio
        scale = max_dim / max(img_width, img_height)
        if scale < 1:
            width = int(img_width * scale)
            height = int(img_height * scale)
        else:
            return image
    elif width is not None and height is None:
        # Maintain aspect ratio based on width
        scale = width / img_width
        height = int(img_height * scale)
    elif height is not None and width is None:
        # Maintain aspect ratio based on height
        scale = height / img_height
        width = int(img_width * scale)
    elif width is None and height is None:
        return image
    
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert image to grayscale.
    
    Args:
        image: Source image (BGR or grayscale)
        
    Returns:
        Grayscale image
    """
    import cv2
    
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def image_to_bytes(image: np.ndarray, format: str = "png") -> bytes:
    """
    Convert image to bytes.
    
    Args:
        image: Image as numpy array
        format: Output format (png or jpg)
        
    Returns:
        Image bytes
    """
    import cv2
    
    if format.lower() == "png":
        _, buffer = cv2.imencode('.png', image)
    else:
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    return buffer.tobytes()


def bytes_to_image(data: bytes) -> np.ndarray:
    """
    Convert bytes to image.
    
    Args:
        data: Image bytes
        
    Returns:
        Image as numpy array
    """
    import cv2
    
    nparr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def compute_image_hash(image: np.ndarray, hash_size: int = 8) -> str:
    """
    Compute perceptual hash of an image.
    
    Args:
        image: Source image
        hash_size: Size of the hash (8x8 = 64 bits)
        
    Returns:
        Hexadecimal hash string
    """
    import cv2
    
    # Convert to grayscale
    gray = convert_to_grayscale(image)
    
    # Resize to hash_size + 1 for gradient computation
    resized = cv2.resize(gray, (hash_size + 1, hash_size))
    
    # Compute horizontal gradient
    diff = resized[:, 1:] > resized[:, :-1]
    
    # Convert to hex string
    hash_value = 0
    for row in diff:
        for val in row:
            hash_value = (hash_value << 1) | int(val)
    
    return format(hash_value, f'0{hash_size * hash_size // 4}x')
