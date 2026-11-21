"""Preview helpers: downsampling for fast live previews."""
import cv2
import numpy as np


def downsample_image(image: np.ndarray, max_side: int = 1280) -> np.ndarray:
    """Return the image downsampled so its longest side is <= ``max_side``."""
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image
    scale = max_side / longest
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
