"""Image loading / saving with RGB normalization and Unicode path support.

The application stores images internally as NumPy ``uint8`` arrays in RGB
channel order. OpenCV uses BGR, so this module converts at the boundary.

Reading/writing goes through ``np.fromfile`` + ``cv2.imdecode`` and
``cv2.imencode`` + ``ndarray.tofile`` instead of ``cv2.imread`` /
``cv2.imwrite``, because the latter fail on non-ASCII (e.g. Chinese) file
paths on Windows.
"""
import os

import cv2
import numpy as np
from PySide6.QtGui import QImage


def load_image(path: str) -> np.ndarray:
    """Load an image file as an RGB ``uint8`` array of shape ``(h, w, 3)``.

    Raises ``ValueError`` if the file cannot be read or decoded.
    """
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        raise ValueError(f"Cannot open image: {path}")
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Cannot decode image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_image_alpha(path: str) -> np.ndarray:
    """Load an image as an RGBA (or RGB) ``uint8`` array.

    Preserves the alpha channel when present (e.g. transparent PNG logos).
    """
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        raise ValueError(f"Cannot open image: {path}")
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot decode image: {path}")
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    raise ValueError(f"Unexpected image shape: {img.shape}")


def save_image(path: str, image: np.ndarray) -> str:
    """Save an RGB/RGBA/grayscale image; return the actual path written.

    RGBA images saved to formats without alpha (JPEG/BMP) are flattened onto
    white. A missing extension defaults to PNG.
    """
    ext = os.path.splitext(path)[1].lower()
    if not ext:
        ext = ".png"
        path = path + ".png"

    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 3:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        if ext in (".jpg", ".jpeg", ".bmp"):
            rgb = image[..., :3].astype(np.float32)
            alpha = image[..., 3:4].astype(np.float32) / 255.0
            flat = np.clip(rgb * alpha + 255.0 * (1.0 - alpha), 0, 255).astype(np.uint8)
            bgr = cv2.cvtColor(flat, cv2.COLOR_RGB2BGR)
        else:
            bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    else:
        raise ValueError(f"Unsupported image shape for saving: {image.shape}")

    ok, buf = cv2.imencode(ext, bgr)
    if not ok:
        raise OSError(f"Cannot encode image for: {path}")
    with open(path, "wb") as f:
        f.write(buf.tobytes())
    return path


def save_image_pdf(path: str, image: np.ndarray) -> None:
    """Save an image as a single-page PDF using Pillow (RGBA flattened to white)."""
    from PIL import Image as PILImage

    if image.ndim == 2:
        mode, img = "L", image
    elif image.ndim == 3 and image.shape[2] == 3:
        mode, img = "RGB", image
    elif image.ndim == 3 and image.shape[2] == 4:
        rgb = image[..., :3].astype(np.float32)
        alpha = image[..., 3:4].astype(np.float32) / 255.0
        img = np.clip(rgb * alpha + 255.0 * (1.0 - alpha), 0, 255).astype(np.uint8)
        mode = "RGB"
    else:
        raise ValueError(f"Unsupported image shape for PDF: {image.shape}")
    PILImage.fromarray(img, mode=mode).save(path, "PDF", resolution=150.0)


def array_to_qimage(image: np.ndarray) -> QImage:
    """Convert an RGB (or grayscale) ``uint8`` array to a QImage.

    The returned QImage owns its data (safe to keep after the array goes away).
    """
    if image.ndim == 2:
        height, width = image.shape
        return QImage(
            image.tobytes(), width, height, width, QImage.Format.Format_Grayscale8
        ).copy()
    if image.ndim == 3 and image.shape[2] == 3:
        height, width = image.shape[:2]
        return QImage(
            image.tobytes(), width, height, width * 3, QImage.Format.Format_RGB888
        ).copy()
    if image.ndim == 3 and image.shape[2] == 4:
        height, width = image.shape[:2]
        return QImage(
            image.tobytes(), width, height, width * 4, QImage.Format.Format_RGBA8888
        ).copy()
    raise ValueError(f"Unsupported image shape for display: {image.shape}")
