"""Tests for Phase 3 operations."""
import numpy as np
from PIL import Image as PILImage

from core.operations.color import ColorOperation
from core.operations.contrast import ContrastOperation
from core.operations.denoise import DenoiseOperation
from core.operations.geometry import FlipOperation, ResizeOperation, RotateOperation
from core.operations.sharpen import SharpenOperation
from core.operations.watermark_image import ImageWatermarkOperation
from core.operations.watermark_text import TextWatermarkOperation


def _img(h=24, w=32):
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


# ---- Denoise ------------------------------------------------------------

def test_denoise_methods_run():
    image = _img()
    op = DenoiseOperation()
    for method in ("bilateral", "gaussian", "median"):
        out = op.process(image, {"method": method, "strength": 50})
        assert out.shape == image.shape
        assert out.dtype == np.uint8


# ---- Sharpen ------------------------------------------------------------

def test_sharpen_zero_is_identity():
    image = _img()
    op = SharpenOperation()
    out = op.process(image, {"amount": 0, "radius": 2, "threshold": 0})
    assert out is image


def test_sharpen_changes_image():
    image = _img()
    op = SharpenOperation()
    out = op.process(image, {"amount": 300, "radius": 2, "threshold": 0})
    assert not np.array_equal(out, image)


# ---- Contrast -----------------------------------------------------------

def test_contrast_linear_stretch():
    image = np.full((8, 8, 3), 100, dtype=np.uint8)
    op = ContrastOperation()
    out = op.process(image, {"contrast": 2.0, "gamma": 1.0, "clahe": False})
    assert np.all(out == 200)


def test_gamma_brightens_midtones():
    image = np.full((8, 8, 3), 100, dtype=np.uint8)
    op = ContrastOperation()
    out = op.process(image, {"contrast": 1.0, "gamma": 2.0, "clahe": False})
    assert np.all(out > 100)


# ---- Color --------------------------------------------------------------

def test_saturation_minus_100_grayscale():
    image = _img(16, 16)
    op = ColorOperation()
    out = op.process(image, {"auto_wb": False, "temperature": 0, "saturation": -100})
    r, g, b = out[..., 0], out[..., 1], out[..., 2]
    assert np.max(np.abs(r.astype(int) - g.astype(int))) <= 1
    assert np.max(np.abs(g.astype(int) - b.astype(int))) <= 1


def test_gray_world_reduces_cast():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[..., 0] = 200
    image[..., 1] = 100
    image[..., 2] = 50
    op = ColorOperation()
    out = op.process(image, {"auto_wb": True, "temperature": 0, "saturation": 0})
    means = out.mean(axis=(0, 1))
    assert abs(means[0] - means[1]) < 80
    assert abs(means[1] - means[2]) < 80


# ---- Watermarks ---------------------------------------------------------

def test_text_watermark_empty_is_identity():
    image = _img()
    op = TextWatermarkOperation()
    assert op.process(image, op.default_params()) is image


def test_text_watermark_draws():
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    op = TextWatermarkOperation()
    params = op.default_params()
    params.update(text="Test", size=24, position="center", color="white", opacity=100)
    out = op.process(image, params)
    assert not np.array_equal(out, image)


def test_image_watermark_no_logo_is_identity():
    image = _img()
    op = ImageWatermarkOperation()
    assert op.process(image, op.default_params()) is image


def test_image_watermark_draws(tmp_path):
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    logo = np.full((20, 20, 4), 255, dtype=np.uint8)  # opaque white RGBA
    logo_path = tmp_path / "logo.png"
    PILImage.fromarray(logo, "RGBA").save(str(logo_path))

    op = ImageWatermarkOperation()
    params = op.default_params()
    params.update(logo=str(logo_path), scale=1.0, opacity=100,
                  position="top-left", margin=0)
    out = op.process(image, params)
    assert np.all(out[:20, :20] == 255)
    assert np.all(out[20:, 20:] == 0)


# ---- Geometry -----------------------------------------------------------

def test_flip_horizontal():
    image = _img()
    op = FlipOperation()
    out = op.process(image, {"mode": "horizontal"})
    assert np.array_equal(out, image[:, ::-1, :])


def test_rotate_180():
    image = _img()
    op = RotateOperation()
    out = op.process(image, {"mode": "180", "angle": 0})
    assert np.array_equal(out, image[::-1, ::-1, :])


def test_rotate_custom_expands_canvas():
    image = _img(h=20, w=30)
    op = RotateOperation()
    out = op.process(image, {"mode": "custom", "angle": 45})
    # Rotating 45 degrees expands the canvas instead of cropping corners.
    assert out.shape[0] > 20
    assert out.shape[1] > 30


def test_rotate_90_swaps_dims():
    image = _img(h=20, w=30)
    op = RotateOperation()
    out = op.process(image, {"mode": "90", "angle": 0})
    assert out.shape == (30, 20, 3)


def test_resize_half():
    image = _img(h=40, w=60)
    op = ResizeOperation()
    out = op.process(image, {"scale": 0.5})
    assert out.shape == (20, 30, 3)
