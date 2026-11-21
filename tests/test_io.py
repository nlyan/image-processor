"""Tests for core.io (BGR/RGB conversion, round-trip, Unicode paths, PDF)."""
import os

import numpy as np

from core.io import load_image, save_image, save_image_pdf


def test_save_then_load_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(24, 32, 3), dtype=np.uint8)
    path = tmp_path / "roundtrip.png"

    save_image(str(path), image)
    loaded = load_image(str(path))

    assert loaded.shape == image.shape
    assert loaded.dtype == np.uint8
    # PNG is lossless, so the round-trip must be exact.
    assert np.array_equal(loaded, image)


def test_load_returns_rgb_not_bgr(tmp_path):
    # A pure-red pixel must come back as R=255, G=0, B=0 (not BGR).
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    image[0, 0] = (255, 0, 0)
    path = tmp_path / "red.png"

    save_image(str(path), image)
    loaded = load_image(str(path))

    assert loaded[0, 0, 0] == 255  # red channel
    assert loaded[0, 0, 2] == 0    # blue channel


def test_save_image_pdf(tmp_path):
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[..., 0] = 255
    path = tmp_path / "out.pdf"
    save_image_pdf(str(path), image)
    assert os.path.exists(str(path))
    assert os.path.getsize(str(path)) > 0


def test_unicode_path_roundtrip(tmp_path):
    # Non-ASCII (Chinese) paths must work on Windows.
    chinese_dir = tmp_path / "中文目录"
    chinese_dir.mkdir()
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:, :, 1] = 255  # pure green
    path = chinese_dir / "绿色图片.png"

    save_image(str(path), image)
    loaded = load_image(str(path))

    assert loaded.shape == image.shape
    assert np.array_equal(loaded, image)
