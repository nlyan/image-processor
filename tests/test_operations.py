"""Tests for built-in operations."""
import numpy as np

from core.operations.brightness import BrightnessOperation


def test_brightness_adds_and_clips():
    op = BrightnessOperation()
    image = np.full((2, 2, 3), 200, dtype=np.uint8)

    bright = op.process(image, {"delta": 100})
    assert np.all(bright == 255)

    dark = op.process(image, {"delta": -100})
    assert np.all(dark == 100)


def test_brightness_zero_returns_same():
    op = BrightnessOperation()
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    out = op.process(image, {"delta": 0})
    assert out is image
