"""Tests for background-removal core algorithms."""
import numpy as np

from core.background import (
    build_mask,
    composite,
    connected_regions,
    grabcut_segment,
    threshold_segment,
)


def test_grabcut_segment_runs():
    image = np.full((200, 200, 3), 255, dtype=np.uint8)
    image[60:140, 60:140] = (200, 50, 50)  # red square
    mask = grabcut_segment(image, (40, 40, 120, 120))
    assert mask.shape == (200, 200)
    assert mask.dtype == np.uint8
    assert mask[100, 100] == 255  # center of subject is foreground


def test_threshold_segment():
    image = np.full((200, 200, 3), (0, 200, 0), dtype=np.uint8)  # green bg
    image[60:140, 60:140] = (200, 0, 0)  # red square
    mask = threshold_segment(image, (10, 10))
    assert mask[100, 100] == 255  # subject kept
    assert mask[10, 10] == 0      # background removed


def test_connected_regions():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:30, 10:30] = 255
    mask[60:80, 60:80] = 255
    num, labels = connected_regions(mask)
    assert num == 3  # background + 2 blobs


def test_build_mask():
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[0:5, 0:5] = 1
    labels[5:10, 5:10] = 2
    mask = build_mask(labels, {1: True, 2: False})
    assert np.all(mask[0:5, 0:5] == 255)
    assert np.all(mask[5:10, 5:10] == 0)


def test_composite_white():
    image = np.full((10, 10, 3), (100, 150, 200), dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:4, 2:4] = 255
    result = composite(image, mask, "white")
    assert result.shape == (10, 10, 3)
    assert np.all(result[2:4, 2:4] == (100, 150, 200))
    assert np.all(result[0, 0] == 255)


def test_composite_transparent():
    image = np.full((10, 10, 3), (100, 150, 200), dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:4, 2:4] = 255
    result = composite(image, mask, "transparent")
    assert result.shape == (10, 10, 4)
    assert result[0, 0, 3] == 0     # background transparent
    assert result[3, 3, 3] == 255   # foreground opaque
