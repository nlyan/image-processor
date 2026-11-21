"""Tests for the document-scanning core algorithms."""
import cv2
import numpy as np

from core.scan import (
    detect_document_corners,
    four_point_transform,
    order_points,
    remove_illumination,
    scan_document,
    warp_polygon_to_rectangle,
)


def test_order_points():
    pts = np.array([[90, 90], [10, 10], [90, 10], [10, 90]], dtype=np.float32)
    ordered = order_points(pts)
    assert np.array_equal(ordered[0], [10, 10])
    assert np.array_equal(ordered[1], [90, 10])
    assert np.array_equal(ordered[2], [90, 90])
    assert np.array_equal(ordered[3], [10, 90])


def test_four_point_transform_dims():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    quad = np.array([[10, 10], [90, 10], [90, 90], [10, 90]], dtype=np.float32)
    out = four_point_transform(image, quad)
    assert 79 <= out.shape[0] <= 81
    assert 79 <= out.shape[1] <= 81


def test_detect_document_corners():
    image = np.zeros((300, 400, 3), dtype=np.uint8)
    quad = np.array([[50, 50], [350, 30], [340, 270], [40, 250]], dtype=np.int32)
    cv2.fillConvexPoly(image, quad, (255, 255, 255))
    corners = detect_document_corners(image)
    assert corners is not None
    assert corners.shape == (4, 2)


def test_detect_document_corners_low_contrast():
    # A gray document on a slightly darker background (realistic low contrast).
    image = np.full((300, 400, 3), 100, dtype=np.uint8)
    quad = np.array([[50, 50], [350, 30], [340, 270], [40, 250]], dtype=np.int32)
    cv2.fillConvexPoly(image, quad, (140, 140, 140))
    corners = detect_document_corners(image)
    assert corners is not None
    assert corners.shape == (4, 2)


def test_warp_polygon_many_points():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    pts = np.array(
        [[10, 10], [50, 5], [90, 10], [90, 90], [50, 95], [10, 90]],
        dtype=np.float32,
    )
    out = warp_polygon_to_rectangle(image, pts)
    assert out.ndim == 3 and out.shape[2] == 3


def test_remove_illumination_reduces_gradient():
    height, width = 100, 200
    gradient = np.linspace(50, 200, width, dtype=np.float32)
    image = np.tile(gradient, (height, 1)).astype(np.uint8)
    image = np.repeat(image[..., None], 3, axis=2)

    out = remove_illumination(image, 60)
    orig_diff = abs(float(image[:, 0].mean()) - float(image[:, -1].mean()))
    new_diff = abs(float(out[:, 0].mean()) - float(out[:, -1].mean()))
    assert new_diff < orig_diff


def test_remove_illumination_preserves_hue():
    # A red document with a horizontal illumination gradient.
    height, width = 100, 200
    gradient = np.linspace(0.4, 1.0, width, dtype=np.float32)  # shape (width,)
    image = np.zeros((height, width, 3), dtype=np.float32)
    image[..., 0] = 200 * gradient  # broadcast to (height, width)
    image[..., 1] = 50 * gradient
    image[..., 2] = 50 * gradient
    image = image.astype(np.uint8)

    out = remove_illumination(image, 60)
    mid = out[height // 2, width // 2].astype(int)
    assert mid[0] > mid[1]  # red channel still dominant
    assert mid[0] > mid[2]


def test_remove_illumination_no_block_artifacts():
    # A smooth gradient with small dark "text" squares should become uniform,
    # without square-block discontinuities.
    height, width = 200, 300
    gradient = np.linspace(60, 200, width, dtype=np.float32)
    image = np.tile(gradient[None, :], (height, 1)).astype(np.uint8)
    image = np.repeat(image[..., None], 3, axis=2)
    for (y, x) in [(50, 50), (100, 120), (150, 200)]:
        image[y:y + 10, x:x + 10] = 20  # small dark "text"

    out = remove_illumination(image, 60)
    flat = out[30:45, 30:45].astype(np.float32)
    assert flat.std() < 12  # no blocky discontinuities in a flat region


def test_scan_document_pipeline():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[20:80, 20:80] = 200
    quad = np.array([[20, 20], [80, 20], [80, 80], [20, 80]], dtype=np.float32)
    out = scan_document(image, quad, illumination_strength=30)
    assert out.ndim == 3 and out.shape[2] == 3
