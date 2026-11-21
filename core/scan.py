"""Document scanning: boundary detection, perspective correction, illumination removal.

All algorithms are classic computer vision (no machine learning).
"""
import cv2
import numpy as np


def order_points(points):
    """Order 4 points as [top-left, top-right, bottom-right, bottom-left]."""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def four_point_transform(image, points):
    """Warp a quadrilateral (4 points) to a top-down rectangle."""
    rect = order_points(points)
    tl, tr, br, bl = rect

    width_bottom = np.linalg.norm(br - bl)
    width_top = np.linalg.norm(tr - tl)
    max_w = max(1, int(round(max(width_top, width_bottom))))

    height_right = np.linalg.norm(tr - br)
    height_left = np.linalg.norm(tl - bl)
    max_h = max(1, int(round(max(height_left, height_right))))

    dst = np.array(
        [[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (max_w, max_h))


def _auto_canny(gray, sigma=0.33):
    """Compute Canny thresholds from the median pixel intensity.

    Fixed thresholds fail on real photos whose contrast varies; median-based
    thresholds adapt to each image.
    """
    v = np.median(gray)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(gray, lower, upper)


def _approx_quad(contour):
    """Return 4 ordered corners if the contour approximates a convex quad."""
    for epsilon in (0.01, 0.02, 0.03, 0.05):
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return order_points(approx.reshape(4, 2).astype(np.float32))
    return None


def detect_document_corners(image):
    """Return ordered 4 corners of the document, or None.

    Tries, in order: Otsu thresholding (robust to bright or dark documents),
    adaptive-Canny edges, and a minimum-area-rectangle fallback.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    height, width = gray.shape
    image_area = height * width
    min_area = image_area * 0.15

    # Strategy 1: Otsu thresholding (separates document from background).
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    for thresh in (binary, cv2.bitwise_not(binary)):
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            if cv2.contourArea(contour) < min_area:
                break
            corners = _approx_quad(contour)
            if corners is not None:
                return corners

    # Strategy 2: Canny edges.
    edged = _auto_canny(gray)
    edged = cv2.morphologyEx(
        edged, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    )
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:20]:
        if cv2.contourArea(contour) < min_area:
            break
        corners = _approx_quad(contour)
        if corners is not None:
            return corners

    # Strategy 3 (fallback): min-area rectangle of the largest thresholded blob.
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > min_area:
            rect = cv2.minAreaRect(largest)
            return order_points(cv2.boxPoints(rect).astype(np.float32))
    return None


def warp_polygon_to_rectangle(image, points):
    """Warp any polygon (>=4 points) to a top-down rectangle."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 4:
        raise ValueError("Need at least 4 points")
    if len(pts) == 4:
        quad = order_points(pts)
    else:
        hull = cv2.convexHull(pts.astype(np.int32)).reshape(-1, 2)
        rect = cv2.minAreaRect(hull)
        quad = order_points(cv2.boxPoints(rect).astype(np.float32))
    return four_point_transform(image, quad)


def remove_illumination(image, strength):
    """Remove uneven illumination via grayscale background division.

    The background is estimated by a morphological closing on the *grayscale*
    image, which removes text/details while keeping the smooth illumination
    field. All color channels are divided by this single field, preserving hue
    (unlike dividing each channel by its own background, which washes colors
    out). ``strength`` (0-100) controls the closing kernel size.
    """
    if strength <= 0:
        return image
    height, width = image.shape[:2]
    k = max(15, int((strength / 100.0) * (max(height, width) / 8.0)))
    if k % 2 == 0:
        k += 1

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Elliptical kernel avoids the square-block artifacts of a rectangular one.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    # Smooth the background field to eliminate any residual staircase steps.
    background = cv2.GaussianBlur(background, (0, 0), sigmaX=max(1.0, k / 4.0))
    background = background.astype(np.float32)

    img = image.astype(np.float32)
    corrected = img / (background[..., None] + 1e-6) * 255.0
    return np.clip(corrected, 0, 255).astype(np.uint8)


def scan_document(image, points, illumination_strength=0):
    """Full pipeline: warp polygon to a rectangle, then normalize illumination."""
    warped = warp_polygon_to_rectangle(image, points)
    if illumination_strength > 0:
        warped = remove_illumination(warped, illumination_strength)
    return warped
