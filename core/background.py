"""Background removal: GrabCut, color thresholding, region selection.

No deep learning. GrabCut uses a classic graph-cut with a Gaussian mixture
color model; color thresholding is pure pixel classification in HSV space.
"""
import cv2
import numpy as np


def grabcut_segment(image, rect, iterations=5):
    """Run GrabCut initialized by a rectangle; return a foreground mask (0/255).

    ``rect`` is ``(x, y, w, h)`` in image coordinates.
    """
    x, y, w, h = (int(rect[0]), int(rect[1]), max(1, int(rect[2])), max(1, int(rect[3])))
    mask = np.zeros(image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask, (x, y, w, h), bgd_model, fgd_model, iterations,
                cv2.GC_INIT_WITH_RECT)
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
    return fg


def threshold_segment(image, point, tolerance=40):
    """Segment by background color thresholding in HSV; return fg mask (0/255).

    ``point`` is ``(x, y)`` on a background pixel.
    """
    x = int(np.clip(point[0], 0, image.shape[1] - 1))
    y = int(np.clip(point[1], 0, image.shape[0] - 1))
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    h_bg, s_bg, v_bg = int(h[y, x]), int(s[y, x]), int(v[y, x])

    h_dist = np.abs(h.astype(np.int32) - h_bg)
    h_dist = np.minimum(h_dist, 180 - h_dist)
    s_dist = np.abs(s.astype(np.int32) - s_bg)
    v_dist = np.abs(v.astype(np.int32) - v_bg)

    bg = (h_dist < tolerance) & (s_dist < tolerance * 2) & (v_dist < tolerance * 2)
    fg = np.where(bg, 0, 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
    return fg


def connected_regions(mask):
    """Return ``(num_labels, labels)`` for a 0/255 mask (label 0 = background)."""
    binary = (mask > 0).astype(np.uint8)
    return cv2.connectedComponents(binary, connectivity=8)


def build_mask(labels, keep):
    """Build a 0/255 mask from the kept region labels.

    ``keep`` maps a non-zero label to True (keep) or False (remove).
    """
    result = np.zeros(labels.shape, np.uint8)
    for label, do_keep in keep.items():
        if do_keep and label > 0:
            result[labels == label] = 255
    return result


def composite(image, mask, background="white"):
    """Composite the foreground (mask > 0) over white or transparent.

    Returns an RGB image for "white" or an RGBA image for "transparent".
    """
    fg = mask > 0
    if background == "white":
        result = np.full_like(image, 255)
        result[fg] = image[fg]
        return result
    result = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    result[fg, :3] = image[fg]
    result[fg, 3] = 255
    return result
