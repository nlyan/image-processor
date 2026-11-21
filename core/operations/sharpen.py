"""Sharpening via unsharp mask."""
import cv2
import numpy as np

from .base import Operation, ParamSpec


class SharpenOperation(Operation):
    name = "sharpen"
    title = "Sharpen"
    title_zh = "锐化"
    params = [
        ParamSpec(
            key="amount",
            label="Amount",
            label_zh="强度",
            kind="float",
            default=100.0,
            minimum=0.0,
            maximum=500.0,
            step=5.0,
        ),
        ParamSpec(
            key="radius",
            label="Radius",
            label_zh="半径",
            kind="float",
            default=2.0,
            minimum=0.1,
            maximum=20.0,
            step=0.1,
        ),
        ParamSpec(
            key="threshold",
            label="Threshold",
            label_zh="阈值",
            kind="float",
            default=0.0,
            minimum=0.0,
            maximum=50.0,
            step=0.5,
        ),
    ]

    def process(self, image, params):
        amount = float(params.get("amount", 100.0))
        radius = float(params.get("radius", 2.0))
        threshold = float(params.get("threshold", 0.0))
        if amount == 0:
            return image

        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=radius)
        detail = image.astype(np.float32) - blurred.astype(np.float32)

        if threshold > 0:
            magnitude = np.abs(detail)
            if magnitude.ndim == 3:
                magnitude = magnitude.max(axis=2)
            mask = (magnitude >= threshold).astype(np.float32)
            detail = detail * (mask[..., None] if detail.ndim == 3 else mask)

        out = image.astype(np.float32) + (amount / 100.0) * detail
        return np.clip(out, 0, 255).astype(np.uint8)
