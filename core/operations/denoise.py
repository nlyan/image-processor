"""Denoising operation (bilateral / gaussian / median)."""
import cv2

from .base import Operation, ParamSpec

METHODS = [
    ("bilateral", "Bilateral", "双边"),
    ("gaussian", "Gaussian", "高斯"),
    ("median", "Median", "中值"),
]


class DenoiseOperation(Operation):
    name = "denoise"
    title = "Denoise"
    title_zh = "降噪"
    params = [
        ParamSpec(
            key="method",
            label="Method",
            label_zh="方法",
            kind="choice",
            default="bilateral",
            choices=METHODS,
        ),
        ParamSpec(
            key="strength",
            label="Strength",
            label_zh="强度",
            kind="int",
            default=20,
            minimum=1,
            maximum=100,
            step=1,
        ),
    ]

    def process(self, image, params):
        method = params.get("method", "bilateral")
        strength = int(params.get("strength", 20))

        if method == "gaussian":
            sigma = max(0.1, strength / 10.0)
            return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)

        if method == "median":
            ksize = max(3, (strength // 10) * 2 + 1)
            return cv2.medianBlur(image, ksize)

        # bilateral (default): edge-preserving
        sigma = max(1.0, float(strength))
        return cv2.bilateralFilter(image, 5, sigma, sigma)
