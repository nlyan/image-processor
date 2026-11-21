"""Contrast adjustment (linear stretch + gamma + optional CLAHE)."""
import cv2
import numpy as np

from .base import Operation, ParamSpec


class ContrastOperation(Operation):
    name = "contrast"
    title = "Contrast"
    title_zh = "对比度"
    params = [
        ParamSpec(
            key="contrast",
            label="Contrast",
            label_zh="对比度",
            kind="float",
            default=1.0,
            minimum=0.1,
            maximum=3.0,
            step=0.05,
        ),
        ParamSpec(
            key="gamma",
            label="Gamma",
            label_zh="伽马",
            kind="float",
            default=1.0,
            minimum=0.3,
            maximum=3.0,
            step=0.05,
        ),
        ParamSpec(
            key="clahe",
            label="CLAHE (local enhance)",
            label_zh="局部增强",
            kind="bool",
            default=False,
        ),
    ]

    def process(self, image, params):
        contrast = float(params.get("contrast", 1.0))
        gamma = float(params.get("gamma", 1.0))
        use_clahe = bool(params.get("clahe", False))

        out = image
        if contrast != 1.0:
            out = cv2.convertScaleAbs(out, alpha=contrast, beta=0)
        if gamma != 1.0:
            lut = np.array(
                [255.0 * (i / 255.0) ** (1.0 / gamma) for i in range(256)],
                dtype=np.uint8,
            )
            out = cv2.LUT(out, lut)
        if use_clahe:
            lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_channel = clahe.apply(l_channel)
            out = cv2.cvtColor(
                cv2.merge([l_channel, a_channel, b_channel]), cv2.COLOR_LAB2RGB
            )
        return out
