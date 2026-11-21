"""Demo brightness operation (exercises the processing framework)."""
import numpy as np

from .base import Operation, ParamSpec


class BrightnessOperation(Operation):
    name = "brightness"
    title = "Brightness"
    title_zh = "亮度"
    params = [
        ParamSpec(
            key="delta",
            label="Brightness",
            label_zh="亮度",
            kind="int",
            default=0,
            minimum=-100,
            maximum=100,
            step=1,
        )
    ]

    def process(self, image: np.ndarray, params: dict) -> np.ndarray:
        delta = int(params.get("delta", 0))
        if delta == 0:
            return image
        out = image.astype(np.int16) + delta
        return np.clip(out, 0, 255).astype(np.uint8)
