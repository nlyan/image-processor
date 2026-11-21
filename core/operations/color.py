"""Color correction: white balance and saturation."""
import cv2
import numpy as np

from .base import Operation, ParamSpec


def _gray_world(image: np.ndarray) -> np.ndarray:
    means = image.astype(np.float64).mean(axis=(0, 1))
    target = means.mean()
    gain = np.where(means > 0, target / means, 1.0)
    out = image.astype(np.float64) * gain
    return np.clip(out, 0, 255).astype(np.uint8)


def _adjust_temperature(image: np.ndarray, temperature: int) -> np.ndarray:
    # temperature in [-100, 100]: positive is warmer (more red, less blue).
    delta = (temperature / 100.0) * 30.0
    out = image.astype(np.float32).copy()
    out[..., 0] += delta  # red
    out[..., 2] -= delta  # blue
    return np.clip(out, 0, 255).astype(np.uint8)


class ColorOperation(Operation):
    name = "color"
    title = "Color"
    title_zh = "色彩"
    params = [
        ParamSpec(
            key="auto_wb",
            label="Auto White Balance",
            label_zh="自动白平衡",
            kind="bool",
            default=False,
        ),
        ParamSpec(
            key="temperature",
            label="Temperature",
            label_zh="色温",
            kind="int",
            default=0,
            minimum=-100,
            maximum=100,
            step=1,
        ),
        ParamSpec(
            key="saturation",
            label="Saturation",
            label_zh="饱和度",
            kind="int",
            default=0,
            minimum=-100,
            maximum=100,
            step=1,
        ),
    ]

    def process(self, image, params):
        out = image
        if bool(params.get("auto_wb", False)):
            out = _gray_world(out)

        temperature = int(params.get("temperature", 0))
        if temperature != 0:
            out = _adjust_temperature(out, temperature)

        saturation = int(params.get("saturation", 0))
        if saturation != 0:
            hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV)
            h_channel, s_channel, v_channel = cv2.split(hsv)
            factor = 1.0 + saturation / 100.0
            s_channel = np.clip(
                s_channel.astype(np.float32) * factor, 0, 255
            ).astype(np.uint8)
            out = cv2.cvtColor(
                cv2.merge([h_channel, s_channel, v_channel]), cv2.COLOR_HSV2RGB
            )
        return out
