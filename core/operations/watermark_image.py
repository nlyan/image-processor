"""Image (logo) watermark with alpha blending."""
import os

import cv2
import numpy as np

from core.io import load_image_alpha
from .base import Operation, ParamSpec
from .watermark_common import POSITIONS, anchor_position


class ImageWatermarkOperation(Operation):
    name = "image_watermark"
    title = "Image Watermark"
    title_zh = "图片水印"
    params = [
        ParamSpec(key="logo", label="Logo", label_zh="Logo 文件", kind="file", default=""),
        ParamSpec(key="scale", label="Scale", label_zh="缩放", kind="float", default=0.2,
                  minimum=0.05, maximum=1.0, step=0.05),
        ParamSpec(key="opacity", label="Opacity", label_zh="透明度", kind="int", default=50,
                  minimum=0, maximum=100, step=1),
        ParamSpec(key="rotation", label="Rotation", label_zh="旋转", kind="float", default=0.0,
                  minimum=-180, maximum=180, step=1.0),
        ParamSpec(key="position", label="Position", label_zh="位置", kind="choice",
                  default="bottom-right", choices=POSITIONS),
        ParamSpec(key="margin", label="Margin", label_zh="边距", kind="int", default=10,
                  minimum=0, maximum=200, step=1),
    ]

    def process(self, image, params):
        logo_path = (params.get("logo") or "").strip()
        if not logo_path or not os.path.exists(logo_path):
            return image
        try:
            logo = load_image_alpha(logo_path)
        except Exception:  # noqa: BLE001 - a missing/broken logo is a no-op
            return image

        scale = float(params.get("scale", 0.2))
        opacity = float(params.get("opacity", 50)) / 100.0
        rotation = float(params.get("rotation", 0.0))
        position = params.get("position", "bottom-right")
        margin = int(params.get("margin", 10))

        logo_h, logo_w = logo.shape[:2]
        new_w = max(1, int(round(logo_w * scale)))
        new_h = max(1, int(round(logo_h * scale)))
        logo = cv2.resize(logo, (new_w, new_h), interpolation=cv2.INTER_AREA)

        if rotation != 0:
            matrix = cv2.getRotationMatrix2D((new_w / 2, new_h / 2), rotation, 1.0)
            logo = cv2.warpAffine(
                logo, matrix, (new_w, new_h), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0),
            )

        x, y = anchor_position(position, image.shape[1], image.shape[0], new_w, new_h, margin)
        x = max(0, min(x, image.shape[1] - new_w))
        y = max(0, min(y, image.shape[0] - new_h))

        out = image.copy()
        roi = out[y:y + new_h, x:x + new_w]
        if logo.shape[2] == 4:
            logo_rgb = logo[..., :3].astype(np.float32)
            alpha = (logo[..., 3:4].astype(np.float32) / 255.0) * opacity
            blended = roi.astype(np.float32) * (1.0 - alpha) + logo_rgb * alpha
            out[y:y + new_h, x:x + new_w] = np.clip(blended, 0, 255).astype(np.uint8)
        else:
            blended = cv2.addWeighted(roi, 1.0 - opacity, logo, opacity, 0)
            out[y:y + new_h, x:x + new_w] = blended
        return out
