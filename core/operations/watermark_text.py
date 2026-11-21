"""Text watermark (drawn with Pillow so Chinese text is supported)."""
import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFont

from .base import Operation, ParamSpec
from .watermark_common import POSITIONS, anchor_position

COLOR_MAP = {
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "gray": (128, 128, 128),
}


def _load_font(size):
    for path in (
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msyh.ttf",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


class TextWatermarkOperation(Operation):
    name = "text_watermark"
    title = "Text Watermark"
    title_zh = "文字水印"
    params = [
        ParamSpec(key="text", label="Text", label_zh="文字内容", kind="text", default=""),
        ParamSpec(key="size", label="Font Size", label_zh="字号", kind="int", default=32,
                  minimum=8, maximum=200, step=1),
        ParamSpec(key="color", label="Color", label_zh="颜色", kind="choice", default="white",
                  choices=[(k, v, k) for k, v in [("white", "White"), ("black", "Black"),
                    ("red", "Red"), ("green", "Green"), ("blue", "Blue"),
                    ("yellow", "Yellow"), ("gray", "Gray")]]),
        ParamSpec(key="opacity", label="Opacity", label_zh="透明度", kind="int", default=50,
                  minimum=0, maximum=100, step=1),
        ParamSpec(key="position", label="Position", label_zh="位置", kind="choice",
                  default="bottom-right", choices=POSITIONS),
        ParamSpec(key="margin", label="Margin", label_zh="边距", kind="int", default=10,
                  minimum=0, maximum=200, step=1),
    ]

    def process(self, image, params):
        text = (params.get("text") or "").strip()
        if not text:
            return image
        size = int(params.get("size", 32))
        opacity = int(params.get("opacity", 50)) / 100.0
        position = params.get("position", "bottom-right")
        margin = int(params.get("margin", 10))
        rgb = COLOR_MAP.get(params.get("color", "white"), (255, 255, 255))

        pil = PILImage.fromarray(image)
        draw = ImageDraw.Draw(pil)
        font = _load_font(size)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x, y = anchor_position(position, pil.width, pil.height, tw, th, margin)

        overlay = PILImage.new("RGBA", pil.size, (0, 0, 0, 0))
        odraw = ImageDraw.Draw(overlay)
        odraw.text((x - bbox[0], y - bbox[1]), text, font=font,
                   fill=(*rgb, int(255 * opacity)))
        composited = PILImage.alpha_composite(pil.convert("RGBA"), overlay)
        return np.array(composited.convert("RGB"), dtype=np.uint8)
