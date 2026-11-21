"""Geometric operations: flip, rotate, resize."""
import cv2

from .base import Operation, ParamSpec


def _rotate_uncropped(image, angle):
    """Rotate by an arbitrary angle, expanding the canvas to avoid cropping.

    Uses Lanczos interpolation for smooth, low-aliasing edges.
    """
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int(round(height * sin + width * cos))
    new_h = int(round(height * cos + width * sin))
    matrix[0, 2] += new_w / 2 - width / 2
    matrix[1, 2] += new_h / 2 - height / 2
    return cv2.warpAffine(
        image, matrix, (new_w, new_h), flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE,
    )


class FlipOperation(Operation):
    name = "flip"
    title = "Flip"
    title_zh = "翻转"
    params = [
        ParamSpec(
            key="mode",
            label="Mode",
            label_zh="方向",
            kind="choice",
            default="horizontal",
            choices=[("horizontal", "Horizontal", "水平"),
                     ("vertical", "Vertical", "垂直"),
                     ("both", "Both", "两者")],
        ),
    ]

    def process(self, image, params):
        code = {"horizontal": 1, "vertical": 0, "both": -1}[
            params.get("mode", "horizontal")
        ]
        return cv2.flip(image, code)


class RotateOperation(Operation):
    name = "rotate"
    title = "Rotate"
    title_zh = "旋转"
    params = [
        ParamSpec(
            key="mode",
            label="Mode",
            label_zh="方式",
            kind="choice",
            default="90",
            choices=[("90", "90°", "90°"), ("180", "180°", "180°"),
                     ("270", "270°", "270°"), ("custom", "Custom", "任意角度")],
        ),
        ParamSpec(
            key="angle",
            label="Angle",
            label_zh="角度",
            kind="float",
            default=0.0,
            minimum=-180,
            maximum=180,
            step=1.0,
        ),
    ]

    def process(self, image, params):
        mode = params.get("mode", "90")
        if mode == "custom":
            angle = float(params.get("angle", 0.0))
            if angle == 0:
                return image
            return _rotate_uncropped(image, angle)
        code = {
            "90": cv2.ROTATE_90_CLOCKWISE,
            "180": cv2.ROTATE_180,
            "270": cv2.ROTATE_90_COUNTERCLOCKWISE,
        }[mode]
        return cv2.rotate(image, code)


class ResizeOperation(Operation):
    name = "resize"
    title = "Resize"
    title_zh = "缩放"
    params = [
        ParamSpec(
            key="scale",
            label="Scale",
            label_zh="比例",
            kind="float",
            default=1.0,
            minimum=0.1,
            maximum=5.0,
            step=0.05,
        ),
    ]

    def process(self, image, params):
        scale = float(params.get("scale", 1.0))
        if scale == 1.0:
            return image
        height, width = image.shape[:2]
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        return cv2.resize(image, (new_w, new_h), interpolation=interp)
