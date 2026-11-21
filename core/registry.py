"""Central registry of image operations."""
from core.operations.base import Operation


class Registry:
    def __init__(self):
        self._ops: dict[str, Operation] = {}

    def register(self, op: Operation) -> None:
        self._ops[op.name] = op

    def get(self, name: str) -> Operation:
        return self._ops[name]

    def all(self) -> list[Operation]:
        return list(self._ops.values())

    def names(self) -> list[str]:
        return list(self._ops.keys())


def build_default_registry() -> Registry:
    """Create a registry populated with all built-in operations."""
    from core.operations.brightness import BrightnessOperation
    from core.operations.color import ColorOperation
    from core.operations.contrast import ContrastOperation
    from core.operations.denoise import DenoiseOperation
    from core.operations.geometry import FlipOperation, ResizeOperation, RotateOperation
    from core.operations.sharpen import SharpenOperation
    from core.operations.watermark_image import ImageWatermarkOperation
    from core.operations.watermark_text import TextWatermarkOperation

    registry = Registry()
    for op in (
        BrightnessOperation(),
        DenoiseOperation(),
        SharpenOperation(),
        ContrastOperation(),
        ColorOperation(),
        TextWatermarkOperation(),
        ImageWatermarkOperation(),
        FlipOperation(),
        RotateOperation(),
        ResizeOperation(),
    ):
        registry.register(op)
    return registry
