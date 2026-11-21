"""Shared helpers for watermark operations."""

POSITIONS = [
    ("top-left", "Top Left", "左上"),
    ("top-center", "Top Center", "上中"),
    ("top-right", "Top Right", "右上"),
    ("center-left", "Center Left", "左中"),
    ("center", "Center", "居中"),
    ("center-right", "Center Right", "右中"),
    ("bottom-left", "Bottom Left", "左下"),
    ("bottom-center", "Bottom Center", "下中"),
    ("bottom-right", "Bottom Right", "右下"),
]


def anchor_position(position, container_w, container_h, item_w, item_h, margin):
    """Return the top-left (x, y) of an item inside a container for a 9-grid position."""
    parts = position.split("-")
    vertical = parts[0] if parts else "center"
    horizontal = parts[1] if len(parts) > 1 else "center"

    if vertical == "top":
        y = margin
    elif vertical == "bottom":
        y = container_h - item_h - margin
    else:
        y = (container_h - item_h) // 2

    if horizontal == "left":
        x = margin
    elif horizontal == "right":
        x = container_w - item_w - margin
    else:
        x = (container_w - item_w) // 2
    return x, y
