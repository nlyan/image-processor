"""Tests for core.registry."""
from core.registry import build_default_registry


def test_default_registry_contains_brightness():
    registry = build_default_registry()
    assert "brightness" in registry.names()

    op = registry.get("brightness")
    assert op.name == "brightness"
    assert op.title
    assert op.default_params() == {"delta": 0}


def test_register_and_get():
    from core.operations.brightness import BrightnessOperation
    from core.registry import Registry

    registry = Registry()
    op = BrightnessOperation()
    registry.register(op)
    assert registry.get("brightness") is op
    assert registry.names() == ["brightness"]
