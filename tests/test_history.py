"""Tests for core.history."""
import numpy as np

from core.history import History


def _img(value):
    return np.full((2, 2, 3), value, dtype=np.uint8)


def test_record_undo_redo():
    h = History()
    a, b = _img(0), _img(10)

    h.record(a)
    assert h.can_undo and not h.can_redo

    got = h.undo(b)
    assert np.array_equal(got, a)
    assert h.can_redo

    got2 = h.redo(a)
    assert np.array_equal(got2, b)
    assert h.can_undo and not h.can_redo


def test_new_record_clears_redo():
    h = History()
    a, b = _img(0), _img(10)

    h.record(a)
    h.undo(b)  # undo stack empty, redo has b
    assert h.can_redo

    h.record(b)  # a new change invalidates the redo branch
    assert not h.can_redo


def test_undo_when_empty_returns_none():
    h = History()
    assert h.undo(_img(0)) is None
    assert h.redo(_img(0)) is None
