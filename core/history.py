"""Undo / redo history of image states."""
import numpy as np


class History:
    """Snapshot-based undo/redo stack.

    Call ``record(image)`` with the current image just before a new change is
    applied. Operations are functional (they return new arrays), so the stored
    references stay valid without copying.
    """

    def __init__(self, max_depth: int = 20):
        self._undo: list[np.ndarray] = []
        self._redo: list[np.ndarray] = []
        self._max_depth = max_depth

    def record(self, image: np.ndarray) -> None:
        self._undo.append(image)
        if len(self._undo) > self._max_depth:
            self._undo.pop(0)
        self._redo.clear()

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self, current: np.ndarray):
        if not self._undo:
            return None
        self._redo.append(current)
        return self._undo.pop()

    def redo(self, current: np.ndarray):
        if not self._redo:
            return None
        self._undo.append(current)
        return self._redo.pop()

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()
