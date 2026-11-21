"""Operation base class and parameter declarations."""
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class ParamSpec:
    """Declarative description of one operation parameter.

    ``kind`` is one of ``"int"``, ``"float"``, ``"bool"`` or ``"choice"``.
    For ``"choice"``, ``choices`` is a list of ``(value, label, label_zh)``
    tuples.
    """

    key: str
    label: str
    label_zh: str
    kind: str = "float"
    default: Any = 0
    minimum: float = 0.0
    maximum: float = 100.0
    step: float = 1.0
    choices: Optional[list] = None


class Operation:
    """Base class for an image filter.

    Subclasses set a unique ``name``, a display ``title`` / ``title_zh`` and a
    list of :class:`ParamSpec` declarations, then implement ``process``.
    """

    name: str = ""
    title: str = ""
    title_zh: str = ""
    params: list[ParamSpec] = []

    def default_params(self) -> dict:
        return {p.key: p.default for p in self.params}

    def process(self, image: np.ndarray, params: dict) -> np.ndarray:
        """Return a new image. Implementations must not mutate ``image``."""
        raise NotImplementedError
