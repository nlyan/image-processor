"""Auto-generated parameter panel for an operation."""
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QWidget,
)

import i18n
from core.operations.base import Operation


class FileEdit(QWidget):
    """A line edit with a browse button, used for file-path parameters."""

    textChanged = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._edit = QLineEdit()
        self._edit.textChanged.connect(self.textChanged)
        self._browse = QPushButton("Browse")
        self._browse.clicked.connect(self._pick)
        layout.addWidget(self._edit)
        layout.addWidget(self._browse)

    def text(self) -> str:
        return self._edit.text()

    def setText(self, value: str) -> None:
        self._edit.setText(value)

    def _pick(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)",
        )
        if path:
            self._edit.setText(path)


class ParamPanel(QWidget):
    """Renders an operation's :class:`ParamSpec` list as widgets."""

    valuesChanged = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QFormLayout(self)
        self._widgets = {}
        self._specs = []
        self._op = None

    def set_operation(self, op: Operation) -> None:
        self._op = op
        self._clear()
        self._specs = list(op.params)
        for spec in self._specs:
            self._add_spec(spec)

    def retranslate(self) -> None:
        """Rebuild widgets with the current language labels, keeping values."""
        if self._op is None:
            return
        current = self.values()
        self.set_operation(self._op)
        for spec in self._specs:
            if spec.key in current:
                widget = self._widgets[spec.key]
                widget.blockSignals(True)
                self._apply_value(widget, spec, current[spec.key])
                widget.blockSignals(False)

    def _clear(self) -> None:
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._widgets = {}

    def _add_spec(self, spec) -> None:
        widget = self._make_widget(spec)
        widget.blockSignals(True)
        self._apply_value(widget, spec, spec.default)
        widget.blockSignals(False)
        self._connect(widget, spec)
        widget.setToolTip(self._tooltip(spec))
        self._layout.addRow(self._label_text(spec), widget)
        self._widgets[spec.key] = widget

    def _make_widget(self, spec):
        if spec.kind == "int":
            w = QSpinBox()
            w.setRange(int(spec.minimum), int(spec.maximum))
            w.setSingleStep(max(1, int(spec.step)))
            return w
        if spec.kind == "float":
            w = QDoubleSpinBox()
            w.setRange(spec.minimum, spec.maximum)
            w.setSingleStep(spec.step)
            w.setDecimals(2)
            return w
        if spec.kind == "bool":
            return QCheckBox()
        if spec.kind == "choice":
            w = QComboBox()
            for value, label_en, label_zh in spec.choices:
                w.addItem(label_en if i18n.LANG == "en" else label_zh, value)
            return w
        if spec.kind == "text":
            return QLineEdit()
        if spec.kind == "file":
            return FileEdit()
        raise ValueError(f"Unsupported param kind: {spec.kind}")

    def _connect(self, widget, spec) -> None:
        if spec.kind in ("int", "float"):
            widget.valueChanged.connect(self._emit_values)
        elif spec.kind == "bool":
            widget.toggled.connect(self._emit_values)
        elif spec.kind == "choice":
            widget.currentIndexChanged.connect(self._emit_values)
        elif spec.kind in ("text", "file"):
            widget.textChanged.connect(self._emit_values)

    def _apply_value(self, widget, spec, value) -> None:
        if spec.kind == "int":
            widget.setValue(int(value))
        elif spec.kind == "float":
            widget.setValue(float(value))
        elif spec.kind == "bool":
            widget.setChecked(bool(value))
        elif spec.kind == "choice":
            index = widget.findData(value)
            if index >= 0:
                widget.setCurrentIndex(index)
        elif spec.kind in ("text", "file"):
            widget.setText(str(value))

    def values(self) -> dict:
        result = {}
        for spec in self._specs:
            result[spec.key] = self._read_value(self._widgets[spec.key], spec)
        return result

    def _read_value(self, widget, spec):
        if spec.kind in ("int", "float"):
            return widget.value()
        if spec.kind == "bool":
            return widget.isChecked()
        if spec.kind == "choice":
            return widget.currentData()
        if spec.kind in ("text", "file"):
            return widget.text()
        raise ValueError(f"Unsupported param kind: {spec.kind}")

    def set_values(self, values: dict) -> None:
        for spec in self._specs:
            if spec.key in values:
                widget = self._widgets[spec.key]
                widget.blockSignals(True)
                self._apply_value(widget, spec, values[spec.key])
                widget.blockSignals(False)
        self._emit_values()

    def reset(self) -> None:
        if self._op is not None:
            self.set_values(self._op.default_params())

    def _label_text(self, spec) -> str:
        label = spec.label if i18n.LANG == "en" else spec.label_zh
        if spec.kind == "int":
            return f"{label} ({int(spec.minimum)}–{int(spec.maximum)})"
        if spec.kind == "float":
            return f"{label} ({spec.minimum:g}–{spec.maximum:g})"
        return label

    def _tooltip(self, spec) -> str:
        if spec.kind == "int":
            return f"Range: {int(spec.minimum)}–{int(spec.maximum)}"
        if spec.kind == "float":
            return f"Range: {spec.minimum:g}–{spec.maximum:g}"
        if spec.kind == "bool":
            return "Toggle on/off"
        if spec.kind == "choice":
            labels = [label_en if i18n.LANG == "en" else label_zh
                      for _v, label_en, label_zh in spec.choices]
            return "Options: " + ", ".join(labels)
        if spec.kind == "text":
            return "Enter text"
        if spec.kind == "file":
            return "Path to an image file"
        return spec.label

    def _emit_values(self, *_args) -> None:
        self.valuesChanged.emit(self.values())
