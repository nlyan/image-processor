"""Background removal dialog (GrabCut + color threshold + region selection)."""
import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal, QPointF, QRectF
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

import i18n
from core.background import (
    build_mask,
    composite,
    connected_regions,
    grabcut_segment,
    threshold_segment,
)
from core.io import array_to_qimage
from core.preview import downsample_image


def _tr(key):
    return i18n.tr(key)


class BackgroundCanvas(QWidget):
    """Canvas for drawing a GrabCut rectangle / picking a color / selecting regions."""

    MODE_GRABCUT = "grabcut"
    MODE_THRESHOLD = "threshold"
    MODE_SELECT = "select"

    rectSelected = Signal(int, int, int, int)
    pointSelected = Signal(int, int)
    regionClicked = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._qimage = None
        self._labels = None
        self._mode = self.MODE_GRABCUT
        self._scale = 1.0
        self._offset = (0.0, 0.0)
        self._drag_start = None
        self._drag_end = None
        self.setMouseTracking(True)
        self.setMinimumSize(420, 320)

    def set_mode(self, mode) -> None:
        self._mode = mode
        self._drag_start = None
        self._drag_end = None
        self.update()

    def set_labels(self, labels) -> None:
        self._labels = labels

    def set_image(self, qimage) -> None:
        self._qimage = qimage
        self._recompute_scale()
        self.update()

    def _recompute_scale(self) -> None:
        if self._qimage is None or self.width() == 0 or self.height() == 0:
            return
        iw, ih = self._qimage.width(), self._qimage.height()
        self._scale = min(self.width() / iw, self.height() / ih)
        self._offset = (
            (self.width() - iw * self._scale) / 2,
            (self.height() - ih * self._scale) / 2,
        )

    def resizeEvent(self, event) -> None:
        self._recompute_scale()
        super().resizeEvent(event)
        self.update()

    def _to_image(self, pos):
        return (
            (pos.x() - self._offset[0]) / self._scale,
            (pos.y() - self._offset[1]) / self._scale,
        )

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(45, 45, 45))
        if self._qimage is None:
            return
        target = QRectF(
            self._offset[0], self._offset[1],
            self._qimage.width() * self._scale,
            self._qimage.height() * self._scale,
        )
        painter.drawImage(target, self._qimage)
        if self._mode == self.MODE_GRABCUT and self._drag_start is not None:
            painter.setPen(QPen(QColor(0, 255, 0), 2))
            painter.drawRect(QRectF(self._drag_start, self._drag_end).normalized())

    def mousePressEvent(self, event) -> None:
        if self._qimage is None or event.button() != Qt.MouseButton.LeftButton:
            return
        if self._mode == self.MODE_GRABCUT:
            self._drag_start = event.position()
            self._drag_end = event.position()
        elif self._mode == self.MODE_THRESHOLD:
            x, y = self._to_image(event.position())
            self.pointSelected.emit(int(x), int(y))
        elif self._mode == self.MODE_SELECT and self._labels is not None:
            x, y = self._to_image(event.position())
            xi = int(np.clip(x, 0, self._labels.shape[1] - 1))
            yi = int(np.clip(y, 0, self._labels.shape[0] - 1))
            self.regionClicked.emit(int(self._labels[yi, xi]))

    def mouseMoveEvent(self, event) -> None:
        if self._mode == self.MODE_GRABCUT and self._drag_start is not None:
            self._drag_end = event.position()
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if self._mode != self.MODE_GRABCUT or self._drag_start is None:
            return
        start = self._to_image(self._drag_start)
        end = self._to_image(self._drag_end)
        x = int(min(start[0], end[0]))
        y = int(min(start[1], end[1]))
        w = int(abs(start[0] - end[0]))
        h = int(abs(start[1] - end[1]))
        self._drag_start = None
        self._drag_end = None
        self.update()
        if w > 5 and h > 5:
            self.rectSelected.emit(x, y, w, h)


class BackgroundDialog(QDialog):
    def __init__(self, image, parent=None):
        super().__init__(parent)
        self._image = image
        self._fg_mask = None
        self._labels = None
        self._result = None
        self._keep = {}
        self._num = 0
        self._last_point = None
        self._display_scale = 1.0
        self.setWindowTitle(_tr("remove_bg"))
        self.resize(1100, 720)

        self._canvas = BackgroundCanvas()
        self._canvas.set_image(array_to_qimage(downsample_image(image, 1200)))
        self._canvas.rectSelected.connect(self._on_rect)
        self._canvas.pointSelected.connect(self._on_point)
        self._canvas.regionClicked.connect(self._on_region_clicked)

        self._grabcut_radio = QRadioButton(_tr("grabcut"))
        self._threshold_radio = QRadioButton(_tr("threshold"))
        self._grabcut_radio.setChecked(True)
        self._method_group = QButtonGroup(self)
        self._method_group.addButton(self._grabcut_radio)
        self._method_group.addButton(self._threshold_radio)
        self._grabcut_radio.toggled.connect(self._on_method_changed)

        self._hint_label = QLabel(_tr("grabcut_hint"))
        self._hint_label.setWordWrap(True)

        self._tol_label = QLabel(f"{_tr('tolerance')}: 40")
        self._tol_slider = QSlider(Qt.Orientation.Horizontal)
        self._tol_slider.setRange(5, 100)
        self._tol_slider.setValue(40)
        self._tol_slider.valueChanged.connect(self._on_tolerance_changed)
        self._tol_slider.setEnabled(False)

        self._white_radio = QRadioButton(_tr("bg_white"))
        self._transparent_radio = QRadioButton(_tr("bg_transparent"))
        self._white_radio.setChecked(True)
        self._bg_group = QButtonGroup(self)
        self._bg_group.addButton(self._white_radio)
        self._bg_group.addButton(self._transparent_radio)
        self._white_radio.toggled.connect(self._update_result)

        self._keep_all_btn = QPushButton(_tr("keep_all"))
        self._keep_largest_btn = QPushButton(_tr("keep_largest"))
        self._remove_all_btn = QPushButton(_tr("remove_all"))
        self._run_again_btn = QPushButton(_tr("run_again"))
        self._keep_all_btn.clicked.connect(self._keep_all)
        self._keep_largest_btn.clicked.connect(self._keep_largest)
        self._remove_all_btn.clicked.connect(self._remove_all)
        self._run_again_btn.clicked.connect(self._run_again)

        self._apply_btn = QPushButton(_tr("apply"))
        self._apply_btn.clicked.connect(self._apply)
        self._cancel_btn = QPushButton(_tr("cancel"))
        self._cancel_btn.clicked.connect(self.reject)

        controls = QVBoxLayout()
        controls.addWidget(QLabel(_tr("bg_method")))
        controls.addWidget(self._grabcut_radio)
        controls.addWidget(self._threshold_radio)
        controls.addSpacing(8)
        controls.addWidget(self._hint_label)
        controls.addWidget(self._tol_label)
        controls.addWidget(self._tol_slider)
        controls.addSpacing(8)
        controls.addWidget(QLabel(_tr("bg_output")))
        controls.addWidget(self._white_radio)
        controls.addWidget(self._transparent_radio)
        controls.addSpacing(8)
        controls.addWidget(self._keep_all_btn)
        controls.addWidget(self._keep_largest_btn)
        controls.addWidget(self._remove_all_btn)
        controls.addWidget(self._run_again_btn)
        controls.addStretch(1)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self._apply_btn)
        buttons.addWidget(self._cancel_btn)

        top = QHBoxLayout()
        top.addWidget(self._canvas, 1)
        top.addLayout(controls, 0)

        main = QVBoxLayout(self)
        main.addLayout(top, 1)
        main.addLayout(buttons, 0)

    # ---- Segmentation ----------------------------------------------------

    def _on_method_changed(self) -> None:
        if self._grabcut_radio.isChecked():
            self._canvas.set_mode(BackgroundCanvas.MODE_GRABCUT)
            self._hint_label.setText(_tr("grabcut_hint"))
            self._tol_slider.setEnabled(False)
        else:
            self._canvas.set_mode(BackgroundCanvas.MODE_THRESHOLD)
            self._hint_label.setText(_tr("threshold_hint"))
            self._tol_slider.setEnabled(True)
        self._last_point = None
        self._fg_mask = None
        self._labels = None
        self._keep = {}
        self._canvas.set_labels(None)
        self._canvas.set_image(array_to_qimage(downsample_image(self._image, 1200)))
        self._update_result()

    def _on_rect(self, x, y, w, h) -> None:
        # map display coords back to full-res coords
        scale = self._display_scale
        rect = (int(x / scale), int(y / scale), int(w / scale), int(h / scale))
        mask = grabcut_segment(self._image, rect)
        self._finish_segmentation(mask)

    def _on_point(self, x, y) -> None:
        scale = self._display_scale
        fx, fy = int(x / scale), int(y / scale)
        self._last_point = (fx, fy)
        mask = threshold_segment(self._image, (fx, fy), self._tol_slider.value())
        self._finish_segmentation(mask)

    def _on_tolerance_changed(self, value) -> None:
        self._tol_label.setText(f"{_tr('tolerance')}: {value}")
        if self._threshold_radio.isChecked() and self._last_point is not None:
            mask = threshold_segment(self._image, self._last_point, value)
            self._finish_segmentation(mask)

    def _finish_segmentation(self, mask) -> None:
        self._fg_mask = mask
        self._num, self._labels = connected_regions(mask)
        self._keep = {label: True for label in range(1, self._num)}
        self._canvas.set_mode(BackgroundCanvas.MODE_SELECT)
        self._hint_label.setText(_tr("select_hint"))
        self._show_overlay()
        self._update_result()

    # ---- Region selection ------------------------------------------------

    def _on_region_clicked(self, label) -> None:
        if label <= 0:
            return
        self._keep[label] = not self._keep.get(label, True)
        self._show_overlay()
        self._update_result()

    def _keep_all(self) -> None:
        if self._labels is None:
            return
        self._keep = {label: True for label in range(1, self._num)}
        self._show_overlay()
        self._update_result()

    def _remove_all(self) -> None:
        if self._labels is None:
            return
        self._keep = {label: False for label in range(1, self._num)}
        self._show_overlay()
        self._update_result()

    def _keep_largest(self) -> None:
        if self._labels is None:
            return
        sizes = {}
        for label in range(1, self._num):
            sizes[label] = int(np.count_nonzero(self._labels == label))
        largest = max(sizes, key=sizes.get) if sizes else 0
        self._keep = {label: (label == largest) for label in range(1, self._num)}
        self._show_overlay()
        self._update_result()

    def _show_overlay(self) -> None:
        small = downsample_image(self._image, 1200)
        h, w = small.shape[:2]
        self._display_scale = w / self._image.shape[1]
        overlay = small.copy().astype(np.float32)
        display_labels = cv2.resize(
            self._labels, (w, h), interpolation=cv2.INTER_NEAREST
        )
        self._canvas.set_labels(display_labels)
        kept = build_mask(display_labels, self._keep) > 0
        fg_all = display_labels > 0
        overlay[kept] = overlay[kept] * 0.5 + np.array([0, 255, 0]) * 0.5
        removed = fg_all & ~kept
        overlay[removed] = overlay[removed] * 0.5 + np.array([255, 0, 0]) * 0.5
        self._canvas.set_image(array_to_qimage(overlay.astype(np.uint8)))

    # ---- Output ----------------------------------------------------------

    def _update_result(self) -> None:
        if self._labels is None:
            self._result = None
            return
        mask = build_mask(self._labels, self._keep)
        background = "transparent" if self._transparent_radio.isChecked() else "white"
        self._result = composite(self._image, mask, background)

    def _run_again(self) -> None:
        if self._grabcut_radio.isChecked():
            self._grabcut_radio.setChecked(False)
            self._grabcut_radio.setChecked(True)
        else:
            self._threshold_radio.setChecked(False)
            self._threshold_radio.setChecked(True)
        self._on_method_changed()

    def _apply(self) -> None:
        if self._result is None:
            QMessageBox.warning(self, _tr("remove_bg"), _tr("please_segment"))
            return
        self.accept()

    def result(self):
        return self._result
