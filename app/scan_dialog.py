"""Document scanning dialog with interactive boundary selection."""
from PySide6.QtCore import Qt, Signal, QPointF, QRectF
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from core.io import array_to_qimage
from core.preview import downsample_image
from core.scan import detect_document_corners, scan_document


class ScanCanvas(QWidget):
    """Displays an image and lets the user place/drag boundary points."""

    pointsChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._qimage = None
        self._points = []      # list of [x, y] in image coordinates
        self._scale = 1.0
        self._offset = (0.0, 0.0)
        self._dragging = None
        self.setMouseTracking(True)
        self.setMinimumSize(420, 320)

    def set_image(self, qimage) -> None:
        self._qimage = qimage
        self._points = []
        self._dragging = None
        self._recompute_scale()
        self.update()

    def set_points(self, points) -> None:
        self._points = [list(map(float, p)) for p in points]
        self._dragging = None
        self.update()
        self.pointsChanged.emit()

    def points(self):
        return [list(p) for p in self._points]

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

    def _to_widget(self, x, y):
        return (
            x * self._scale + self._offset[0],
            y * self._scale + self._offset[1],
        )

    def _clamp(self, img_x, img_y):
        iw, ih = self._qimage.width(), self._qimage.height()
        return (
            max(0.0, min(iw - 1.0, img_x)),
            max(0.0, min(ih - 1.0, img_y)),
        )

    def _nearest_point(self, img_x, img_y):
        radius_img = 12 / self._scale
        best, best_d = None, radius_img
        for i, (x, y) in enumerate(self._points):
            d = ((x - img_x) ** 2 + (y - img_y) ** 2) ** 0.5
            if d < best_d:
                best_d, best = d, i
        return best

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

        if len(self._points) >= 2:
            poly = QPolygonF([QPointF(*self._to_widget(x, y)) for x, y in self._points])
            painter.setPen(QPen(QColor(0, 255, 0), 2))
            painter.drawPolyline(poly)
            if len(self._points) >= 3:
                painter.drawLine(poly.last(), poly.first())

        painter.setPen(QPen(QColor(0, 0, 0), 1))
        painter.setBrush(QBrush(QColor(255, 0, 0)))
        for x, y in self._points:
            wx, wy = self._to_widget(x, y)
            painter.drawEllipse(QPointF(wx, wy), 5, 5)

    def mousePressEvent(self, event) -> None:
        if self._qimage is None:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            img_x, img_y = self._to_image(event.position())
            idx = self._nearest_point(img_x, img_y)
            if idx is not None:
                self._dragging = idx
            else:
                img_x, img_y = self._clamp(img_x, img_y)
                self._points.append([img_x, img_y])
                self.update()
                self.pointsChanged.emit()
        elif event.button() == Qt.MouseButton.RightButton:
            img_x, img_y = self._to_image(event.position())
            idx = self._nearest_point(img_x, img_y)
            if idx is not None:
                del self._points[idx]
                self.update()
                self.pointsChanged.emit()

    def mouseMoveEvent(self, event) -> None:
        if self._dragging is not None and self._qimage is not None:
            img_x, img_y = self._to_image(event.position())
            img_x, img_y = self._clamp(img_x, img_y)
            self._points[self._dragging] = [img_x, img_y]
            self.update()
            self.pointsChanged.emit()

    def mouseReleaseEvent(self, event) -> None:
        self._dragging = None


class ScanDialog(QDialog):
    def __init__(self, image, parent=None):
        super().__init__(parent)
        self._image = image
        self._result = None
        self.setWindowTitle("Scan Document")
        self.resize(1100, 720)

        self._canvas = ScanCanvas()
        self._canvas.set_image(array_to_qimage(image))
        self._canvas.pointsChanged.connect(self._update_preview)

        self._auto_btn = QPushButton("Auto Detect")
        self._auto_btn.clicked.connect(self._auto_detect)
        self._clear_btn = QPushButton("Clear Points")
        self._clear_btn.clicked.connect(self._clear_points)

        self._illum_slider = QSlider(Qt.Orientation.Horizontal)
        self._illum_slider.setRange(0, 100)
        self._illum_slider.setValue(0)
        self._illum_slider.valueChanged.connect(self._on_slider_changed)
        self._illum_label = QLabel("Illumination removal: 0")

        self._preview_label = QLabel("Place at least 4 corner points")
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_label.setMinimumSize(280, 200)
        self._preview_label.setStyleSheet("background: #333; color: #ccc;")

        self._apply_btn = QPushButton("Apply")
        self._apply_btn.clicked.connect(self._apply)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self.reject)

        controls = QVBoxLayout()
        controls.addWidget(self._auto_btn)
        controls.addWidget(self._clear_btn)
        controls.addSpacing(12)
        controls.addWidget(self._illum_label)
        controls.addWidget(self._illum_slider)
        controls.addStretch(1)
        controls.addWidget(QLabel("Preview:"))
        controls.addWidget(self._preview_label)

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

        self._update_preview()

    def _auto_detect(self) -> None:
        corners = detect_document_corners(self._image)
        if corners is not None:
            self._canvas.set_points(corners.tolist())
        else:
            QMessageBox.information(
                self, "Scan",
                "Could not auto-detect the boundary.\n"
                "Please click the four corners of the document manually.",
            )

    def _clear_points(self) -> None:
        self._canvas.set_points([])

    def _on_slider_changed(self, _value) -> None:
        self._illum_label.setText(
            f"Illumination removal: {self._illum_slider.value()}"
        )
        self._update_preview()

    def _update_preview(self) -> None:
        points = self._canvas.points()
        if len(points) < 4:
            self._result = None
            self._preview_label.setText("Place at least 4 corner points")
            return
        strength = self._illum_slider.value()
        try:
            result = scan_document(self._image, points, strength)
        except Exception as exc:  # noqa: BLE001 - surface in the preview
            self._result = None
            self._preview_label.setText(f"Error: {exc}")
            return
        self._result = result
        small = downsample_image(result, 360)
        self._preview_label.setPixmap(QPixmap.fromImage(array_to_qimage(small)))

    def _apply(self) -> None:
        if self._result is None:
            QMessageBox.warning(
                self, "Scan", "Please place at least 4 corner points first."
            )
            return
        self.accept()

    def result(self):
        return self._result
