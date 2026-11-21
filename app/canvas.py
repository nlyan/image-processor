"""Zoomable / pannable image canvas with an optional draggable compare line."""
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPainter, QPixmap, QWheelEvent
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView


class ImageCanvas(QGraphicsView):
    """Displays a single image with wheel zoom and drag panning."""

    zoomChanged = Signal(float)
    compareLineChanged = Signal(float)  # fraction in [0, 1]

    MIN_SCALE = 0.02
    MAX_SCALE = 50.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._item = QGraphicsPixmapItem()
        self._scene.addItem(self._item)

        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setBackgroundBrush(Qt.GlobalColor.darkGray)
        self._scale = 1.0
        self._compare_drag = False

    @property
    def scale_factor(self) -> float:
        return self._scale

    def has_image(self) -> bool:
        return not self._item.pixmap().isNull()

    def set_compare_drag(self, enabled: bool) -> None:
        self._compare_drag = enabled
        self.setDragMode(
            QGraphicsView.DragMode.NoDrag if enabled
            else QGraphicsView.DragMode.ScrollHandDrag
        )

    def set_image(self, pixmap: QPixmap) -> None:
        self._item.setPixmap(pixmap)
        self._scene.setSceneRect(self._item.boundingRect())
        self.resetTransform()
        self.fit_to_window()

    def update_pixmap(self, pixmap: QPixmap) -> None:
        """Replace the pixmap without resetting the current zoom/pan."""
        self._item.setPixmap(pixmap)
        self._scene.setSceneRect(self._item.boundingRect())

    def fit_to_window(self) -> None:
        if not self.has_image():
            return
        self.fitInView(self._item, Qt.AspectRatioMode.KeepAspectRatio)
        self._update_scale()

    def actual_size(self) -> None:
        if not self.has_image():
            return
        self.resetTransform()
        self._update_scale()

    def _update_scale(self) -> None:
        self._scale = self.transform().m11()
        self.zoomChanged.emit(self._scale)

    def _zoom(self, factor: float) -> None:
        if not self.has_image():
            return
        target = self.transform().m11() * factor
        if target < self.MIN_SCALE or target > self.MAX_SCALE:
            return
        self.scale(factor, factor)
        self._update_scale()

    def wheelEvent(self, event: QWheelEvent) -> None:
        if not self.has_image():
            event.ignore()
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.15 if delta > 0 else 1.0 / 1.15
        self._zoom(factor)
        event.accept()

    def _emit_compare_fraction(self, pos) -> None:
        scene_pos = self.mapToScene(pos.toPoint())
        rect = self._item.boundingRect()
        if rect.width() > 1:
            fraction = (scene_pos.x() - rect.left()) / rect.width()
            self.compareLineChanged.emit(max(0.0, min(1.0, fraction)))

    def mousePressEvent(self, event) -> None:
        if self._compare_drag and event.button() == Qt.MouseButton.LeftButton:
            self._emit_compare_fraction(event.position())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if (
            self._compare_drag
            and event.buttons() & Qt.MouseButton.LeftButton
        ):
            self._emit_compare_fraction(event.position())
            event.accept()
            return
        super().mouseMoveEvent(event)
