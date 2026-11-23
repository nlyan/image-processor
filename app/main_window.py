"""Main application window."""
import os

import cv2
from PySide6.QtCore import Qt, QStandardPaths, QThreadPool, QTimer
from PySide6.QtGui import QAction, QActionGroup, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

import i18n
from core.history import History
from core.io import array_to_qimage, load_image, save_image, save_image_pdf
from core.preview import downsample_image
from core.registry import build_default_registry
from core.worker import FunctionWorker
from .background_dialog import BackgroundDialog
from .canvas import ImageCanvas
from .scan_dialog import ScanDialog
from .widgets.param_panel import ParamPanel

IMAGE_FILTER = (
    "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;All Files (*)"
)
PDF_FILTER = "PDF (*.pdf)"

DARK_QSS = """
QMainWindow, QDialog { background: #2b2b2b; color: #e0e0e0; }
QMenuBar, QMenu { background: #3c3c3c; color: #e0e0e0; }
QMenu::item:selected { background: #505050; }
QToolBar { background: #3c3c3c; border: none; }
QStatusBar { background: #3c3c3c; color: #e0e0e0; }
QDockWidget { color: #e0e0e0; }
QDockWidget::title { background: #3c3c3c; }
QLabel, QCheckBox, QGroupBox { color: #e0e0e0; }
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
    background: #3c3c3c; color: #e0e0e0; border: 1px solid #555;
}
QPushButton { background: #4a4a4a; color: #e0e0e0; }
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Processor")
        self.resize(1200, 760)

        self._image = None          # committed RGB uint8 image
        self._original_image = None # loaded image, never modified
        self._preview_image = None  # pending preview result (or None)
        self._path = None
        self._registry = build_default_registry()
        self._history = History()
        self._thread_pool = QThreadPool(self)
        self._current_op = None
        self._apply_worker = None
        self._busy = False
        self._suppress_preview = False
        self._compare_mode = "result"
        self._compare_fraction = 0.5
        self._theme = "light"

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(500)
        self._preview_timer.timeout.connect(self._refresh_preview)

        self._canvas = ImageCanvas(self)
        self.setCentralWidget(self._canvas)
        self._canvas.zoomChanged.connect(self._on_zoom_changed)
        self._canvas.compareLineChanged.connect(self._on_compare_line)

        self._status_image = QLabel(tr("no_image"))
        self._status_zoom = QLabel("")
        self._status_busy = QLabel("")
        self.statusBar().addWidget(self._status_image)
        self.statusBar().addWidget(self._status_busy)
        self.statusBar().addPermanentWidget(self._status_zoom)

        self._build_actions()
        self._build_menus()
        self._build_toolbar()
        self._build_adjust_dock()

        self._update_status()
        self._update_history_actions()
        self._update_adjust_enabled()

    # ---- UI construction -------------------------------------------------

    def _build_actions(self) -> None:
        self.open_action = QAction(tr("open"), self)
        self.open_action.setShortcut(QKeySequence.StandardKey.Open)
        self.open_action.triggered.connect(self.open_image)

        self.save_as_action = QAction(tr("save_as"), self)
        self.save_as_action.setShortcut(QKeySequence.StandardKey.SaveAs)
        self.save_as_action.triggered.connect(self.save_as)

        self.save_pdf_action = QAction(tr("save_pdf"), self)
        self.save_pdf_action.triggered.connect(self.save_as_pdf)

        self.scan_action = QAction(tr("scan"), self)
        self.scan_action.triggered.connect(self.scan_document)

        self.remove_bg_action = QAction(tr("remove_bg"), self)
        self.remove_bg_action.triggered.connect(self.remove_background)

        self.exit_action = QAction(tr("exit"), self)
        self.exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        self.exit_action.triggered.connect(self.close)

        self.undo_action = QAction(tr("undo"), self)
        self.undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        self.undo_action.triggered.connect(self.undo)

        self.redo_action = QAction(tr("redo"), self)
        self.redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        self.redo_action.triggered.connect(self.redo)

        self.fit_action = QAction(tr("fit"), self)
        self.fit_action.setShortcut("Ctrl+0")
        self.fit_action.triggered.connect(self._canvas.fit_to_window)

        self.actual_action = QAction(tr("actual"), self)
        self.actual_action.setShortcut("Ctrl+1")
        self.actual_action.triggered.connect(self._canvas.actual_size)

        # Language
        self.lang_en_action = QAction(tr("lang_en"), self)
        self.lang_en_action.setCheckable(True)
        self.lang_en_action.triggered.connect(lambda: self._on_language_changed("en"))
        self.lang_zh_action = QAction(tr("lang_zh"), self)
        self.lang_zh_action.setCheckable(True)
        self.lang_zh_action.triggered.connect(lambda: self._on_language_changed("zh"))
        self.lang_en_action.setChecked(i18n.LANG == "en")
        self.lang_zh_action.setChecked(i18n.LANG == "zh")

        # Compare modes
        self._compare_group = QActionGroup(self)
        self._compare_group.setExclusive(True)
        self._compare_group.triggered.connect(self._on_compare_changed)
        self.cmp_original_action = QAction(tr("cmp_original"), self)
        self.cmp_result_action = QAction(tr("cmp_result"), self)
        self.cmp_split_action = QAction(tr("cmp_split"), self)
        self.cmp_drag_action = QAction(tr("cmp_drag"), self)
        for action, data in (
            (self.cmp_original_action, "original"),
            (self.cmp_result_action, "result"),
            (self.cmp_split_action, "split"),
            (self.cmp_drag_action, "drag"),
        ):
            action.setCheckable(True)
            action.setData(data)
            self._compare_group.addAction(action)
        self.cmp_result_action.setChecked(True)

        # Theme
        self.theme_light_action = QAction(tr("theme_light"), self)
        self.theme_light_action.setCheckable(True)
        self.theme_light_action.triggered.connect(lambda: self._on_theme_changed("light"))
        self.theme_dark_action = QAction(tr("theme_dark"), self)
        self.theme_dark_action.setCheckable(True)
        self.theme_dark_action.triggered.connect(lambda: self._on_theme_changed("dark"))
        self.theme_light_action.setChecked(True)

    def _build_menus(self) -> None:
        self._file_menu = self.menuBar().addMenu(tr("file"))
        self._file_menu.addAction(self.open_action)
        self._file_menu.addAction(self.save_as_action)
        self._file_menu.addAction(self.save_pdf_action)
        self._file_menu.addAction(self.scan_action)
        self._file_menu.addAction(self.remove_bg_action)
        self._file_menu.addSeparator()
        self._file_menu.addAction(self.exit_action)

        self._edit_menu = self.menuBar().addMenu(tr("edit"))
        self._edit_menu.addAction(self.undo_action)
        self._edit_menu.addAction(self.redo_action)

        self._view_menu = self.menuBar().addMenu(tr("view"))
        self._view_menu.addAction(self.fit_action)
        self._view_menu.addAction(self.actual_action)

        self._compare_menu = self.menuBar().addMenu(tr("compare"))
        self._compare_menu.addAction(self.cmp_original_action)
        self._compare_menu.addAction(self.cmp_result_action)
        self._compare_menu.addAction(self.cmp_split_action)
        self._compare_menu.addAction(self.cmp_drag_action)

        self._lang_menu = self.menuBar().addMenu(tr("language"))
        self._lang_menu.addAction(self.lang_en_action)
        self._lang_menu.addAction(self.lang_zh_action)

        self._theme_menu = self.menuBar().addMenu(tr("theme"))
        self._theme_menu.addAction(self.theme_light_action)
        self._theme_menu.addAction(self.theme_dark_action)

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Main Toolbar", self)
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        toolbar.addAction(self.open_action)
        toolbar.addAction(self.save_as_action)
        toolbar.addSeparator()
        toolbar.addAction(self.undo_action)
        toolbar.addAction(self.redo_action)
        toolbar.addSeparator()
        toolbar.addAction(self.fit_action)
        toolbar.addAction(self.actual_action)

    def _build_adjust_dock(self) -> None:
        self._op_combo = QComboBox()
        for op in self._registry.all():
            self._op_combo.addItem(
                op.title if i18n.LANG == "en" else op.title_zh, op
            )
        self._op_combo.currentIndexChanged.connect(self._on_operation_changed)

        self._param_panel = ParamPanel()
        self._param_panel.valuesChanged.connect(self._on_params_changed)

        self._apply_button = QPushButton(tr("apply"))
        self._apply_button.clicked.connect(self.apply_operation)
        self._reset_button = QPushButton(tr("reset"))
        self._reset_button.clicked.connect(self.reset_params)

        buttons = QHBoxLayout()
        buttons.addWidget(self._apply_button)
        buttons.addWidget(self._reset_button)

        self._operation_label = QLabel(tr("operation"))

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self._operation_label)
        layout.addWidget(self._op_combo)
        layout.addSpacing(8)
        layout.addWidget(self._param_panel)
        layout.addStretch(1)
        layout.addLayout(buttons)

        self._adjust_dock = QDockWidget(tr("adjust"), self)
        self._adjust_dock.setWidget(container)
        self._adjust_dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._adjust_dock)

        if self._op_combo.count():
            self._op_combo.setCurrentIndex(0)
            self._on_operation_changed(0)

    # ---- File actions ----------------------------------------------------

    def open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", IMAGE_FILTER)
        if not path:
            return
        try:
            image = load_image(path)
        except Exception as exc:  # noqa: BLE001 - surface any load failure
            QMessageBox.critical(self, "Error", f"Failed to open image:\n{exc}")
            return
        self._image = image
        self._original_image = image
        self._preview_image = None
        self._path = path
        self._history.clear()
        self.setWindowTitle(f"Image Processor - {os.path.basename(path)}")
        self._show_committed()
        self._update_status()
        self._update_history_actions()
        self._update_adjust_enabled()

    def _default_dir(self) -> str:
        return self._path or QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.PicturesLocation
        )

    def save_as(self) -> None:
        if self._image is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", self._default_dir(), IMAGE_FILTER
        )
        if not path:
            return
        try:
            path = save_image(path, self._image)
        except Exception as exc:  # noqa: BLE001 - surface any save failure
            QMessageBox.critical(
                self, "Error",
                f"Failed to save image:\n{exc}\n\n"
                "Tip: choose a writable folder (e.g. Desktop or Documents).",
            )
            return
        self._path = path
        self.setWindowTitle(f"Image Processor - {os.path.basename(path)}")

    def save_as_pdf(self) -> None:
        if self._image is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save as PDF", self._default_dir(), PDF_FILTER
        )
        if not path:
            return
        try:
            save_image_pdf(path, self._image)
        except Exception as exc:  # noqa: BLE001 - surface any save failure
            QMessageBox.critical(
                self, "Error",
                f"Failed to save PDF:\n{exc}\n\n"
                "Tip: choose a writable folder (e.g. Desktop or Documents).",
            )
            return

    def scan_document(self) -> None:
        if self._image is None:
            return
        dialog = ScanDialog(self._image, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            result = dialog.result()
            if result is not None:
                self._history.record(self._image)
                self._image = result
                self._preview_image = None
                self._show_committed()
                self._update_status()
                self._update_history_actions()

    def remove_background(self) -> None:
        if self._image is None:
            return
        dialog = BackgroundDialog(self._to_rgb(self._image), self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            result = dialog.result()
            if result is not None:
                self._history.record(self._image)
                self._image = result
                self._preview_image = None
                self._show_committed()
                self._update_status()
                self._update_history_actions()

    # ---- Edit actions ----------------------------------------------------

    def undo(self) -> None:
        if self._image is None or not self._history.can_undo:
            return
        self._image = self._history.undo(self._image)
        self._show_committed()
        self._update_status()
        self._update_history_actions()

    def redo(self) -> None:
        if self._image is None or not self._history.can_redo:
            return
        self._image = self._history.redo(self._image)
        self._show_committed()
        self._update_status()
        self._update_history_actions()

    # ---- Operation flow --------------------------------------------------

    def _on_operation_changed(self, index: int) -> None:
        op = self._op_combo.itemData(index)
        self._current_op = op
        self._param_panel.set_operation(op)
        self._refresh_preview()

    def _on_params_changed(self, _params: dict) -> None:
        if not self._suppress_preview:
            self._preview_timer.start()

    def _refresh_preview(self) -> None:
        if self._image is None or self._current_op is None or self._busy:
            return
        small = downsample_image(self._to_rgb(self._image))
        self._preview_image = self._current_op.process(
            small, self._param_panel.values()
        )
        self._update_display()

    def reset_params(self) -> None:
        self._show_committed()

    def apply_operation(self) -> None:
        if self._image is None or self._current_op is None or self._busy:
            return
        params = self._param_panel.values()
        if params == self._current_op.default_params():
            return
        op = self._current_op
        base = self._to_rgb(self._image)

        self._busy = True
        self._status_busy.setText("Processing...")
        self._apply_button.setEnabled(False)

        worker = FunctionWorker(op.process, base, params)
        worker.signals.done.connect(self._on_apply_done)
        worker.signals.error.connect(self._on_apply_error)
        self._apply_worker = worker
        self._thread_pool.start(worker)

    def _on_apply_done(self, result) -> None:
        self._history.record(self._image)
        self._image = result
        self._finish_apply()
        self._update_status()
        self._update_history_actions()

    def _on_apply_error(self, message: str) -> None:
        QMessageBox.critical(self, "Error", f"Operation failed:\n{message}")
        self._finish_apply()

    def _finish_apply(self) -> None:
        self._busy = False
        self._apply_worker = None
        self._apply_button.setEnabled(True)
        self._status_busy.setText("")
        self._show_committed()

    def _show_committed(self) -> None:
        self._preview_timer.stop()
        self._suppress_preview = True
        self._param_panel.reset()
        self._suppress_preview = False
        self._preview_image = None
        self._update_display(fit=True)

    @staticmethod
    def _to_rgb(image):
        if image.ndim == 3 and image.shape[2] == 4:
            return image[..., :3]
        return image

    # ---- Display / comparison -------------------------------------------

    def _current_display(self):
        if self._preview_image is not None:
            return self._preview_image
        return self._image

    def _comparison_display(self):
        after = self._current_display()
        if after is None:
            return None
        if self._original_image is None:
            return after
        mode = self._compare_mode
        if mode == "original":
            return self._original_image
        if mode == "result":
            return after

        max_side = 1600
        orig = downsample_image(self._original_image, max_side)
        curr = downsample_image(after, max_side)
        height, width = curr.shape[:2]
        if orig.shape[:2] != (height, width):
            orig = cv2.resize(orig, (width, height), interpolation=cv2.INTER_AREA)
        x = width // 2 if mode == "split" else int(self._compare_fraction * (width - 1))
        result = orig.copy()
        result[:, x:] = curr[:, x:]
        return result

    def _update_display(self, fit: bool = False) -> None:
        display = self._comparison_display()
        if display is None:
            return
        pixmap = QPixmap.fromImage(array_to_qimage(display))
        if fit:
            self._canvas.set_image(pixmap)
        else:
            self._canvas.update_pixmap(pixmap)

    def _on_compare_changed(self, action) -> None:
        self._compare_mode = action.data()
        self._canvas.set_compare_drag(self._compare_mode == "drag")
        self._update_display()

    def _on_compare_line(self, fraction: float) -> None:
        self._compare_fraction = fraction
        self._update_display()

    # ---- Language / theme ------------------------------------------------

    def _on_language_changed(self, lang: str) -> None:
        i18n.set_language(lang)
        self.lang_en_action.setChecked(lang == "en")
        self.lang_zh_action.setChecked(lang == "zh")
        self._retranslate()

    def _retranslate(self) -> None:
        self._file_menu.setTitle(tr("file"))
        self._edit_menu.setTitle(tr("edit"))
        self._view_menu.setTitle(tr("view"))
        self._compare_menu.setTitle(tr("compare"))
        self._lang_menu.setTitle(tr("language"))
        self._theme_menu.setTitle(tr("theme"))

        self.open_action.setText(tr("open"))
        self.save_as_action.setText(tr("save_as"))
        self.save_pdf_action.setText(tr("save_pdf"))
        self.scan_action.setText(tr("scan"))
        self.remove_bg_action.setText(tr("remove_bg"))
        self.exit_action.setText(tr("exit"))
        self.undo_action.setText(tr("undo"))
        self.redo_action.setText(tr("redo"))
        self.fit_action.setText(tr("fit"))
        self.actual_action.setText(tr("actual"))
        self.lang_en_action.setText(tr("lang_en"))
        self.lang_zh_action.setText(tr("lang_zh"))
        self.cmp_original_action.setText(tr("cmp_original"))
        self.cmp_result_action.setText(tr("cmp_result"))
        self.cmp_split_action.setText(tr("cmp_split"))
        self.cmp_drag_action.setText(tr("cmp_drag"))
        self.theme_light_action.setText(tr("theme_light"))
        self.theme_dark_action.setText(tr("theme_dark"))

        self._adjust_dock.setWindowTitle(tr("adjust"))
        self._operation_label.setText(tr("operation"))
        self._apply_button.setText(tr("apply"))
        self._reset_button.setText(tr("reset"))

        current = self._current_op
        self._op_combo.blockSignals(True)
        self._op_combo.clear()
        for op in self._registry.all():
            self._op_combo.addItem(
                op.title if i18n.LANG == "en" else op.title_zh, op
            )
        if current is not None:
            index = self._op_combo.findData(current)
            if index >= 0:
                self._op_combo.setCurrentIndex(index)
        self._op_combo.blockSignals(False)

        self._param_panel.retranslate()
        self._update_status()

    def _on_theme_changed(self, theme: str) -> None:
        self._theme = theme
        self.theme_light_action.setChecked(theme == "light")
        self.theme_dark_action.setChecked(theme == "dark")
        self.setStyleSheet(DARK_QSS if theme == "dark" else "")

    # ---- Status / enabled state -----------------------------------------

    def _update_status(self) -> None:
        if self._image is None:
            self._status_image.setText(tr("no_image"))
        else:
            height, width = self._image.shape[:2]
            self._status_image.setText(f"{width} x {height} px")
        self._on_zoom_changed(self._canvas.scale_factor)

    def _on_zoom_changed(self, scale: float) -> None:
        self._status_zoom.setText(f"{scale * 100:.0f}%")

    def _update_history_actions(self) -> None:
        self.undo_action.setEnabled(self._history.can_undo)
        self.redo_action.setEnabled(self._history.can_redo)

    def _update_adjust_enabled(self) -> None:
        enabled = self._image is not None
        self._op_combo.setEnabled(enabled)
        self._param_panel.setEnabled(enabled)
        self._apply_button.setEnabled(enabled)
        self._reset_button.setEnabled(enabled)


def tr(key: str) -> str:
    return i18n.tr(key)
