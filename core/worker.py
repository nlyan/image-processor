"""Background worker for running operations off the UI thread."""
from PySide6.QtCore import QObject, QRunnable, Signal, Slot


class WorkerSignals(QObject):
    done = Signal(object)
    error = Signal(str)


class FunctionWorker(QRunnable):
    """Run a callable on a thread pool and emit its result or error."""

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        try:
            result = self._fn(*self._args, **self._kwargs)
        except Exception as exc:  # noqa: BLE001 - forwarded to the UI thread
            self.signals.error.emit(str(exc))
        else:
            self.signals.done.emit(result)
