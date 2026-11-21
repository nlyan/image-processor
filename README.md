# Image Processor

A simple digital image processing desktop application built with Python,
PySide6 and OpenCV. No machine learning; all algorithms are classic
computer vision.

## Features

- **Open / save**: JPG, PNG, BMP, TIFF, WebP, and **export to PDF**.
- **Adjustments**: brightness, denoising (bilateral / Gaussian / median),
  sharpening (unsharp mask), contrast (linear stretch + gamma + CLAHE),
  color correction (white balance + temperature + saturation).
- **Watermarks**: text watermark (Chinese text supported) and image/logo
  watermark (with transparency).
- **Geometry**: flip, rotate (90°/180°/270°/any angle), resize.
- **Document scanning**: auto-detect the document boundary (or define a
  polygon manually), perspective correction, and uneven-illumination /
  shadow removal while preserving color.
- **Background removal**: GrabCut or color thresholding, with click-to-keep
  region selection and white or transparent output.
- **Workflow**: undo / redo, real-time preview (debounced), before/after
  comparison (original / result / split view / draggable line).
- **UI**: bilingual (English / 中文) and light / dark theme.

## Requirements

- Python 3.13+
- Dependencies in `requirements.txt` (dev extras in `requirements-dev.txt`)

## Installation (from source)

```powershell
git clone image-processor
cd image-processor
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage

```powershell
python main.py
```

## Build a standalone executable

No Python is needed on the target machine — PyInstaller bundles the
interpreter and all dependencies. We use `--onedir` (a folder), which avoids
the temporary-directory extraction issues of `--onefile` and starts faster.

```powershell
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install pyinstaller
.\.venv\Scripts\python.exe -m PyInstaller --onedir --windowed ^
    --name ImageProcessor --icon assets/image.ico main.py
```

The result is the folder `dist\ImageProcessor\`; distribute it as a zip and
run `ImageProcessor.exe` inside it.

### Add a custom icon

1. Create an `.ico` file (e.g. 256x256) and place it at `assets/image.ico`.
2. Pass `--icon assets/image.ico` to PyInstaller as shown above. The icon is
   embedded into the executable **at build time** (before packaging is done).

## License

MIT — see [LICENSE](LICENSE).
