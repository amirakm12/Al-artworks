# UI Components Directory

## What this folder is for
UI components for Qt6 GUI with cosmic theme implementation.

## What has to be kept in this folder
- `.py` files for windows and widgets:
  - `main_window.py` - Main application window
  - `canvas.py` - Image editing canvas
  - `toolbar.py` - Tool selection and controls
  - `dialogs.py` - Custom dialog windows
  - `widgets.py` - Custom Qt widgets
  - `themes.py` - Cosmic gradient themes
  - `eve_avatar.py` - Eve's 3D avatar display
  - `ar_preview.py` - AR preview components
- `__init__.py` for package structure

## What NOT to keep in this folder
- Agents (use `../agents/`)
- Models (use `../models/`)
- Data files (use `../../data/`)
- Business logic (keep UI-specific only)

## Component Overview
- **main_window.py**: QMainWindow with cosmic gradient background
- **canvas.py**: QGraphicsView with zoom/pan for 500x500 image editing
- **toolbar.py**: QToolBar with cropping, filters, and layer controls
- **themes.py**: Dark-to-light cosmic gradients and neon accents
- **eve_avatar.py**: OpenGL widget for Eve's 3D NeRF avatar
- **ar_preview.py**: Holographic AR preview effects