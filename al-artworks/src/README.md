# Source Code Directory

## What this folder is for
Contains all Python source code for the Al-artworks application.

## What has to be kept in this folder
- `.py` files and subfolders for modules:
  - `ui/` - User interface components
  - `agents/` - AI agent implementations
  - `models/` - ML model wrappers
  - `utils/` - Utility functions
- `__init__.py` files for package structure
- `main.py` - Main application logic

## What NOT to keep in this folder
- Assets (use `../assets/`)
- Data files (use `../data/`)
- Documentation (use `../docs/`)
- Tests (use `../tests/`)
- Configuration files (use `../config/`)

## Module Structure
- **ui/**: Qt6-based GUI components with cosmic theme
- **agents/**: 100 specialized agents organized by category
- **models/**: Wrappers for AI/ML models (NeRF, Bark, Whisper, etc.)
- **utils/**: Helper functions for GPU, memory, threading, etc.