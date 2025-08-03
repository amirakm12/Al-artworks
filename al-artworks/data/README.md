# Data Directory

## What this folder is for
Bundled offline data and model weights (10GB total).

## What has to be kept in this folder
- Databases and archives:
  - `preferences.db` - SQLite database for user preferences
  - `models/` - Pre-trained AI model weights
  - `assets/` - Static data assets (templates, filters, styles)
  - `cache/` - Runtime cache data
- Large data files for offline functionality
- Pre-processed datasets

## What NOT to keep in this folder
- Code files (use `../src/`)
- UI assets (use `../assets/`)
- Tests (use `../tests/`)
- Documentation (use `../docs/`)

## Data Structure

### Models (10GB total)
- `whisper/` - Whisper ASR models (500MB)
- `bark/` - Bark TTS models (3GB)
- `nerf/` - NeRF models for Eve (1GB)
- `llama/` - Distilled LLaMA (4GB)
- `stable_diffusion/` - SD models (2GB)
- `clip/` - CLIP models (500MB)
- `checkpoints/` - Model checkpoints

### Assets
- `templates/` - Image templates
- `filters/` - Filter presets
- `styles/` - Style presets
- `examples/` - Example images

### Cache
- `thumbnails/` - Generated image thumbnails
- `temp/` - Temporary processing files
- `sessions/` - User session data