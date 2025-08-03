# Assets Directory

## What this folder is for
Static resources like images, icons, voice packs, and theme files.

## What has to be kept in this folder
- Subfolders for different asset types:
  - `icons/` - UI icons (tools, actions, eve)
  - `themes/` - Qt stylesheets and gradient definitions
  - `sounds/` - Audio assets (Eve voices, UI sounds, notifications)
- Static resources that don't change during runtime
- Pre-processed assets ready for use

## What NOT to keep in this folder
- Code files (use `../src/`)
- Data archives (use `../data/`)
- Documentation (use `../docs/`)
- Temporary or generated files

## Asset Categories

### Icons
- `tools/` - Tool icons for the toolbar
- `actions/` - Action icons for menus and buttons
- `eve/` - Eve avatar-related icons and images

### Themes
- `cosmic_dark.qss` - Dark cosmic theme stylesheet
- `cosmic_light.qss` - Light cosmic theme stylesheet
- `gradients/` - Gradient definition files

### Sounds
- `eve_voices/` - Eve's voice samples (3GB)
- `ui_sounds/` - Interface interaction sounds
- `notifications/` - Alert and notification sounds