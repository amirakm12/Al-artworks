# Tests Directory

## What this folder is for
Unit and integration tests for the Al-artworks application.

## What has to be kept in this folder
- `.py` test files:
  - `test_ui.py` - UI component tests
  - `test_agents.py` - Agent functionality tests
  - `test_models.py` - Model integration tests
  - `test_workflows.py` - Workflow integration tests
  - `test_performance.py` - Performance benchmarks
  - `test_offline.py` - Offline functionality tests
  - `conftest.py` - Pytest configuration
- `__init__.py` for test discovery
- Test fixtures and mock data

## What NOT to keep in this folder
- Production code (use `../src/`)
- Assets (use `../assets/`)
- Data files (use `../data/`)
- Documentation (use `../docs/`)

## Test Categories
- **Unit Tests**: Individual component testing
- **Integration Tests**: Multi-component workflows
- **Performance Tests**: 60fps rendering, <500ms vectorization
- **Accessibility Tests**: Voice navigation, AR previews
- **Offline Tests**: 50+ queries without internet
- **Coverage Goal**: 95%+ code coverage