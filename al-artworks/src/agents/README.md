# Agents Directory

## What this folder is for
Definitions and orchestration for 100 specialized AI agents.

## What has to be kept in this folder
- `.py` files for agent categories:
  - `vector_agents.py` - 20 vector processing agents
  - `image_agents.py` - 30 image processing agents  
  - `ui_agents.py` - 20 UI management agents
  - `audio_agents.py` - 10 audio handling agents
  - `data_agents.py` - 10 data management agents
  - `misc_agents.py` - 10 miscellaneous agents
  - `orchestrator.py` - Multi-agent orchestration system
  - `base_agent.py` - Base agent class
- `__init__.py` for package structure

## What NOT to keep in this folder
- UI components (use `../ui/`)
- ML models (use `../models/`)
- Assets (use `../../assets/`)
- Raw data (use `../../data/`)

## Agent Categories

### Vector Agents (20)
- VectorConversionAgent, PathOptimizationAgent, BezierCurveAgent, SVGExportAgent, etc.

### Image Processing Agents (30)
- DenoisingAgent, EnhancementAgent, FilterAgent, CropToolAgent, LayerManagementAgent, etc.

### UI Management Agents (20)
- ThemeAgent, LayoutAgent, AnimationAgent, GestureRecognitionAgent, etc.

### Audio Agents (10)
- WhisperVoiceAgent, BarkVoiceAgent, NoiseReductionAgent, EmotionDetectionAgent, etc.

### Data Management Agents (10)
- CacheManagementAgent, DatabaseAgent, FileIOAgent, CompressionAgent, etc.

### Miscellaneous Agents (10)
- LLMMetaAgent, PredictiveIntentAgent, FeedbackLoopAgent, QualityCheckAgent, etc.