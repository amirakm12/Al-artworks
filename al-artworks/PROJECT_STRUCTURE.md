# Al-artworks Project Structure

```
al-artworks/
│
├── main.py                    # Main entry point - launches the application
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup configuration
├── pyproject.toml            # Modern Python project configuration
├── README.md                 # Project documentation
├── .gitignore                # Git ignore rules
├── LICENSE                   # Project license
├── PROJECT_STRUCTURE.md      # This file
│
├── src/                      # Source code directory
│   ├── __init__.py
│   │
│   ├── ui/                   # User Interface components
│   │   ├── __init__.py
│   │   ├── main_window.py    # Main application window with cosmic theme
│   │   ├── canvas.py         # Image canvas with zoom/pan capabilities
│   │   ├── toolbar.py        # Tools toolbar (crop, filters, layers)
│   │   ├── dialogs.py        # Custom dialog windows
│   │   ├── widgets.py        # Custom Qt widgets
│   │   ├── themes.py         # Cosmic gradient themes
│   │   ├── eve_avatar.py     # Eve's 3D avatar display
│   │   └── ar_preview.py     # AR preview components
│   │
│   ├── agents/               # 100 Specialized AI Agents
│   │   ├── __init__.py
│   │   ├── orchestrator.py   # Multi-agent orchestration system
│   │   ├── base_agent.py     # Base agent class
│   │   │
│   │   ├── vector_agents.py  # 20 Vector Processing Agents
│   │   │   # - VectorConversionAgent
│   │   │   # - PathOptimizationAgent
│   │   │   # - BezierCurveAgent
│   │   │   # - SVGExportAgent
│   │   │   # - VectorCleanupAgent
│   │   │   # ... (15 more)
│   │   │
│   │   ├── image_agents.py   # 30 Image Processing Agents
│   │   │   # - DenoisingAgent
│   │   │   # - EnhancementAgent
│   │   │   # - FilterAgent
│   │   │   # - CropToolAgent
│   │   │   # - LayerManagementAgent
│   │   │   # ... (25 more)
│   │   │
│   │   ├── ui_agents.py      # 20 UI Management Agents
│   │   │   # - ThemeAgent
│   │   │   # - LayoutAgent
│   │   │   # - AnimationAgent
│   │   │   # - GestureRecognitionAgent
│   │   │   # ... (16 more)
│   │   │
│   │   ├── audio_agents.py   # 10 Audio Processing Agents
│   │   │   # - WhisperVoiceAgent
│   │   │   # - BarkVoiceAgent
│   │   │   # - NoiseReductionAgent
│   │   │   # - EmotionDetectionAgent
│   │   │   # ... (6 more)
│   │   │
│   │   ├── data_agents.py    # 10 Data Management Agents
│   │   │   # - CacheManagementAgent
│   │   │   # - DatabaseAgent
│   │   │   # - FileIOAgent
│   │   │   # - CompressionAgent
│   │   │   # ... (6 more)
│   │   │
│   │   └── misc_agents.py    # 10 Miscellaneous Agents
│   │       # - LLMMetaAgent
│   │       # - PredictiveIntentAgent
│   │       # - FeedbackLoopAgent
│   │       # - QualityCheckAgent
│   │       # ... (6 more)
│   │
│   ├── models/               # AI/ML Model Wrappers
│   │   ├── __init__.py
│   │   ├── nerf_model.py     # NeRF for Eve's 3D avatar
│   │   ├── bark_tts.py       # Bark text-to-speech
│   │   ├── whisper_asr.py    # Whisper speech recognition
│   │   ├── llama_agent.py    # LLaMA language model
│   │   ├── stable_diffusion.py # Stable Diffusion integration
│   │   ├── clip_model.py     # CLIP model wrapper
│   │   ├── esrgan_model.py   # Real-ESRGAN upscaling
│   │   └── model_loader.py   # Efficient model loading utilities
│   │
│   └── utils/                # Utility modules
│       ├── __init__.py
│       ├── config.py         # Configuration management
│       ├── logger.py         # Logging utilities
│       ├── gpu_utils.py      # GPU optimization utilities
│       ├── memory_manager.py # Memory management for 64GB RAM
│       ├── threading_utils.py # Python 3.13 no-GIL utilities
│       ├── voice_utils.py    # Voice processing utilities
│       └── image_utils.py    # Image processing helpers
│
├── data/                     # Data directory (10GB bundled content)
│   ├── models/               # Pre-trained AI models
│   │   ├── whisper/          # Whisper ASR models (500MB)
│   │   ├── bark/             # Bark TTS models (3GB)
│   │   ├── nerf/             # NeRF models (1GB)
│   │   ├── llama/            # Distilled LLaMA (4GB)
│   │   ├── stable_diffusion/ # SD models (2GB)
│   │   ├── clip/             # CLIP models (500MB)
│   │   └── checkpoints/      # Model checkpoints
│   │
│   ├── assets/               # Static assets
│   │   ├── templates/        # Image templates
│   │   ├── filters/          # Filter presets
│   │   ├── styles/           # Style presets
│   │   └── examples/         # Example images
│   │
│   ├── cache/                # Runtime cache
│   │   ├── thumbnails/       # Image thumbnails
│   │   ├── temp/             # Temporary files
│   │   └── sessions/         # Session data
│   │
│   └── preferences.db        # SQLite database for user preferences
│
├── assets/                   # Application assets
│   ├── icons/                # UI icons
│   │   ├── tools/            # Tool icons
│   │   ├── actions/          # Action icons
│   │   └── eve/              # Eve avatar assets
│   │
│   ├── themes/               # UI themes
│   │   ├── cosmic_dark.qss   # Dark cosmic theme
│   │   ├── cosmic_light.qss  # Light cosmic theme
│   │   └── gradients/        # Gradient definitions
│   │
│   └── sounds/               # Audio assets
│       ├── eve_voices/       # Eve's voice samples
│       ├── ui_sounds/        # UI interaction sounds
│       └── notifications/    # Notification sounds
│
├── config/                   # Configuration files
│   ├── default_config.yaml   # Default settings
│   ├── agent_config.yaml     # Agent configurations
│   ├── model_config.yaml     # Model parameters
│   └── ui_config.yaml        # UI settings
│
├── tests/                    # Test suite
│   ├── __init__.py
│   ├── conftest.py           # Pytest configuration
│   ├── test_ui.py            # UI component tests
│   ├── test_agents.py        # Agent functionality tests
│   ├── test_models.py        # Model integration tests
│   ├── test_workflows.py     # Workflow integration tests
│   ├── test_performance.py   # Performance benchmarks
│   └── test_offline.py       # Offline functionality tests
│
├── docs/                     # Documentation
│   ├── user_guide.md         # User documentation
│   ├── developer_guide.md    # Developer documentation
│   ├── agent_list.md         # Complete list of 100 agents
│   ├── api_reference.md      # API documentation
│   ├── troubleshooting.md    # Common issues and solutions
│   └── architecture.md       # System architecture
│
├── scripts/                  # Utility scripts
│   ├── download_models.py    # Download pre-trained models
│   ├── optimize_models.py    # Model optimization script
│   ├── build_installer.py    # Build standalone installer
│   └── run_tests.sh          # Test runner script
│
└── build/                    # Build artifacts (generated)
    ├── dist/                 # Distribution packages
    ├── temp/                 # Temporary build files
    └── installers/           # Platform-specific installers
```

## Key Features of the Structure:

1. **Modular Design**: Each component is separated into its own module for maintainability
2. **100 Agents**: Organized by category (vector, image, UI, audio, data, misc)
3. **Offline-First**: All models and assets bundled in the `data/` directory
4. **Eve Integration**: Dedicated components for Eve's avatar, voice, and AI
5. **Cosmic Theme**: Theme files and gradients for the celestial UI
6. **Performance Optimized**: Utilities for GPU/memory management
7. **Comprehensive Testing**: Full test suite for all components
8. **Documentation**: Complete guides for users and developers