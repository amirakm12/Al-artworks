"""
Configuration management for Al-artworks.
Handles application settings and preferences.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


class Config:
    """Application configuration manager."""
    
    DEFAULT_CONFIG = {
        "app": {
            "name": "Al-artworks",
            "version": "1.0.0",
            "theme": "cosmic_dark"
        },
        "ui": {
            "window_width": 1400,
            "window_height": 900,
            "show_splash": True,
            "splash_duration": 3000,
            "enable_animations": True,
            "animation_speed": 1.0
        },
        "canvas": {
            "default_width": 500,
            "default_height": 500,
            "background_color": "#FFFFFF",
            "grid_enabled": False,
            "grid_size": 20,
            "antialiasing": True
        },
        "eve": {
            "enabled": True,
            "auto_activate": True,
            "voice_enabled": True,
            "voice_language": "en",
            "personality": "creative_goddess",
            "greeting_message": "Hello, I am Eve, your celestial creative goddess"
        },
        "performance": {
            "gpu_acceleration": True,
            "max_memory_usage": 8192,  # MB
            "cache_size": 1024,  # MB
            "render_fps": 60,
            "thread_count": -1  # Auto-detect
        },
        "models": {
            "model_path": "data/models",
            "lazy_loading": True,
            "whisper_model": "small.en",
            "bark_voice_pack": "default",
            "nerf_quality": "high",
            "stable_diffusion_steps": 50
        },
        "offline_mode": {
            "enabled": True,
            "allow_online_boost": True,
            "cache_online_results": True
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager."""
        self.config_path = config_path or self._get_default_config_path()
        self.config: Dict[str, Any] = self.DEFAULT_CONFIG.copy()
        
    def _get_default_config_path(self) -> str:
        """Get default configuration file path."""
        # Try to use user config directory
        config_dir = Path.home() / ".config" / "al-artworks"
        config_dir.mkdir(parents=True, exist_ok=True)
        return str(config_dir / "config.json")
        
    def load(self) -> bool:
        """Load configuration from file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    loaded_config = json.load(f)
                    self._merge_config(loaded_config)
                return True
            else:
                # Save default config
                self.save()
                return True
        except Exception as e:
            print(f"Error loading config: {e}")
            return False
            
    def save(self) -> bool:
        """Save configuration to file."""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False
            
    def _merge_config(self, loaded_config: Dict[str, Any]):
        """Merge loaded config with defaults."""
        def merge_dict(default: Dict, loaded: Dict) -> Dict:
            result = default.copy()
            for key, value in loaded.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = merge_dict(result[key], value)
                else:
                    result[key] = value
            return result
            
        self.config = merge_dict(self.DEFAULT_CONFIG, loaded_config)
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-notation key."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
        
    def set(self, key: str, value: Any):
        """Set configuration value by dot-notation key."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
            
        config[keys[-1]] = value
        
    def get_model_path(self, model_name: str) -> Path:
        """Get path to a specific model."""
        base_path = Path(self.get("models.model_path", "data/models"))
        return base_path / model_name
        
    def get_cache_path(self) -> Path:
        """Get cache directory path."""
        return Path("data/cache")
        
    def is_offline_mode(self) -> bool:
        """Check if offline mode is enabled."""
        return self.get("offline_mode.enabled", True)
        
    def get_gpu_enabled(self) -> bool:
        """Check if GPU acceleration is enabled."""
        return self.get("performance.gpu_acceleration", True)
        
    def get_theme(self) -> str:
        """Get current theme name."""
        return self.get("app.theme", "cosmic_dark")
        
    def get_eve_config(self) -> Dict[str, Any]:
        """Get Eve-specific configuration."""
        return self.config.get("eve", {})
        
    def get_canvas_config(self) -> Dict[str, Any]:
        """Get canvas configuration."""
        return self.config.get("canvas", {})
        
    def reset_to_defaults(self):
        """Reset configuration to defaults."""
        self.config = self.DEFAULT_CONFIG.copy()
        self.save()