"""
Config Manager - Persistent Configuration System
Handles global app settings and plugin-specific configurations
"""

import json
import threading
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

class ConfigManager:
    """Singleton config manager for persistent settings"""
    
    _lock = threading.Lock()
    _instance = None
    
    def __new__(cls, path="config.json"):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._path = Path(path)
            cls._instance._data = {}
            cls._instance._logger = logging.getLogger(__name__)
            cls._instance.load()
        return cls._instance
    
    def load(self):
        """Load configuration from file"""
        with self._lock:
            if self._path.exists():
                try:
                    with open(self._path, "r", encoding="utf-8") as f:
                        self._data = json.load(f)
                    self._logger.info(f"✅ Config loaded from {self._path}")
                except Exception as e:
                    self._logger.error(f"❌ Failed to load config: {e}")
                    self._data = {}
            else:
                self._data = {}
                self._logger.info("📝 No config file found, using defaults")
    
    def save(self):
        """Save configuration to file"""
        with self._lock:
            try:
                # Ensure directory exists
                self._path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(self._path, "w", encoding="utf-8") as f:
                    json.dump(self._data, f, indent=4, ensure_ascii=False)
                self._logger.info(f"💾 Config saved to {self._path}")
            except Exception as e:
                self._logger.error(f"❌ Failed to save config: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value"""
        with self._lock:
            return self._data.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set a configuration value"""
        with self._lock:
            self._data[key] = value
        self.save()
    
    def delete(self, key: str) -> bool:
        """Delete a configuration key"""
        with self._lock:
            if key in self._data:
                del self._data[key]
                self.save()
                return True
            return False
    
    def get_plugin_config(self, plugin_name: str) -> Dict[str, Any]:
        """Get plugin-specific configuration"""
        with self._lock:
            plugins = self._data.setdefault("plugins", {})
            return plugins.setdefault(plugin_name, {})
    
    def set_plugin_config(self, plugin_name: str, config_dict: Dict[str, Any]):
        """Set plugin-specific configuration"""
        with self._lock:
            plugins = self._data.setdefault("plugins", {})
            plugins[plugin_name] = config_dict
        self.save()
    
    def get_plugin_value(self, plugin_name: str, key: str, default: Any = None) -> Any:
        """Get a specific value from plugin config"""
        plugin_config = self.get_plugin_config(plugin_name)
        return plugin_config.get(key, default)
    
    def set_plugin_value(self, plugin_name: str, key: str, value: Any):
        """Set a specific value in plugin config"""
        plugin_config = self.get_plugin_config(plugin_name)
        plugin_config[key] = value
        self.set_plugin_config(plugin_name, plugin_config)
    
    def get_app_settings(self) -> Dict[str, Any]:
        """Get application settings"""
        return self.get("app_settings", {})
    
    def set_app_settings(self, settings: Dict[str, Any]):
        """Set application settings"""
        self.set("app_settings", settings)
    
    def get_app_setting(self, key: str, default: Any = None) -> Any:
        """Get a specific app setting"""
        app_settings = self.get_app_settings()
        return app_settings.get(key, default)
    
    def set_app_setting(self, key: str, value: Any):
        """Set a specific app setting"""
        app_settings = self.get_app_settings()
        app_settings[key] = value
        self.set_app_settings(app_settings)
    
    def get_ui_settings(self) -> Dict[str, Any]:
        """Get UI-specific settings"""
        return self.get("ui_settings", {})
    
    def set_ui_settings(self, settings: Dict[str, Any]):
        """Set UI-specific settings"""
        self.set("ui_settings", settings)
    
    def get_voice_settings(self) -> Dict[str, Any]:
        """Get voice-related settings"""
        return self.get("voice_settings", {})
    
    def set_voice_settings(self, settings: Dict[str, Any]):
        """Set voice-related settings"""
        self.set("voice_settings", settings)
    
    def get_llm_settings(self) -> Dict[str, Any]:
        """Get LLM-related settings"""
        return self.get("llm_settings", {})
    
    def set_llm_settings(self, settings: Dict[str, Any]):
        """Set LLM-related settings"""
        self.set("llm_settings", settings)
    
    def export_config(self, export_path: str):
        """Export configuration to a file"""
        try:
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=4, ensure_ascii=False)
            self._logger.info(f"📤 Config exported to {export_path}")
        except Exception as e:
            self._logger.error(f"❌ Failed to export config: {e}")
    
    def import_config(self, import_path: str):
        """Import configuration from a file"""
        try:
            with open(import_path, "r", encoding="utf-8") as f:
                imported_data = json.load(f)
            
            with self._lock:
                self._data.update(imported_data)
            self.save()
            self._logger.info(f"📥 Config imported from {import_path}")
        except Exception as e:
            self._logger.error(f"❌ Failed to import config: {e}")
    
    def reset_to_defaults(self):
        """Reset configuration to defaults"""
        with self._lock:
            self._data = {}
        self.save()
        self._logger.info("🔄 Config reset to defaults")
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of current configuration"""
        with self._lock:
            return {
                "config_file": str(self._path),
                "app_settings_count": len(self.get_app_settings()),
                "plugins_count": len(self._data.get("plugins", {})),
                "ui_settings_count": len(self.get_ui_settings()),
                "voice_settings_count": len(self.get_voice_settings()),
                "llm_settings_count": len(self.get_llm_settings()),
                "total_keys": len(self._data)
            }
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate configuration and return issues"""
        issues = []
        warnings = []
        
        # Check required app settings
        app_settings = self.get_app_settings()
        required_app_settings = ["voice_enabled", "default_model"]
        for setting in required_app_settings:
            if setting not in app_settings:
                issues.append(f"Missing required app setting: {setting}")
        
        # Check plugin configurations
        plugins = self._data.get("plugins", {})
        for plugin_name, plugin_config in plugins.items():
            if not isinstance(plugin_config, dict):
                issues.append(f"Invalid plugin config for {plugin_name}")
        
        # Check for deprecated settings
        deprecated_settings = ["old_voice_setting", "legacy_model"]
        for setting in deprecated_settings:
            if setting in app_settings:
                warnings.append(f"Deprecated setting found: {setting}")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings
        }

# Global config manager instance
config_manager = ConfigManager()

# Convenience functions
def get_config() -> ConfigManager:
    """Get the global config manager instance"""
    return config_manager

def get_setting(key: str, default: Any = None) -> Any:
    """Get a global setting"""
    return config_manager.get(key, default)

def set_setting(key: str, value: Any):
    """Set a global setting"""
    config_manager.set(key, value)

def get_plugin_config(plugin_name: str) -> Dict[str, Any]:
    """Get plugin configuration"""
    return config_manager.get_plugin_config(plugin_name)

def set_plugin_config(plugin_name: str, config: Dict[str, Any]):
    """Set plugin configuration"""
    config_manager.set_plugin_config(plugin_name, config)

if __name__ == "__main__":
    # Test the config manager
    print("🧪 Testing Config Manager...")
    
    # Test basic operations
    config = get_config()
    config.set("test_key", "test_value")
    assert config.get("test_key") == "test_value"
    
    # Test plugin config
    config.set_plugin_config("test_plugin", {"setting1": "value1"})
    plugin_config = config.get_plugin_config("test_plugin")
    assert plugin_config["setting1"] == "value1"
    
    # Test app settings
    config.set_app_setting("voice_enabled", True)
    assert config.get_app_setting("voice_enabled") == True
    
    # Test validation
    validation = config.validate_config()
    print(f"Config validation: {validation}")
    
    # Test summary
    summary = config.get_config_summary()
    print(f"Config summary: {summary}")
    
    print("✅ Config manager tests passed!")