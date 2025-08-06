"""
Configuration Manager for Athena 3D Avatar
Cosmic-themed settings with performance optimization
"""

import yaml
import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import logging

class PerformanceMode(Enum):
    ULTRA_LIGHT = "ultra_light"
    LIGHT = "light"
    MEDIUM = "medium"
    HEAVY = "heavy"

class VoiceQuality(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"

@dataclass
class CosmicConfig:
    """Cosmic configuration for Athena"""
    # Performance settings
    performance_mode: PerformanceMode = PerformanceMode.MEDIUM
    max_memory_gb: float = 12.0
    target_latency_ms: float = 250.0
    target_fps: float = 60.0
    
    # Voice settings
    voice_quality: VoiceQuality = VoiceQuality.MEDIUM
    enable_lip_sync: bool = True
    enable_emotion: bool = True
    default_voice_tone: str = "wisdom"
    
    # Rendering settings
    rendering_quality: int = 3  # 1-5
    enable_cosmic_effects: bool = True
    enable_holographic_veins: bool = True
    enable_marble_texture: bool = True
    
    # Appearance settings
    robe_color: str = "marble_white"
    wreath_style: str = "laurel"
    metallic_arms: bool = True
    holographic_veins: bool = True
    
    # Animation settings
    animation_speed: float = 1.0
    enable_gestures: bool = True
    enable_facial_expressions: bool = True
    
    # UI settings
    cosmic_theme: bool = True
    enable_performance_monitoring: bool = True
    auto_save_settings: bool = True
    
    # Advanced settings
    enable_quantization: bool = True
    enable_pruning: bool = True
    enable_fusion: bool = True
    enable_compilation: bool = True

class ConfigManager:
    """Configuration manager for Athena 3D Avatar"""
    
    def __init__(self, config_file: str = "athena_config.yaml"):
        self.config_file = config_file
        self.logger = logging.getLogger(__name__)
        self.config = CosmicConfig()
        
        # Load configuration
        self.load_config()
        
    def load_config(self):
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                    
                # Update config with loaded data
                for key, value in config_data.items():
                    if hasattr(self.config, key):
                        # Handle enum values
                        if key == 'performance_mode':
                            value = PerformanceMode(value)
                        elif key == 'voice_quality':
                            value = VoiceQuality(value)
                        
                        setattr(self.config, key, value)
                
                self.logger.info(f"Configuration loaded from {self.config_file}")
            else:
                self.logger.info("No configuration file found, using defaults")
                self.save_config()
                
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
    
    def save_config(self):
        """Save configuration to file"""
        try:
            # Convert config to dict
            config_dict = asdict(self.config)
            
            # Convert enums to strings
            config_dict['performance_mode'] = config_dict['performance_mode'].value
            config_dict['voice_quality'] = config_dict['voice_quality'].value
            
            # Save to file
            with open(self.config_file, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            self.logger.info(f"Configuration saved to {self.config_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
    
    def get_performance_settings(self) -> Dict[str, Any]:
        """Get performance-related settings"""
        return {
            'performance_mode': self.config.performance_mode.value,
            'max_memory_gb': self.config.max_memory_gb,
            'target_latency_ms': self.config.target_latency_ms,
            'target_fps': self.config.target_fps,
            'rendering_quality': self.config.rendering_quality,
            'enable_quantization': self.config.enable_quantization,
            'enable_pruning': self.config.enable_pruning,
            'enable_fusion': self.config.enable_fusion,
            'enable_compilation': self.config.enable_compilation
        }
    
    def get_voice_settings(self) -> Dict[str, Any]:
        """Get voice-related settings"""
        return {
            'voice_quality': self.config.voice_quality.value,
            'enable_lip_sync': self.config.enable_lip_sync,
            'enable_emotion': self.config.enable_emotion,
            'default_voice_tone': self.config.default_voice_tone
        }
    
    def get_rendering_settings(self) -> Dict[str, Any]:
        """Get rendering-related settings"""
        return {
            'rendering_quality': self.config.rendering_quality,
            'enable_cosmic_effects': self.config.enable_cosmic_effects,
            'enable_holographic_veins': self.config.enable_holographic_veins,
            'enable_marble_texture': self.config.enable_marble_texture
        }
    
    def get_appearance_settings(self) -> Dict[str, Any]:
        """Get appearance-related settings"""
        return {
            'robe_color': self.config.robe_color,
            'wreath_style': self.config.wreath_style,
            'metallic_arms': self.config.metallic_arms,
            'holographic_veins': self.config.holographic_veins
        }
    
    def get_animation_settings(self) -> Dict[str, Any]:
        """Get animation-related settings"""
        return {
            'animation_speed': self.config.animation_speed,
            'enable_gestures': self.config.enable_gestures,
            'enable_facial_expressions': self.config.enable_facial_expressions
        }
    
    def get_ui_settings(self) -> Dict[str, Any]:
        """Get UI-related settings"""
        return {
            'cosmic_theme': self.config.cosmic_theme,
            'enable_performance_monitoring': self.config.enable_performance_monitoring,
            'auto_save_settings': self.config.auto_save_settings
        }
    
    def update_setting(self, category: str, setting: str, value: Any):
        """Update a specific setting"""
        try:
            if hasattr(self.config, setting):
                # Handle enum values
                if setting == 'performance_mode':
                    value = PerformanceMode(value)
                elif setting == 'voice_quality':
                    value = VoiceQuality(value)
                
                setattr(self.config, setting, value)
                
                # Auto-save if enabled
                if self.config.auto_save_settings:
                    self.save_config()
                
                self.logger.info(f"Updated setting: {category}.{setting} = {value}")
            else:
                self.logger.warning(f"Unknown setting: {setting}")
                
        except Exception as e:
            self.logger.error(f"Failed to update setting {setting}: {e}")
    
    def get_optimization_level(self) -> int:
        """Get optimization level based on performance mode"""
        optimization_levels = {
            PerformanceMode.ULTRA_LIGHT: 1,
            PerformanceMode.LIGHT: 2,
            PerformanceMode.MEDIUM: 3,
            PerformanceMode.HEAVY: 4
        }
        return optimization_levels.get(self.config.performance_mode, 3)
    
    def get_memory_budget(self) -> float:
        """Get memory budget in GB"""
        return self.config.max_memory_gb
    
    def get_target_latency(self) -> float:
        """Get target latency in milliseconds"""
        return self.config.target_latency_ms
    
    def get_target_fps(self) -> float:
        """Get target FPS"""
        return self.config.target_fps
    
    def is_high_performance_mode(self) -> bool:
        """Check if high performance mode is enabled"""
        return self.config.performance_mode in [PerformanceMode.ULTRA_LIGHT, PerformanceMode.LIGHT]
    
    def is_quality_mode(self) -> bool:
        """Check if quality mode is enabled"""
        return self.config.performance_mode in [PerformanceMode.MEDIUM, PerformanceMode.HEAVY]
    
    def get_cosmic_theme_enabled(self) -> bool:
        """Check if cosmic theme is enabled"""
        return self.config.cosmic_theme
    
    def get_voice_quality_level(self) -> int:
        """Get voice quality level (1-4)"""
        quality_levels = {
            VoiceQuality.LOW: 1,
            VoiceQuality.MEDIUM: 2,
            VoiceQuality.HIGH: 3,
            VoiceQuality.ULTRA: 4
        }
        return quality_levels.get(self.config.voice_quality, 2)
    
    def export_config(self, export_file: str):
        """Export configuration to file"""
        try:
            config_dict = asdict(self.config)
            
            # Convert enums to strings
            config_dict['performance_mode'] = config_dict['performance_mode'].value
            config_dict['voice_quality'] = config_dict['voice_quality'].value
            
            with open(export_file, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            self.logger.info(f"Configuration exported to {export_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to export configuration: {e}")
    
    def import_config(self, import_file: str):
        """Import configuration from file"""
        try:
            with open(import_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Update config with imported data
            for key, value in config_data.items():
                if hasattr(self.config, key):
                    # Handle enum values
                    if key == 'performance_mode':
                        value = PerformanceMode(value)
                    elif key == 'voice_quality':
                        value = VoiceQuality(value)
                    
                    setattr(self.config, key, value)
            
            # Save imported config
            self.save_config()
            
            self.logger.info(f"Configuration imported from {import_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to import configuration: {e}")
    
    def reset_to_defaults(self):
        """Reset configuration to defaults"""
        try:
            self.config = CosmicConfig()
            self.save_config()
            self.logger.info("Configuration reset to defaults")
            
        except Exception as e:
            self.logger.error(f"Failed to reset configuration: {e}")
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary"""
        return {
            'performance_mode': self.config.performance_mode.value,
            'memory_budget_gb': self.config.max_memory_gb,
            'target_latency_ms': self.config.target_latency_ms,
            'voice_quality': self.config.voice_quality.value,
            'rendering_quality': self.config.rendering_quality,
            'cosmic_theme': self.config.cosmic_theme,
            'auto_save': self.config.auto_save_settings
        }