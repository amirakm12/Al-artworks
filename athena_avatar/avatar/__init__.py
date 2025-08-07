"""
Avatar components for Athena 3D Avatar
3D model, voice synthesis, animations, gestures, and emotions
"""

from .athena_model import AthenaModel, AvatarPart, MaterialProperties
from .voice_agent import BarkVoiceAgent, LipSyncAgent, VoiceTone, VoiceConfig
from .animation_controller import AnimationController, AnimationType, AnimationConfig, BaseAnimation
from .gesture_system import GestureSystem, GestureType, GestureConfig, BaseGestureModel
from .emotion_engine import EmotionEngine, EmotionType, EmotionConfig, BaseEmotionModel

__all__ = [
    # Core avatar model
    'AthenaModel',
    'AvatarPart', 
    'MaterialProperties',
    
    # Voice synthesis
    'BarkVoiceAgent',
    'LipSyncAgent',
    'VoiceTone',
    'VoiceConfig',
    
    # Animation system
    'AnimationController',
    'AnimationType',
    'AnimationConfig',
    'BaseAnimation',
    
    # Gesture system
    'GestureSystem',
    'GestureType',
    'GestureConfig',
    'BaseGestureModel',
    
    # Emotion engine
    'EmotionEngine',
    'EmotionType',
    'EmotionConfig',
    'BaseEmotionModel'
]