"""
Avatar components for Athena 3D Avatar
3D model, voice synthesis, and animations
"""

from .athena_model import AthenaModel, AvatarPart, MaterialProperties
from .voice_agent import BarkVoiceAgent, LipSyncAgent, VoiceTone, VoiceConfig
from .animation_controller import AnimationController, AnimationType, AnimationConfig

__all__ = [
    'AthenaModel',
    'AvatarPart',
    'MaterialProperties',
    'BarkVoiceAgent',
    'LipSyncAgent', 
    'VoiceTone',
    'VoiceConfig',
    'AnimationController',
    'AnimationType',
    'AnimationConfig'
]