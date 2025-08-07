#!/usr/bin/env python3
"""Test script to check imports and basic functionality"""

import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("TestImports")

def test_imports():
    """Test all the main imports"""
    tests = [
        ("asyncio", "asyncio"),
        ("json", "json"),
        ("logging", "logging"),
        ("sys", "sys"),
        ("time", "time"),
        ("pathlib", "pathlib"),
        ("typing", "typing"),
        ("dataclasses", "dataclasses"),
    ]
    
    print("Testing basic Python imports...")
    for name, module in tests:
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError as e:
            print(f"✗ {name}: {e}")
    
    # Test optional dependencies
    optional_tests = [
        ("PyQt6", "PyQt6"),
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("sounddevice", "sounddevice"),
        ("numpy", "numpy"),
        ("psutil", "psutil"),
        ("keyboard", "keyboard"),
        ("whisper", "openai-whisper"),
        ("TTS", "TTS"),
    ]
    
    print("\nTesting optional dependencies...")
    for name, module in optional_tests:
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError as e:
            print(f"✗ {name}: {e}")

def test_local_imports():
    """Test local module imports"""
    local_tests = [
        ("ai_agent", "ai_agent"),
        ("voice_agent", "voice_agent"),
        ("voice_hotkey", "voice_hotkey"),
        ("overlay_ar", "overlay_ar"),
        ("plugin_loader", "plugin_loader"),
        ("logger", "logger"),
    ]
    
    print("\nTesting local module imports...")
    for name, module in local_tests:
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError as e:
            print(f"✗ {name}: {e}")

def test_config():
    """Test configuration loading"""
    print("\nTesting configuration...")
    try:
        import json
        with open("config.json", "r") as f:
            config = json.load(f)
        print("✓ config.json loaded successfully")
        print(f"  - Voice hotkey enabled: {config.get('voice_hotkey_enabled', False)}")
        print(f"  - AR overlay enabled: {config.get('ar_overlay_enabled', False)}")
        print(f"  - Plugins enabled: {config.get('plugins_enabled', False)}")
    except Exception as e:
        print(f"✗ config.json: {e}")

def main():
    """Run all tests"""
    print("=== ChatGPT+ Clone Import Test ===\n")
    
    test_imports()
    test_local_imports()
    test_config()
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    main()