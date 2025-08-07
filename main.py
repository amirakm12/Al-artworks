import asyncio
import json
import logging
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent))

from plugin_loader import PluginManager
from voice_agent import VoiceAgent
from voice_hotkey import VoiceHotkey
from overlay_ar import AROverlay
from profiling.dashboard import ProfilerDashboard
from remote_control.server import main as remote_control_main
from utils.logger import setup_logger
from ai_agent import SovereignAgent

# Setup global logger
log = setup_logger()

CONFIG_PATH = "config.json"

async def main():
    """Main entry point with full async integration"""
    try:
        # Load config
        with open(CONFIG_PATH) as f:
            config = json.load(f)
        log.info("Configuration loaded successfully")
    except FileNotFoundError:
        log.warning(f"Config file {CONFIG_PATH} not found, using defaults")
        config = {
            "voice_hotkey_enabled": True,
            "ar_overlay_enabled": True,
            "plugins_enabled": True,
            "remote_control_enabled": True,
            "profiling_enabled": True
        }

    # Initialize core components
    log.info("Initializing ChatGPT+ Clone system...")
    
    # Init AI agent and plugin system
    ai_agent = SovereignAgent()
    plugin_api = ai_agent.get_plugin_api()
    plugin_manager = PluginManager(plugin_api)
    
    # Load plugins
    if config.get("plugins_enabled", True):
        plugin_paths = ["plugins/sample_plugin.py"]
        await plugin_manager.load_plugins(plugin_paths)
        log.info(f"Loaded {len(plugin_manager.plugins)} plugins")

    # Start voice hotkey if enabled
    if config.get("voice_hotkey_enabled", True):
        voice_hotkey = VoiceHotkey()
        voice_hotkey.start()
        log.info("Voice hotkey listener started")

    # Start AR Overlay if enabled
    if config.get("ar_overlay_enabled", True):
        ar_overlay = AROverlay()
        ar_overlay.start()
        log.info("AR Overlay started")

    # Start profiler dashboard
    if config.get("profiling_enabled", True):
        profiler = ProfilerDashboard()
        profiler_task = asyncio.create_task(profiler.start())
        log.info("Profiler dashboard started")

    # Start remote control server
    if config.get("remote_control_enabled", True):
        remote_task = asyncio.create_task(remote_control_main())
        log.info("Remote control server started")

    # Start voice agent pipeline
    voice_agent = VoiceAgent(plugin_manager)
    log.info("Starting voice agent...")
    await voice_agent.start()

if __name__ == "__main__":
    # Setup exception handler
    def handle_exception(loop, context):
        log.error(f"Unhandled Exception: {context}")
    
    loop = asyncio.get_event_loop()
    loop.set_exception_handler(handle_exception)
    
    try:
        log.info("=== ChatGPT+ Clone Starting ===")
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        log.info("Shutting down gracefully...")
    except Exception as e:
        log.exception(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        loop.close()
        log.info("System shutdown complete")