import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent))

# Import automation components
from continuous_voice import AsyncVoiceInput, VoiceActivityDetector
from task_scheduler import TaskScheduler
from plugin_loader import PluginManager
from update_checker import UpdateChecker
from logger import setup_logger, log_manager
from tray_app import TrayApp
from ai_agent import SovereignAgent
from voice_agent import VoiceAgent
from voice_hotkey import VoiceHotkey
from overlay_ar import AROverlay
from profiling.dashboard import ProfilerDashboard
from remote_control.server import main as remote_control_main

# Setup global logger
log = setup_logger("ChatGPTPlus", level=logging.INFO)

CONFIG_PATH = "config.json"

class ChatGPTPlusApp:
    """Main application class with full automation integration"""
    
    def __init__(self):
        self.config = self._load_config()
        self.running = False
        
        # Core components
        self.ai_agent = None
        self.plugin_manager = None
        self.voice_agent = None
        self.voice_input = None
        self.task_scheduler = None
        self.update_checker = None
        self.tray_app = None
        self.ar_overlay = None
        self.profiler_dashboard = None
        
        # Voice activity detection
        self.vad = VoiceActivityDetector(threshold=0.01, min_duration=0.5)
        self.speech_buffer = []
        
        # Tasks
        self.tasks = {}

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            with open(CONFIG_PATH) as f:
                config = json.load(f)
            log.info("Configuration loaded successfully")
            return config
        except FileNotFoundError:
            log.warning(f"Config file {CONFIG_PATH} not found, using defaults")
            return {
                "voice_hotkey_enabled": True,
                "ar_overlay_enabled": True,
                "plugins_enabled": True,
                "remote_control_enabled": True,
                "profiling_enabled": True,
                "auto_update_enabled": True,
                "tray_app_enabled": True,
                "continuous_voice_enabled": True,
                "task_scheduler_enabled": True,
                "voice_settings": {
                    "sample_rate": 16000,
                    "channels": 1,
                    "blocksize": 1024,
                    "device": None
                },
                "ai_models": {
                    "llm_model": "gpt2",
                    "whisper_model": "base",
                    "tts_model": "tts_models/en/ljspeech/tacotron2-DDC"
                },
                "performance": {
                    "gpu_acceleration": True,
                    "max_concurrent_tasks": 5,
                    "memory_limit_mb": 2048
                },
                "security": {
                    "plugin_sandbox": False,
                    "system_control": True,
                    "auto_launch": True
                },
                "ui": {
                    "theme": "dark",
                    "language": "en",
                    "notifications": True
                }
            }

    async def initialize_components(self):
        """Initialize all application components"""
        log.info("Initializing ChatGPT+ Clone components...")
        
        # Initialize AI agent
        self.ai_agent = SovereignAgent()
        log.info("AI agent initialized")
        
        # Initialize plugin system
        plugin_api = self.ai_agent.get_plugin_api()
        self.plugin_manager = PluginManager(plugin_api)
        
        if self.config.get("plugins_enabled", True):
            await self.plugin_manager.load_all_plugins()
            log.info(f"Loaded {self.plugin_manager.get_plugin_count()} plugins")
        
        # Initialize voice components
        if self.config.get("continuous_voice_enabled", True):
            voice_settings = self.config.get("voice_settings", {})
            self.voice_input = AsyncVoiceInput(
                samplerate=voice_settings.get("sample_rate", 16000),
                channels=voice_settings.get("channels", 1),
                blocksize=voice_settings.get("blocksize", 1024),
                device=voice_settings.get("device")
            )
            log.info("Continuous voice input initialized")
        
        # Initialize task scheduler
        if self.config.get("task_scheduler_enabled", True):
            self.task_scheduler = TaskScheduler()
            self._setup_scheduled_tasks()
            log.info("Task scheduler initialized")
        
        # Initialize update checker
        if self.config.get("auto_update_enabled", True):
            self.update_checker = UpdateChecker(
                repo_owner="yourusername",  # Replace with your GitHub username
                repo_name="chatgpt-plus-clone",
                current_version="1.0.0"
            )
            log.info("Update checker initialized")
        
        # Initialize tray app
        if self.config.get("tray_app_enabled", True):
            self.tray_app = TrayApp("ChatGPT+ Clone")
            self.tray_app.run()
            log.info("Tray app initialized")
        
        # Initialize AR overlay
        if self.config.get("ar_overlay_enabled", True):
            self.ar_overlay = AROverlay()
            log.info("AR overlay initialized")
        
        # Initialize profiler dashboard
        if self.config.get("profiling_enabled", True):
            self.profiler_dashboard = ProfilerDashboard()
            log.info("Profiler dashboard initialized")

    def _setup_scheduled_tasks(self):
        """Setup scheduled tasks"""
        if not self.task_scheduler:
            return
        
        # Daily cleanup task
        self.task_scheduler.add_cron_job(
            self._daily_cleanup,
            job_id="daily_cleanup",
            hour=2,
            minute=0
        )
        
        # Hourly status check
        self.task_scheduler.add_cron_job(
            self._hourly_status_check,
            job_id="hourly_status_check",
            minute=0,
            second=0
        )
        
        # Weekly update check
        self.task_scheduler.add_cron_job(
            self._weekly_update_check,
            job_id="weekly_update_check",
            day_of_week="sun",
            hour=10,
            minute=0
        )
        
        log.info("Scheduled tasks configured")

    async def _daily_cleanup(self):
        """Daily cleanup task"""
        log.info("Running daily cleanup...")
        
        # Clean up old logs
        log_manager.cleanup_old_logs(days_to_keep=30)
        
        # Clean up temporary files
        # Add your cleanup logic here
        
        log.info("Daily cleanup completed")

    async def _hourly_status_check(self):
        """Hourly status check task"""
        log.info("Running hourly status check...")
        
        # Check system resources
        import psutil
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        
        log.info(f"System status - CPU: {cpu_percent}%, Memory: {memory_percent}%")
        
        # Alert if resources are high
        if cpu_percent > 80 or memory_percent > 80:
            log.warning(f"High resource usage detected - CPU: {cpu_percent}%, Memory: {memory_percent}%")

    async def _weekly_update_check(self):
        """Weekly update check task"""
        log.info("Running weekly update check...")
        
        if self.update_checker:
            update_info = self.update_checker.get_update_info()
            if update_info.get("update_available", False):
                log.info("Update available, installing...")
                success = self.update_checker.auto_update()
                if success:
                    log.info("Update completed successfully")
                else:
                    log.error("Update failed")

    async def _process_audio_chunk(self, chunk):
        """Process incoming audio chunks"""
        try:
            # Detect voice activity
            if self.vad.detect_activity(chunk):
                self.speech_buffer.append(chunk)
                log.debug("Voice activity detected")
            else:
                # Process accumulated speech if we have enough
                if len(self.speech_buffer) > 0:
                    await self._process_speech()
                    self.speech_buffer = []
            
        except Exception as e:
            log.error(f"Error processing audio chunk: {e}")

    async def _process_speech(self):
        """Process accumulated speech"""
        if not self.speech_buffer:
            return
        
        try:
            # Concatenate speech chunks
            speech_audio = self.vad.get_speech_audio()
            if speech_audio is not None:
                log.info("Processing speech...")
                
                # Here you would integrate with Whisper for transcription
                # For now, we'll simulate transcription
                transcribed_text = "simulated transcribed text"
                
                # Dispatch to plugins
                if self.plugin_manager:
                    handled = await self.plugin_manager.dispatch_voice_command(transcribed_text)
                    if not handled and self.ai_agent:
                        # Generate AI response
                        response = await self.ai_agent.generate_response(transcribed_text)
                        await self.plugin_manager.dispatch_ai_response(response)
                
        except Exception as e:
            log.error(f"Error processing speech: {e}")

    async def start_voice_listening(self):
        """Start continuous voice listening"""
        if not self.voice_input:
            log.warning("Voice input not initialized")
            return
        
        log.info("Starting continuous voice listening...")
        
        try:
            await self.voice_input.start_listening(self._process_audio_chunk)
        except Exception as e:
            log.error(f"Error in voice listening: {e}")

    async def start_remote_control(self):
        """Start remote control server"""
        if not self.config.get("remote_control_enabled", True):
            return
        
        log.info("Starting remote control server...")
        try:
            await remote_control_main()
        except Exception as e:
            log.error(f"Error in remote control server: {e}")

    async def start_profiler_dashboard(self):
        """Start profiler dashboard"""
        if not self.profiler_dashboard:
            return
        
        log.info("Starting profiler dashboard...")
        try:
            await self.profiler_dashboard.start()
        except Exception as e:
            log.error(f"Error in profiler dashboard: {e}")

    async def start_ar_overlay(self):
        """Start AR overlay"""
        if not self.ar_overlay:
            return
        
        log.info("Starting AR overlay...")
        try:
            self.ar_overlay.start()
        except Exception as e:
            log.error(f"Error in AR overlay: {e}")

    async def run(self):
        """Main application loop"""
        self.running = True
        log.info("=== ChatGPT+ Clone Starting ===")
        
        try:
            # Initialize components
            await self.initialize_components()
            
            # Start all services
            tasks = []
            
            # Start voice listening
            if self.voice_input:
                voice_task = asyncio.create_task(self.start_voice_listening())
                tasks.append(voice_task)
            
            # Start remote control server
            remote_task = asyncio.create_task(self.start_remote_control())
            tasks.append(remote_task)
            
            # Start profiler dashboard
            profiler_task = asyncio.create_task(self.start_profiler_dashboard())
            tasks.append(profiler_task)
            
            # Start AR overlay
            if self.ar_overlay:
                ar_task = asyncio.create_task(self.start_ar_overlay())
                tasks.append(ar_task)
            
            # Periodic update check
            async def periodic_update_check():
                while self.running:
                    if self.update_checker:
                        try:
                            self.update_checker.check_for_updates()
                        except Exception as e:
                            log.error(f"Update check failed: {e}")
                    await asyncio.sleep(3600)  # Check every hour
            
            update_task = asyncio.create_task(periodic_update_check())
            tasks.append(update_task)
            
            # Wait for all tasks
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except KeyboardInterrupt:
            log.info("Shutting down gracefully...")
        except Exception as e:
            log.exception(f"Fatal error: {e}")
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Shutdown the application"""
        log.info("Shutting down ChatGPT+ Clone...")
        self.running = False
        
        # Stop all components
        if self.voice_input:
            self.voice_input.stop()
        
        if self.task_scheduler:
            self.task_scheduler.shutdown()
        
        if self.tray_app:
            self.tray_app.stop()
        
        if self.plugin_manager:
            await self.plugin_manager.unload_all_plugins()
        
        log.info("ChatGPT+ Clone shutdown complete")

def main():
    """Main entry point"""
    # Setup exception handler
    def handle_exception(loop, context):
        log.error(f"Unhandled Exception: {context}")
    
    loop = asyncio.get_event_loop()
    loop.set_exception_handler(handle_exception)
    
    # Create and run application
    app = ChatGPTPlusApp()
    
    try:
        loop.run_until_complete(app.run())
    except KeyboardInterrupt:
        log.info("Application interrupted by user")
    except Exception as e:
        log.exception(f"Application error: {e}")
        sys.exit(1)
    finally:
        loop.close()

if __name__ == "__main__":
    main()