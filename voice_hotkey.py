import keyboard
import threading
import logging
import asyncio
from typing import Optional, Callable

log = logging.getLogger("VoiceHotkey")

class VoiceHotkey:
    """Global hotkey listener for voice activation"""
    
    def __init__(self, hotkey: str = "ctrl+shift+v", callback: Optional[Callable] = None):
        self.hotkey = hotkey
        self.callback = callback
        self.running = False
        self.listener_thread = None
        
        log.info(f"VoiceHotkey initialized with hotkey: {hotkey}")

    def start(self):
        """Start the hotkey listener"""
        if self.running:
            log.warning("VoiceHotkey already running")
            return
        
        self.running = True
        log.info(f"Starting voice hotkey listener on {self.hotkey}")
        
        # Add the hotkey
        keyboard.add_hotkey(self.hotkey, self._trigger_voice)
        
        # Start listener thread
        self.listener_thread = threading.Thread(target=self._listener_loop, daemon=True)
        self.listener_thread.start()
        
        log.info("Voice hotkey listener started successfully")

    def _trigger_voice(self):
        """Called when hotkey is pressed"""
        log.info("Voice hotkey triggered")
        
        if self.callback:
            try:
                # Run callback in asyncio event loop if available
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.run_coroutine_threadsafe(self.callback(), loop)
                else:
                    # If no event loop, run synchronously
                    self.callback()
            except Exception as e:
                log.error(f"Error in voice hotkey callback: {e}")

    def _listener_loop(self):
        """Main listener loop"""
        try:
            keyboard.wait()
        except Exception as e:
            log.error(f"Error in keyboard listener: {e}")
        finally:
            self.running = False

    def stop(self):
        """Stop the hotkey listener"""
        if not self.running:
            return
        
        self.running = False
        keyboard.remove_hotkey(self.hotkey)
        
        if self.listener_thread and self.listener_thread.is_alive():
            self.listener_thread.join(timeout=1.0)
        
        log.info("Voice hotkey listener stopped")

    def set_callback(self, callback: Callable):
        """Set the callback function for hotkey triggers"""
        self.callback = callback
        log.info("Voice hotkey callback set")

    def get_status(self) -> dict:
        """Get hotkey status"""
        return {
            "running": self.running,
            "hotkey": self.hotkey,
            "callback_set": self.callback is not None
        }

# Example usage
async def example_voice_callback():
    """Example callback for voice hotkey"""
    log.info("Voice hotkey callback executed")
    # Here you would typically start voice recording
    # or toggle voice listening state

def example_sync_callback():
    """Example synchronous callback"""
    log.info("Voice hotkey sync callback executed")

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create hotkey listener
    hotkey = VoiceHotkey("ctrl+shift+v", example_sync_callback)
    
    try:
        # Start listener
        hotkey.start()
        
        # Keep running
        while True:
            import time
            time.sleep(1)
            
    except KeyboardInterrupt:
        log.info("Stopping voice hotkey...")
        hotkey.stop()