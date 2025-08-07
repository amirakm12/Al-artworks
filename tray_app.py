import pystray
from PIL import Image, ImageDraw, ImageFont
import threading
import logging
import os
import sys
import subprocess
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger("TrayApp")

@dataclass
class TrayAction:
    """Represents a tray menu action"""
    name: str
    callback: Callable
    enabled: bool = True
    separator: bool = False

class TrayApp:
    """System tray application with menu controls"""
    
    def __init__(self, app_name: str = "ChatGPT+ Clone"):
        self.app_name = app_name
        self.icon = None
        self.thread = None
        self.running = False
        self.service_running = False
        self.actions = {}
        
        # Create icon
        self.icon_image = self._create_icon()
        
        # Initialize actions
        self._setup_actions()
        
        # Create tray icon
        self._create_tray_icon()

    def _create_icon(self) -> Image.Image:
        """Create a custom icon for the tray"""
        try:
            # Try to load from file first
            icon_path = "assets/icon.png"
            if os.path.exists(icon_path):
                return Image.open(icon_path).resize((64, 64))
        except Exception as e:
            logger.warning(f"Could not load icon from file: {e}")
        
        # Create a simple icon programmatically
        img = Image.new('RGBA', (64, 64), color=(0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Draw a simple AI/robot icon
        # Background circle
        draw.ellipse([8, 8, 56, 56], fill=(52, 152, 219, 255), outline=(41, 128, 185, 255), width=2)
        
        # Eyes
        draw.ellipse([20, 20, 30, 30], fill=(255, 255, 255, 255))
        draw.ellipse([34, 20, 44, 30], fill=(255, 255, 255, 255))
        draw.ellipse([22, 22, 28, 28], fill=(0, 0, 0, 255))
        draw.ellipse([36, 22, 42, 28], fill=(0, 0, 0, 255))
        
        # Mouth
        draw.arc([25, 35, 39, 45], start=0, end=180, fill=(255, 255, 255, 255), width=2)
        
        # Add "AI" text
        try:
            # Try to use a system font
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        draw.text((28, 50), "AI", fill=(255, 255, 255, 255), font=font)
        
        return img

    def _setup_actions(self):
        """Setup tray menu actions"""
        self.actions = {
            "start": TrayAction("Start Service", self._start_service),
            "stop": TrayAction("Stop Service", self._stop_service),
            "status": TrayAction("Show Status", self._show_status),
            "settings": TrayAction("Settings", self._open_settings),
            "logs": TrayAction("View Logs", self._view_logs),
            "update": TrayAction("Check for Updates", self._check_updates),
            "restart": TrayAction("Restart Application", self._restart_app),
            "separator1": TrayAction("", None, separator=True),
            "separator2": TrayAction("", None, separator=True),
            "quit": TrayAction("Quit", self._quit_app)
        }

    def _create_tray_icon(self):
        """Create the system tray icon with menu"""
        menu_items = []
        
        for action_name, action in self.actions.items():
            if action.separator:
                menu_items.append(pystray.MenuItem.SEPARATOR)
            else:
                menu_items.append(
                    pystray.MenuItem(
                        action.name,
                        action.callback,
                        enabled=action.enabled
                    )
                )
        
        self.icon = pystray.Icon(
            "chatgpt_plus",
            self.icon_image,
            self.app_name,
            pystray.Menu(*menu_items)
        )

    def _start_service(self, icon, item):
        """Start the ChatGPT+ service"""
        try:
            logger.info("Starting ChatGPT+ service...")
            self.service_running = True
            self._update_menu_state()
            
            # Here you would start your actual service
            # For example, start the main application
            self._start_main_app()
            
            logger.info("ChatGPT+ service started successfully")
        except Exception as e:
            logger.error(f"Failed to start service: {e}")

    def _stop_service(self, icon, item):
        """Stop the ChatGPT+ service"""
        try:
            logger.info("Stopping ChatGPT+ service...")
            self.service_running = False
            self._update_menu_state()
            
            # Here you would stop your actual service
            self._stop_main_app()
            
            logger.info("ChatGPT+ service stopped successfully")
        except Exception as e:
            logger.error(f"Failed to stop service: {e}")

    def _show_status(self, icon, item):
        """Show application status"""
        try:
            status_info = self._get_status_info()
            
            # Create a simple status window or notification
            self._show_notification(
                "ChatGPT+ Status",
                f"Service: {'Running' if self.service_running else 'Stopped'}\n"
                f"Uptime: {status_info.get('uptime', 'Unknown')}\n"
                f"Memory: {status_info.get('memory_usage', 'Unknown')}\n"
                f"CPU: {status_info.get('cpu_usage', 'Unknown')}"
            )
            
            logger.info("Status displayed")
        except Exception as e:
            logger.error(f"Failed to show status: {e}")

    def _open_settings(self, icon, item):
        """Open settings dialog"""
        try:
            logger.info("Opening settings...")
            
            # Here you would open your settings dialog
            # For now, just show a notification
            self._show_notification(
                "Settings",
                "Settings dialog would open here"
            )
        except Exception as e:
            logger.error(f"Failed to open settings: {e}")

    def _view_logs(self, icon, item):
        """Open log viewer"""
        try:
            logger.info("Opening log viewer...")
            
            # Open logs directory in file explorer
            logs_dir = "logs"
            if os.path.exists(logs_dir):
                if sys.platform == "win32":
                    subprocess.run(["explorer", logs_dir])
                elif sys.platform == "darwin":
                    subprocess.run(["open", logs_dir])
                else:
                    subprocess.run(["xdg-open", logs_dir])
            else:
                self._show_notification(
                    "Logs",
                    "Logs directory not found"
                )
        except Exception as e:
            logger.error(f"Failed to open logs: {e}")

    def _check_updates(self, icon, item):
        """Check for application updates"""
        try:
            logger.info("Checking for updates...")
            
            # Here you would implement update checking
            # For now, just show a notification
            self._show_notification(
                "Updates",
                "Checking for updates...\nNo updates available."
            )
        except Exception as e:
            logger.error(f"Failed to check updates: {e}")

    def _restart_app(self, icon, item):
        """Restart the application"""
        try:
            logger.info("Restarting application...")
            
            # Restart the current process
            python = sys.executable
            os.execl(python, python, *sys.argv)
        except Exception as e:
            logger.error(f"Failed to restart app: {e}")

    def _quit_app(self, icon, item):
        """Quit the application"""
        try:
            logger.info("Quitting application...")
            
            # Stop service if running
            if self.service_running:
                self._stop_service(icon, item)
            
            # Stop tray icon
            self.stop()
            
            # Exit application
            sys.exit(0)
        except Exception as e:
            logger.error(f"Failed to quit app: {e}")

    def _update_menu_state(self):
        """Update menu item states based on service status"""
        if self.icon:
            # Update start/stop menu items
            for item in self.icon.menu:
                if hasattr(item, 'text'):
                    if item.text == "Start Service":
                        item.enabled = not self.service_running
                    elif item.text == "Stop Service":
                        item.enabled = self.service_running

    def _get_status_info(self) -> Dict[str, Any]:
        """Get current application status information"""
        import psutil
        import time
        
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                "uptime": f"{time.time() - process.create_time():.1f}s",
                "memory_usage": f"{memory_info.rss / 1024 / 1024:.1f} MB",
                "cpu_usage": f"{process.cpu_percent():.1f}%",
                "service_running": self.service_running
            }
        except Exception as e:
            logger.error(f"Failed to get status info: {e}")
            return {"error": str(e)}

    def _show_notification(self, title: str, message: str):
        """Show a system notification"""
        try:
            if self.icon:
                self.icon.notify(title, message)
        except Exception as e:
            logger.error(f"Failed to show notification: {e}")

    def _start_main_app(self):
        """Start the main application"""
        # This would start your main ChatGPT+ application
        # For now, just log that it would start
        logger.info("Main application would start here")

    def _stop_main_app(self):
        """Stop the main application"""
        # This would stop your main ChatGPT+ application
        # For now, just log that it would stop
        logger.info("Main application would stop here")

    def run(self):
        """Run the tray application"""
        if self.thread and self.thread.is_alive():
            logger.warning("Tray app is already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run_tray, daemon=True)
        self.thread.start()
        
        logger.info("Tray application started")

    def _run_tray(self):
        """Internal method to run the tray icon"""
        try:
            self.icon.run()
        except Exception as e:
            logger.error(f"Tray icon error: {e}")

    def stop(self):
        """Stop the tray application"""
        self.running = False
        if self.icon:
            self.icon.stop()
        logger.info("Tray application stopped")

    def is_running(self) -> bool:
        """Check if tray app is running"""
        return self.running and self.thread and self.thread.is_alive()

    def get_status(self) -> Dict[str, Any]:
        """Get tray application status"""
        return {
            "running": self.running,
            "service_running": self.service_running,
            "thread_alive": self.thread.is_alive() if self.thread else False,
            "icon_visible": self.icon.visible if self.icon else False
        }

# Example usage
def example_tray_app():
    """Example of using the tray application"""
    logging.basicConfig(level=logging.INFO)
    
    # Create tray app
    tray = TrayApp("ChatGPT+ Clone")
    
    try:
        # Start tray app
        tray.run()
        
        # Keep running for a while
        import time
        time.sleep(30)
        
        # Get status
        status = tray.get_status()
        logger.info(f"Tray status: {status}")
        
    except KeyboardInterrupt:
        logger.info("Stopping tray app...")
    finally:
        tray.stop()

if __name__ == "__main__":
    example_tray_app()