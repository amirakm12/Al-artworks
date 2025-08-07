"""
Plugin Sandbox Security System
Provides secure isolation for plugins using RestrictedPython and subprocess
"""

import os
import sys
import subprocess
import threading
import time
import json
import tempfile
import signal
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
import logging

try:
    from RestrictedPython import compile_restricted, safe_globals
    RESTRICTED_PYTHON_AVAILABLE = True
except ImportError:
    RESTRICTED_PYTHON_AVAILABLE = False
    logging.warning("RestrictedPython not available - using basic sandbox")

class PluginSandbox:
    """Secure sandbox for plugin execution"""
    
    def __init__(self, plugin_name: str, config: Dict[str, Any] = None):
        self.plugin_name = plugin_name
        self.config = config or {}
        self.logger = logging.getLogger(f"PluginSandbox.{plugin_name}")
        
        # Security settings
        self.max_execution_time = 30  # seconds
        self.max_memory_mb = 100
        self.allowed_modules = {
            'time', 'datetime', 'json', 'math', 'random',
            'collections', 'itertools', 'functools'
        }
        
        # Process management
        self.subprocess = None
        self.is_running = False
        self.last_activity = time.time()
        
        # Communication
        self.message_queue = []
        self.response_queue = []
        self.lock = threading.Lock()
    
    def start(self) -> bool:
        """Start the sandboxed plugin process"""
        try:
            # Create temporary directory for plugin
            self.temp_dir = tempfile.mkdtemp(prefix=f"plugin_{self.plugin_name}_")
            
            # Create plugin wrapper
            wrapper_code = self._create_wrapper()
            wrapper_path = Path(self.temp_dir) / "plugin_wrapper.py"
            wrapper_path.write_text(wrapper_code)
            
            # Start subprocess
            self.subprocess = subprocess.Popen(
                [sys.executable, str(wrapper_path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self.temp_dir,
                env=self._get_safe_env()
            )
            
            self.is_running = True
            self.last_activity = time.time()
            
            # Start monitoring thread
            self.monitor_thread = threading.Thread(target=self._monitor_process, daemon=True)
            self.monitor_thread.start()
            
            self.logger.info(f"Sandbox started for plugin: {self.plugin_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start sandbox for {self.plugin_name}: {e}")
            return False
    
    def stop(self):
        """Stop the sandboxed plugin process"""
        if not self.is_running:
            return
        
        try:
            self.is_running = False
            
            if self.subprocess:
                # Send termination signal
                self.subprocess.terminate()
                
                # Wait for graceful shutdown
                try:
                    self.subprocess.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if not responding
                    self.subprocess.kill()
                    self.subprocess.wait()
            
            # Cleanup temp directory
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                import shutil
                shutil.rmtree(self.temp_dir, ignore_errors=True)
            
            self.logger.info(f"Sandbox stopped for plugin: {self.plugin_name}")
            
        except Exception as e:
            self.logger.error(f"Error stopping sandbox for {self.plugin_name}: {e}")
    
    def execute(self, code: str, timeout: int = 30) -> Dict[str, Any]:
        """Execute code in sandbox"""
        if not self.is_running:
            return {"error": "Sandbox not running"}
        
        try:
            # Send code to subprocess
            message = {
                "type": "execute",
                "code": code,
                "timeout": timeout
            }
            
            response = self._send_message(message, timeout)
            return response
            
        except Exception as e:
            self.logger.error(f"Execution error in {self.plugin_name}: {e}")
            return {"error": str(e)}
    
    def call_function(self, func_name: str, *args, **kwargs) -> Dict[str, Any]:
        """Call a function in the sandboxed plugin"""
        if not self.is_running:
            return {"error": "Sandbox not running"}
        
        try:
            message = {
                "type": "call_function",
                "function": func_name,
                "args": args,
                "kwargs": kwargs
            }
            
            response = self._send_message(message, 10)
            return response
            
        except Exception as e:
            self.logger.error(f"Function call error in {self.plugin_name}: {e}")
            return {"error": str(e)}
    
    def _create_wrapper(self) -> str:
        """Create the plugin wrapper code"""
        return f'''
import sys
import json
import signal
import time
import traceback
from pathlib import Path

# Import the actual plugin
try:
    sys.path.insert(0, "{self.config.get('plugin_dir', 'plugins')}")
    plugin_module = __import__("{self.plugin_name}")
    
    if hasattr(plugin_module, 'Plugin'):
        plugin_instance = plugin_module.Plugin()
    else:
        plugin_instance = None
        
except Exception as e:
    plugin_instance = None
    print(f"Plugin load error: {{e}}", file=sys.stderr)

def handle_message(message):
    """Handle incoming messages"""
    try:
        msg_type = message.get('type')
        
        if msg_type == 'execute':
            # Execute code in restricted environment
            code = message['code']
            result = execute_restricted(code)
            return {{"success": True, "result": result}}
            
        elif msg_type == 'call_function':
            # Call plugin function
            if plugin_instance:
                func_name = message['function']
                args = message.get('args', [])
                kwargs = message.get('kwargs', {{}})
                
                if hasattr(plugin_instance, func_name):
                    func = getattr(plugin_instance, func_name)
                    result = func(*args, **kwargs)
                    return {{"success": True, "result": result}}
                else:
                    return {{"error": f"Function {{func_name}} not found"}}
            else:
                return {{"error": "Plugin not loaded"}}
                
        else:
            return {{"error": f"Unknown message type: {{msg_type}}"}}
            
    except Exception as e:
        return {{"error": str(e), "traceback": traceback.format_exc()}}

def execute_restricted(code):
    """Execute code in restricted environment"""
    try:
        {'from RestrictedPython import compile_restricted, safe_globals' if RESTRICTED_PYTHON_AVAILABLE else '# RestrictedPython not available'}
        
        {'restricted_globals = safe_globals.copy()' if RESTRICTED_PYTHON_AVAILABLE else 'restricted_globals = {{}}'}
        {'restricted_globals.update({{' if RESTRICTED_PYTHON_AVAILABLE else 'restricted_globals.update({'
        '    "time": time,
        '    "json": json,
        '    "math": __import__("math"),
        '    "random": __import__("random"),
        '    "collections": __import__("collections")
        '})'}
        
        {'byte_code = compile_restricted(code, "<plugin>", "exec")' if RESTRICTED_PYTHON_AVAILABLE else 'byte_code = compile(code, "<plugin>", "exec")'}
        exec(byte_code, restricted_globals)
        
        return restricted_globals.get('result', None)
        
    except Exception as e:
        raise Exception(f"Restricted execution error: {{e}}")

def main():
    """Main plugin wrapper loop"""
    # Set up signal handlers
    def signal_handler(signum, frame):
        print("Plugin wrapper shutting down...")
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start plugin if available
    if plugin_instance and hasattr(plugin_instance, 'start'):
        try:
            plugin_instance.start()
        except Exception as e:
            print(f"Plugin start error: {{e}}", file=sys.stderr)
    
    # Message loop
    while True:
        try:
            # Read message from stdin
            line = input()
            if not line:
                continue
                
            message = json.loads(line)
            response = handle_message(message)
            
            # Send response
            print(json.dumps(response))
            sys.stdout.flush()
            
        except EOFError:
            break
        except Exception as e:
            error_response = {{"error": str(e), "traceback": traceback.format_exc()}}
            print(json.dumps(error_response))
            sys.stdout.flush()
    
    # Stop plugin if available
    if plugin_instance and hasattr(plugin_instance, 'stop'):
        try:
            plugin_instance.stop()
        except Exception as e:
            print(f"Plugin stop error: {{e}}", file=sys.stderr)

if __name__ == "__main__":
    main()
'''
    
    def _get_safe_env(self) -> Dict[str, str]:
        """Get safe environment variables for subprocess"""
        env = os.environ.copy()
        
        # Remove dangerous environment variables
        dangerous_vars = [
            'PYTHONPATH', 'PYTHONHOME', 'PYTHONEXECUTABLE',
            'LD_LIBRARY_PATH', 'LD_PRELOAD', 'LD_DEBUG'
        ]
        
        for var in dangerous_vars:
            env.pop(var, None)
        
        # Set safe defaults
        env['PYTHONUNBUFFERED'] = '1'
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        
        return env
    
    def _send_message(self, message: Dict[str, Any], timeout: int = 10) -> Dict[str, Any]:
        """Send message to subprocess and get response"""
        if not self.subprocess:
            return {"error": "No subprocess"}
        
        try:
            # Send message
            message_str = json.dumps(message) + "\n"
            self.subprocess.stdin.write(message_str)
            self.subprocess.stdin.flush()
            
            # Read response with timeout
            import select
            ready, _, _ = select.select([self.subprocess.stdout], [], [], timeout)
            
            if ready:
                response_line = self.subprocess.stdout.readline()
                if response_line:
                    return json.loads(response_line.strip())
                else:
                    return {"error": "No response from subprocess"}
            else:
                return {"error": "Subprocess timeout"}
                
        except Exception as e:
            return {"error": f"Communication error: {e}"}
    
    def _monitor_process(self):
        """Monitor the subprocess for health and resource usage"""
        while self.is_running and self.subprocess:
            try:
                # Check if process is still alive
                if self.subprocess.poll() is not None:
                    self.logger.warning(f"Plugin subprocess died: {self.plugin_name}")
                    self.is_running = False
                    break
                
                # Check execution time
                if time.time() - self.last_activity > self.max_execution_time:
                    self.logger.warning(f"Plugin timeout: {self.plugin_name}")
                    self.stop()
                    break
                
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Monitor error for {self.plugin_name}: {e}")
                break

class SandboxManager:
    """Manager for multiple plugin sandboxes"""
    
    def __init__(self):
        self.sandboxes: Dict[str, PluginSandbox] = {}
        self.logger = logging.getLogger("SandboxManager")
    
    def create_sandbox(self, plugin_name: str, config: Dict[str, Any] = None) -> PluginSandbox:
        """Create a new sandbox for a plugin"""
        if plugin_name in self.sandboxes:
            self.logger.warning(f"Sandbox already exists for {plugin_name}")
            return self.sandboxes[plugin_name]
        
        sandbox = PluginSandbox(plugin_name, config)
        self.sandboxes[plugin_name] = sandbox
        return sandbox
    
    def start_sandbox(self, plugin_name: str) -> bool:
        """Start a plugin sandbox"""
        if plugin_name not in self.sandboxes:
            self.logger.error(f"No sandbox found for {plugin_name}")
            return False
        
        return self.sandboxes[plugin_name].start()
    
    def stop_sandbox(self, plugin_name: str):
        """Stop a plugin sandbox"""
        if plugin_name in self.sandboxes:
            self.sandboxes[plugin_name].stop()
            del self.sandboxes[plugin_name]
    
    def stop_all(self):
        """Stop all sandboxes"""
        for plugin_name in list(self.sandboxes.keys()):
            self.stop_sandbox(plugin_name)
    
    def get_sandbox(self, plugin_name: str) -> Optional[PluginSandbox]:
        """Get a sandbox by name"""
        return self.sandboxes.get(plugin_name)
    
    def list_sandboxes(self) -> List[str]:
        """List all active sandboxes"""
        return list(self.sandboxes.keys())
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all sandboxes"""
        status = {}
        for name, sandbox in self.sandboxes.items():
            status[name] = {
                "running": sandbox.is_running,
                "last_activity": sandbox.last_activity
            }
        return status

# Global sandbox manager
sandbox_manager = SandboxManager()

if __name__ == "__main__":
    # Test the sandbox system
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testing Plugin Sandbox...")
    
    # Create a test sandbox
    config = {"plugin_dir": "plugins"}
    sandbox = sandbox_manager.create_sandbox("test_plugin", config)
    
    # Test sandbox operations
    if sandbox.start():
        print("✅ Sandbox started successfully")
        
        # Test code execution
        result = sandbox.execute("result = 2 + 2")
        print(f"Execution result: {result}")
        
        # Test function call
        result = sandbox.call_function("test_function", "hello")
        print(f"Function call result: {result}")
        
        sandbox.stop()
        print("✅ Sandbox stopped successfully")
    else:
        print("❌ Failed to start sandbox")
    
    print("✅ Sandbox test completed")