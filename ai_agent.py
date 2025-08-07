import asyncio
import logging
import subprocess
import os
import sys
import torch
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

log = logging.getLogger("SovereignAgent")

@dataclass
class SystemInfo:
    platform: str
    python_version: str
    cpu_count: int
    cpu_percent: float
    memory_percent: float
    disk_usage: float

class SovereignAgent:
    """Full system control agent with AI model management capabilities"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.running = False
        self.command_history = []
        self.max_history = 100
        
        # AI model management
        self.device = self._detect_device()
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        
        # System control state
        self.processes = {}
        self.files_created = []
        self.environment_vars = {}
        
        log.info(f"SovereignAgent initialized on device: {self.device}")

    def _detect_device(self) -> torch.device:
        """Detect the best available device for AI models"""
        if torch.cuda.is_available():
            log.info("Using CUDA GPU")
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            log.info("Using Apple MPS GPU")
            return torch.device("mps")
        else:
            log.info("Using CPU")
            return torch.device("cpu")

    async def load_ai_model(self, model_name: str = "gpt2", precision: str = "fp32"):
        """Load AI model with specified precision"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            log.info(f"Loading model {model_name} on device {self.device} with precision {precision}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Set precision
            dtype = torch.float32
            if precision == "fp16" and self.device.type == "cuda":
                dtype = torch.float16
            elif precision == "bf16" and self.device.type == "cuda" and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            elif precision == "bf16" and self.device.type == "mps":
                dtype = torch.bfloat16
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map="auto" if self.device.type == "cuda" else None
            )
            self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True
            
            log.info(f"Model loaded successfully with dtype {dtype}")
            
        except Exception as e:
            log.error(f"Failed to load model {model_name}: {e}")
            self.model_loaded = False

    async def generate_response(self, prompt: str, max_length: int = 100) -> str:
        """Generate AI response using loaded model"""
        if not self.model_loaded:
            return "Error: AI model not loaded"
        
        try:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            
            # Run generation in thread pool to avoid blocking
            outputs = await asyncio.to_thread(
                self.model.generate, 
                input_ids, 
                max_length=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            log.info(f"Generated response for prompt: '{prompt[:50]}...'")
            return text
            
        except Exception as e:
            log.error(f"Failed to generate response: {e}")
            return f"Error generating response: {str(e)}"

    async def execute_command(self, cmd: str) -> Dict[str, Any]:
        """Execute system command and return detailed result"""
        log.info(f"Executing system command: {cmd}")
        
        try:
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            result = {
                "command": cmd,
                "return_code": process.returncode,
                "stdout": stdout.decode('utf-8', errors='ignore'),
                "stderr": stderr.decode('utf-8', errors='ignore'),
                "success": process.returncode == 0
            }
            
            # Log command history
            self.command_history.append({
                **result,
                "timestamp": asyncio.get_event_loop().time()
            })
            
            if len(self.command_history) > self.max_history:
                self.command_history.pop(0)
            
            log.info(f"Command completed with return code: {result['return_code']}")
            return result
            
        except Exception as e:
            log.error(f"Failed to execute system command: {e}")
            return {
                "command": cmd,
                "return_code": -1,
                "stdout": "",
                "stderr": str(e),
                "success": False
            }

    async def get_system_info(self) -> SystemInfo:
        """Get comprehensive system information"""
        import psutil
        
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/') if os.name != 'nt' else psutil.disk_usage('C:\\')
            
            return SystemInfo(
                platform=sys.platform,
                python_version=sys.version,
                cpu_count=psutil.cpu_count(),
                cpu_percent=psutil.cpu_percent(),
                memory_percent=memory.percent,
                disk_usage=disk.percent
            )
        except Exception as e:
            log.error(f"Failed to get system info: {e}")
            return SystemInfo("unknown", "unknown", 0, 0.0, 0.0, 0.0)

    async def get_process_list(self) -> List[Dict[str, Any]]:
        """Get list of running processes"""
        import psutil
        
        processes = []
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except Exception as e:
            log.error(f"Failed to get process list: {e}")
        
        return processes

    async def kill_process(self, pid: int) -> bool:
        """Kill a process by PID"""
        import psutil
        
        try:
            process = psutil.Process(pid)
            process.terminate()
            log.info(f"Terminated process {pid}")
            return True
        except psutil.NoSuchProcess:
            log.warning(f"Process {pid} not found")
            return False
        except psutil.AccessDenied:
            log.error(f"Access denied to process {pid}")
            return False

    async def open_file(self, filepath: str) -> bool:
        """Open a file with default application"""
        try:
            if os.name == "nt":  # Windows
                os.startfile(filepath)
            else:  # Unix-like
                await self.execute_command(f"xdg-open '{filepath}'")
            
            log.info(f"Opened file: {filepath}")
            return True
        except Exception as e:
            log.error(f"Failed to open file {filepath}: {e}")
            return False

    async def create_file(self, filepath: str, content: str) -> bool:
        """Create a file with content"""
        try:
            with open(filepath, 'w') as f:
                f.write(content)
            self.files_created.append(filepath)
            log.info(f"Created file: {filepath}")
            return True
        except Exception as e:
            log.error(f"Failed to create file {filepath}: {e}")
            return False

    async def read_file(self, filepath: str) -> Optional[str]:
        """Read file content"""
        try:
            with open(filepath, 'r') as f:
                return f.read()
        except Exception as e:
            log.error(f"Failed to read file {filepath}: {e}")
            return None

    async def delete_file(self, filepath: str) -> bool:
        """Delete a file"""
        try:
            os.remove(filepath)
            log.info(f"Deleted file: {filepath}")
            return True
        except Exception as e:
            log.error(f"Failed to delete file {filepath}: {e}")
            return False

    async def list_directory(self, path: str) -> List[str]:
        """List directory contents"""
        try:
            return os.listdir(path)
        except Exception as e:
            log.error(f"Failed to list directory {path}: {e}")
            return []

    async def create_directory(self, path: str) -> bool:
        """Create a directory"""
        try:
            os.makedirs(path, exist_ok=True)
            log.info(f"Created directory: {path}")
            return True
        except Exception as e:
            log.error(f"Failed to create directory {path}: {e}")
            return False

    def get_plugin_api(self):
        """Get API for plugins to interact with the agent"""
        from plugins.sdk import AIManagerAPI
        return AIManagerAPI(self)

    def get_command_history(self) -> List[Dict[str, Any]]:
        """Get command execution history"""
        return self.command_history

    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            "running": self.running,
            "device": str(self.device),
            "model_loaded": self.model_loaded,
            "commands_executed": len(self.command_history),
            "files_created": len(self.files_created)
        }

    async def start(self):
        """Start the sovereign agent"""
        self.running = True
        log.info("SovereignAgent started")

    async def stop(self):
        """Stop the sovereign agent"""
        self.running = False
        log.info("SovereignAgent stopped")

    def generate(self, prompt: str) -> str:
        """Synchronous wrapper for generate_response"""
        return asyncio.run(self.generate_response(prompt))