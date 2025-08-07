"""
Code Executor - No Sandbox Version
Unrestricted code execution with full system access
WARNING: This version provides NO security isolation
"""

import subprocess
import sys
import tempfile
import os
import logging
import threading
import time
import signal
import psutil
from typing import Dict, Any, Optional, List
from pathlib import Path
import json

class CodeExecutor:
    """Unrestricted code executor with full system access"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.temp_dir = Path(tempfile.gettempdir()) / "chatgpt_code_exec"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Execution history
        self.execution_history = []
        self.max_history = 100
        
        # Performance monitoring
        self.performance_stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'average_execution_time': 0.0
        }
        
        self.logger.warning("⚠️  NO-SANDBOX MODE: Code execution has full system access!")
    
    def execute_python(self, code: str, timeout: int = 30) -> Dict[str, Any]:
        """
        Execute Python code with full system access
        WARNING: No sandbox - code can access entire system
        """
        start_time = time.time()
        execution_id = f"exec_{int(start_time * 1000)}"
        
        result = {
            'success': False,
            'output': '',
            'error': '',
            'execution_time': 0.0,
            'execution_id': execution_id,
            'code': code
        }
        
        try:
            # Create temporary file for code
            temp_file = self.temp_dir / f"{execution_id}.py"
            
            # Write code to file
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(code)
            
            # Execute with subprocess for output capture
            process = subprocess.run(
                [sys.executable, str(temp_file)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                cwd=str(self.temp_dir)  # Set working directory
            )
            
            # Capture output
            result['output'] = process.stdout
            result['error'] = process.stderr
            result['return_code'] = process.returncode
            result['success'] = process.returncode == 0
            
        except subprocess.TimeoutExpired:
            result['error'] = f"Execution timed out after {timeout} seconds"
            result['success'] = False
        except Exception as e:
            result['error'] = f"Execution failed: {str(e)}"
            result['success'] = False
        finally:
            # Clean up temp file
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except Exception as e:
                self.logger.warning(f"Failed to clean up temp file: {e}")
        
        # Calculate execution time
        result['execution_time'] = time.time() - start_time
        
        # Update performance stats
        self._update_performance_stats(result)
        
        # Add to history
        self._add_to_history(result)
        
        self.logger.info(f"Code execution completed: {execution_id} (success: {result['success']}, time: {result['execution_time']:.2f}s)")
        
        return result
    
    def execute_python_direct(self, code: str) -> Dict[str, Any]:
        """
        Execute Python code directly in current process
        WARNING: This runs in the same process as the application
        """
        start_time = time.time()
        execution_id = f"direct_{int(start_time * 1000)}"
        
        result = {
            'success': False,
            'output': '',
            'error': '',
            'execution_time': 0.0,
            'execution_id': execution_id,
            'code': code
        }
        
        try:
            # Capture stdout/stderr
            import io
            import contextlib
            
            output_buffer = io.StringIO()
            error_buffer = io.StringIO()
            
            with contextlib.redirect_stdout(output_buffer), contextlib.redirect_stderr(error_buffer):
                # Execute code directly
                exec_globals = {
                    '__builtins__': __builtins__,
                    '__name__': '__main__',
                    '__file__': f'<exec_{execution_id}>',
                    'sys': sys,
                    'os': os,
                    'json': json,
                    'time': time,
                    'threading': threading,
                    'pathlib': Path,
                    'psutil': psutil
                }
                
                exec(code, exec_globals)
            
            result['output'] = output_buffer.getvalue()
            result['error'] = error_buffer.getvalue()
            result['success'] = True
            
        except Exception as e:
            result['error'] = f"Direct execution failed: {str(e)}"
            result['success'] = False
        
        result['execution_time'] = time.time() - start_time
        self._update_performance_stats(result)
        self._add_to_history(result)
        
        self.logger.info(f"Direct code execution completed: {execution_id} (success: {result['success']}, time: {result['execution_time']:.2f}s)")
        
        return result
    
    def execute_shell_command(self, command: str, timeout: int = 30) -> Dict[str, Any]:
        """
        Execute shell commands with full system access
        WARNING: No sandbox - commands can access entire system
        """
        start_time = time.time()
        execution_id = f"shell_{int(start_time * 1000)}"
        
        result = {
            'success': False,
            'output': '',
            'error': '',
            'execution_time': 0.0,
            'execution_id': execution_id,
            'command': command
        }
        
        try:
            # Execute shell command
            process = subprocess.run(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout
            )
            
            result['output'] = process.stdout
            result['error'] = process.stderr
            result['return_code'] = process.returncode
            result['success'] = process.returncode == 0
            
        except subprocess.TimeoutExpired:
            result['error'] = f"Shell command timed out after {timeout} seconds"
            result['success'] = False
        except Exception as e:
            result['error'] = f"Shell command failed: {str(e)}"
            result['success'] = False
        
        result['execution_time'] = time.time() - start_time
        self._update_performance_stats(result)
        self._add_to_history(result)
        
        self.logger.info(f"Shell command completed: {execution_id} (success: {result['success']}, time: {result['execution_time']:.2f}s)")
        
        return result
    
    def execute_with_dependencies(self, code: str, dependencies: List[str] = None, timeout: int = 60) -> Dict[str, Any]:
        """
        Execute code with optional dependency installation
        WARNING: This can install packages and modify system
        """
        start_time = time.time()
        execution_id = f"dep_{int(start_time * 1000)}"
        
        result = {
            'success': False,
            'output': '',
            'error': '',
            'execution_time': 0.0,
            'execution_id': execution_id,
            'code': code,
            'dependencies': dependencies or []
        }
        
        try:
            # Install dependencies if provided
            if dependencies:
                self.logger.info(f"Installing dependencies: {dependencies}")
                for dep in dependencies:
                    install_result = self.execute_shell_command(f"pip install {dep}", timeout=timeout)
                    if not install_result['success']:
                        result['error'] = f"Failed to install dependency {dep}: {install_result['error']}"
                        return result
            
            # Execute the code
            exec_result = self.execute_python(code, timeout=timeout)
            result.update(exec_result)
            
        except Exception as e:
            result['error'] = f"Dependency execution failed: {str(e)}"
            result['success'] = False
        
        result['execution_time'] = time.time() - start_time
        self._update_performance_stats(result)
        self._add_to_history(result)
        
        return result
    
    def execute_interactive(self, code: str, inputs: List[str] = None) -> Dict[str, Any]:
        """
        Execute code with interactive input simulation
        WARNING: No sandbox - code can access entire system
        """
        start_time = time.time()
        execution_id = f"interactive_{int(start_time * 1000)}"
        
        result = {
            'success': False,
            'output': '',
            'error': '',
            'execution_time': 0.0,
            'execution_id': execution_id,
            'code': code,
            'inputs': inputs or []
        }
        
        try:
            # Create temporary file
            temp_file = self.temp_dir / f"{execution_id}.py"
            
            # Write code to file
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(code)
            
            # Prepare input string
            input_data = '\n'.join(inputs or []) + '\n'
            
            # Execute with input
            process = subprocess.run(
                [sys.executable, str(temp_file)],
                input=input_data,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30
            )
            
            result['output'] = process.stdout
            result['error'] = process.stderr
            result['return_code'] = process.returncode
            result['success'] = process.returncode == 0
            
        except subprocess.TimeoutExpired:
            result['error'] = "Interactive execution timed out"
            result['success'] = False
        except Exception as e:
            result['error'] = f"Interactive execution failed: {str(e)}"
            result['success'] = False
        finally:
            # Clean up
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except Exception:
                pass
        
        result['execution_time'] = time.time() - start_time
        self._update_performance_stats(result)
        self._add_to_history(result)
        
        return result
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information for execution context"""
        return {
            'python_version': sys.version,
            'platform': sys.platform,
            'architecture': sys.maxsize > 2**32 and '64bit' or '32bit',
            'executable': sys.executable,
            'cwd': os.getcwd(),
            'temp_dir': str(self.temp_dir),
            'cpu_count': os.cpu_count(),
            'memory': psutil.virtual_memory()._asdict() if psutil else {},
            'disk': psutil.disk_usage('/')._asdict() if psutil else {},
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return self.performance_stats.copy()
    
    def get_execution_history(self, limit: int = None) -> List[Dict[str, Any]]:
        """Get execution history"""
        if limit is None:
            return self.execution_history.copy()
        return self.execution_history[-limit:]
    
    def clear_history(self):
        """Clear execution history"""
        self.execution_history.clear()
        self.logger.info("Execution history cleared")
    
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        try:
            for file in self.temp_dir.glob("*.py"):
                file.unlink()
            self.logger.info("Temporary files cleaned up")
        except Exception as e:
            self.logger.warning(f"Failed to clean up temp files: {e}")
    
    def _update_performance_stats(self, result: Dict[str, Any]):
        """Update performance statistics"""
        self.performance_stats['total_executions'] += 1
        
        if result['success']:
            self.performance_stats['successful_executions'] += 1
        else:
            self.performance_stats['failed_executions'] += 1
        
        # Update average execution time
        total_time = self.performance_stats['average_execution_time'] * (self.performance_stats['total_executions'] - 1)
        total_time += result['execution_time']
        self.performance_stats['average_execution_time'] = total_time / self.performance_stats['total_executions']
    
    def _add_to_history(self, result: Dict[str, Any]):
        """Add result to execution history"""
        self.execution_history.append(result)
        
        # Limit history size
        if len(self.execution_history) > self.max_history:
            self.execution_history.pop(0)
    
    def __del__(self):
        """Cleanup on destruction"""
        try:
            self.cleanup_temp_files()
        except Exception:
            pass

# Convenience functions
def execute_python_code(code: str, timeout: int = 30) -> Dict[str, Any]:
    """Execute Python code with full system access"""
    executor = CodeExecutor()
    return executor.execute_python(code, timeout)

def execute_shell_command(command: str, timeout: int = 30) -> Dict[str, Any]:
    """Execute shell command with full system access"""
    executor = CodeExecutor()
    return executor.execute_shell_command(command, timeout)

def execute_with_dependencies(code: str, dependencies: List[str] = None, timeout: int = 60) -> Dict[str, Any]:
    """Execute code with optional dependency installation"""
    executor = CodeExecutor()
    return executor.execute_with_dependencies(code, dependencies, timeout)

if __name__ == "__main__":
    # Test the no-sandbox code executor
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testing No-Sandbox Code Executor...")
    print("⚠️  WARNING: This version provides NO security isolation!")
    
    executor = CodeExecutor()
    
    # Test basic Python execution
    test_code = """
import os
import sys
print("Hello from unrestricted code execution!")
print(f"Current directory: {os.getcwd()}")
print(f"Python version: {sys.version}")
print(f"System platform: {sys.platform}")
"""
    
    result = executor.execute_python(test_code)
    print(f"✅ Python execution result: {result['success']}")
    print(f"Output: {result['output']}")
    if result['error']:
        print(f"Error: {result['error']}")
    
    # Test shell command
    shell_result = executor.execute_shell_command("echo 'Hello from shell!'")
    print(f"✅ Shell execution result: {shell_result['success']}")
    print(f"Output: {shell_result['output']}")
    
    # Test system info
    system_info = executor.get_system_info()
    print(f"✅ System info: {system_info}")
    
    # Test performance stats
    stats = executor.get_performance_stats()
    print(f"✅ Performance stats: {stats}")