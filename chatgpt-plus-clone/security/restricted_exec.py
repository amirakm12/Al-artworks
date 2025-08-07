"""
Security Framework - Restricted Execution and Sandboxing
Provides secure code execution with restricted access and monitoring
"""

import builtins
import sys
import os
import logging
import time
import threading
import json
import hashlib
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
import tempfile
import subprocess

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Safe builtins that are allowed in restricted execution
SAFE_BUILTINS = {
    'abs', 'all', 'any', 'bin', 'bool', 'bytes', 'chr', 'complex', 'dict',
    'divmod', 'enumerate', 'filter', 'float', 'format', 'frozenset', 'hash',
    'hex', 'int', 'isinstance', 'issubclass', 'iter', 'len', 'list', 'map',
    'max', 'min', 'next', 'oct', 'ord', 'pow', 'range', 'repr', 'reversed',
    'round', 'set', 'slice', 'sorted', 'str', 'sum', 'tuple', 'zip'
}

# Dangerous builtins that should be blocked
DANGEROUS_BUILTINS = {
    'eval', 'exec', 'compile', 'open', 'file', 'input', 'raw_input',
    'reload', 'delattr', 'setattr', 'getattr', 'hasattr', 'vars',
    'locals', 'globals', '__import__', 'apply', 'buffer', 'callable',
    'coerce', 'intern', 'reduce', 'xrange'
}

class SecurityViolation(Exception):
    """Exception raised when security violations are detected"""
    pass

class RestrictedExecutor:
    """Secure code executor with restricted access"""
    
    def __init__(self, security_level: str = "medium"):
        self.security_level = security_level
        self.logger = logging.getLogger(__name__)
        self.execution_history = []
        self.violation_count = 0
        self.max_violations = 10
        
        # Security monitoring
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Allowed modules (whitelist approach)
        self.allowed_modules = {
            'math', 'random', 'datetime', 'collections', 'itertools',
            'functools', 'operator', 're', 'string', 'json'
        }
        
        # Blocked modules (blacklist approach)
        self.blocked_modules = {
            'os', 'sys', 'subprocess', 'shutil', 'tempfile', 'pathlib',
            'socket', 'urllib', 'http', 'ftplib', 'smtplib', 'poplib',
            'imaplib', 'telnetlib', 'pickle', 'marshal', 'ctypes',
            'multiprocessing', 'threading', 'asyncio', 'concurrent'
        }
        
        self.logger.info(f"RestrictedExecutor initialized with security level: {security_level}")
    
    def restricted_exec(self, code_str: str, timeout: int = 30) -> Dict[str, Any]:
        """
        Execute code with restricted access
        Returns execution result with security information
        """
        start_time = time.time()
        execution_id = f"restricted_{int(start_time * 1000)}"
        
        result = {
            'success': False,
            'output': '',
            'error': '',
            'execution_time': 0.0,
            'execution_id': execution_id,
            'security_violations': [],
            'code_hash': hashlib.sha256(code_str.encode()).hexdigest()[:8]
        }
        
        try:
            # Security checks
            violations = self._check_code_security(code_str)
            if violations:
                result['security_violations'] = violations
                result['error'] = f"Security violations detected: {violations}"
                self.violation_count += 1
                raise SecurityViolation(f"Code contains security violations: {violations}")
            
            # Create restricted globals
            restricted_globals = self._create_restricted_globals()
            restricted_locals = {}
            
            # Execute with timeout
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Execution timed out")
            
            # Set up timeout (Unix-like systems)
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout)
            
            try:
                # Execute the code
                exec(code_str, restricted_globals, restricted_locals)
                
                # Capture output
                if 'result' in restricted_locals:
                    result['output'] = str(restricted_locals['result'])
                else:
                    result['output'] = "Code executed successfully"
                
                result['success'] = True
                
            finally:
                # Clear timeout
                if hasattr(signal, 'SIGALRM'):
                    signal.alarm(0)
            
        except SecurityViolation as e:
            result['error'] = str(e)
            self.logger.warning(f"Security violation in execution {execution_id}: {e}")
        except TimeoutError as e:
            result['error'] = f"Execution timed out after {timeout} seconds"
            self.logger.warning(f"Execution timeout in {execution_id}")
        except Exception as e:
            result['error'] = f"Execution failed: {str(e)}"
            self.logger.error(f"Execution error in {execution_id}: {e}")
        
        result['execution_time'] = time.time() - start_time
        self._add_to_history(result)
        
        return result
    
    def _check_code_security(self, code_str: str) -> List[str]:
        """Check code for security violations"""
        violations = []
        
        # Check for dangerous builtins
        for dangerous in DANGEROUS_BUILTINS:
            if dangerous in code_str:
                violations.append(f"Dangerous builtin: {dangerous}")
        
        # Check for import statements
        import_lines = [line.strip() for line in code_str.split('\n') if line.strip().startswith('import') or line.strip().startswith('from')]
        for import_line in import_lines:
            if 'import' in import_line:
                # Extract module name
                if import_line.startswith('from'):
                    # from module import ...
                    parts = import_line.split()
                    if len(parts) >= 2:
                        module = parts[1]
                        if module in self.blocked_modules:
                            violations.append(f"Blocked module import: {module}")
                else:
                    # import module
                    parts = import_line.split()
                    if len(parts) >= 2:
                        module = parts[1].split('.')[0]  # Handle dotted imports
                        if module in self.blocked_modules:
                            violations.append(f"Blocked module import: {module}")
        
        # Check for file operations
        file_ops = ['open(', 'file(', 'read(', 'write(', 'save(']
        for op in file_ops:
            if op in code_str:
                violations.append(f"File operation detected: {op}")
        
        # Check for network operations
        network_ops = ['socket', 'urllib', 'http', 'ftp', 'smtp', 'pop', 'imap', 'telnet']
        for op in network_ops:
            if op in code_str:
                violations.append(f"Network operation detected: {op}")
        
        # Check for system operations
        system_ops = ['os.', 'sys.', 'subprocess', 'exec', 'eval', 'compile']
        for op in system_ops:
            if op in code_str:
                violations.append(f"System operation detected: {op}")
        
        return violations
    
    def _create_restricted_globals(self) -> Dict[str, Any]:
        """Create restricted globals environment"""
        restricted_globals = {}
        
        # Add safe builtins
        for safe_builtin in SAFE_BUILTINS:
            if hasattr(builtins, safe_builtin):
                restricted_globals[safe_builtin] = getattr(builtins, safe_builtin)
        
        # Add safe modules
        for module_name in self.allowed_modules:
            try:
                module = __import__(module_name)
                restricted_globals[module_name] = module
            except ImportError:
                pass
        
        # Add custom safe functions
        restricted_globals['print'] = self._safe_print
        restricted_globals['len'] = len
        restricted_globals['range'] = range
        restricted_globals['list'] = list
        restricted_globals['dict'] = dict
        restricted_globals['set'] = set
        restricted_globals['tuple'] = tuple
        
        return restricted_globals
    
    def _safe_print(self, *args, **kwargs):
        """Safe print function that logs output"""
        output = ' '.join(str(arg) for arg in args)
        self.logger.info(f"Restricted print: {output}")
        return output
    
    def sandboxed_exec(self, code_str: str, timeout: int = 30) -> Dict[str, Any]:
        """
        Execute code in a sandboxed environment using subprocess
        More secure but slower than restricted_exec
        """
        start_time = time.time()
        execution_id = f"sandbox_{int(start_time * 1000)}"
        
        result = {
            'success': False,
            'output': '',
            'error': '',
            'execution_time': 0.0,
            'execution_id': execution_id,
            'security_violations': [],
            'code_hash': hashlib.sha256(code_str.encode()).hexdigest()[:8]
        }
        
        try:
            # Create temporary file for code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmpfile:
                # Wrap code in sandbox
                sandboxed_code = self._wrap_in_sandbox(code_str)
                tmpfile.write(sandboxed_code)
                tmp_path = tmpfile.name
            
            try:
                # Execute in subprocess with restrictions
                process = subprocess.run(
                    [sys.executable, tmp_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=timeout,
                    env=self._create_restricted_env()
                )
                
                result['output'] = process.stdout
                result['error'] = process.stderr
                result['return_code'] = process.returncode
                result['success'] = process.returncode == 0
                
            finally:
                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                except Exception as e:
                    self.logger.warning(f"Failed to clean up temp file: {e}")
        
        except subprocess.TimeoutExpired:
            result['error'] = f"Sandbox execution timed out after {timeout} seconds"
        except Exception as e:
            result['error'] = f"Sandbox execution failed: {str(e)}"
        
        result['execution_time'] = time.time() - start_time
        self._add_to_history(result)
        
        return result
    
    def _wrap_in_sandbox(self, code_str: str) -> str:
        """Wrap code in sandbox environment"""
        sandbox_wrapper = f'''
import sys
import os
import signal

# Set up signal handlers for timeout
def timeout_handler(signum, frame):
    print("Execution timed out", file=sys.stderr)
    sys.exit(1)

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(30)  # 30 second timeout

# Restrict imports
original_import = __builtins__.__import__

def restricted_import(name, *args, **kwargs):
    blocked_modules = {{'os', 'sys', 'subprocess', 'shutil', 'tempfile', 'pathlib',
                       'socket', 'urllib', 'http', 'ftplib', 'smtplib', 'poplib',
                       'imaplib', 'telnetlib', 'pickle', 'marshal', 'ctypes',
                       'multiprocessing', 'threading', 'asyncio', 'concurrent'}}
    
    if name in blocked_modules:
        raise ImportError(f"Module {{name}} is not allowed in sandbox")
    
    return original_import(name, *args, **kwargs)

__builtins__.__import__ = restricted_import

# Safe modules
allowed_modules = {{'math', 'random', 'datetime', 'collections', 'itertools',
                   'functools', 'operator', 're', 'string', 'json'}}

# Execute user code
try:
{chr(10).join('    ' + line for line in code_str.split(chr(10)))}
    print("Code executed successfully")
except Exception as e:
    print(f"Error: {{e}}", file=sys.stderr)
    sys.exit(1)
'''
        return sandbox_wrapper
    
    def _create_restricted_env(self) -> Dict[str, str]:
        """Create restricted environment variables"""
        env = os.environ.copy()
        
        # Remove dangerous environment variables
        dangerous_vars = ['PATH', 'PYTHONPATH', 'LD_LIBRARY_PATH', 'LD_PRELOAD']
        for var in dangerous_vars:
            if var in env:
                del env[var]
        
        # Set restrictive environment
        env['PYTHONPATH'] = ''
        env['PATH'] = '/usr/bin:/bin'
        
        return env
    
    def start_monitoring(self):
        """Start security monitoring"""
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_security, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Security monitoring started")
    
    def stop_monitoring(self):
        """Stop security monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("Security monitoring stopped")
    
    def _monitor_security(self):
        """Monitor security violations"""
        while self.monitoring_active:
            if self.violation_count > self.max_violations:
                self.logger.critical(f"Too many security violations: {self.violation_count}")
                # Could implement automatic blocking here
            time.sleep(1)
    
    def get_security_report(self) -> Dict[str, Any]:
        """Get security monitoring report"""
        return {
            'violation_count': self.violation_count,
            'max_violations': self.max_violations,
            'monitoring_active': self.monitoring_active,
            'execution_history_count': len(self.execution_history),
            'security_level': self.security_level
        }
    
    def _add_to_history(self, result: Dict[str, Any]):
        """Add execution result to history"""
        self.execution_history.append(result)
        
        # Limit history size
        if len(self.execution_history) > 100:
            self.execution_history.pop(0)
    
    def clear_history(self):
        """Clear execution history"""
        self.execution_history.clear()
        self.violation_count = 0
        self.logger.info("Execution history and violation count cleared")

# Convenience functions
def restricted_exec(code_str: str, timeout: int = 30) -> Dict[str, Any]:
    """Execute code with restricted access"""
    executor = RestrictedExecutor()
    return executor.restricted_exec(code_str, timeout)

def sandboxed_exec(code_str: str, timeout: int = 30) -> Dict[str, Any]:
    """Execute code in sandboxed environment"""
    executor = RestrictedExecutor()
    return executor.sandboxed_exec(code_str, timeout)

def check_code_security(code_str: str) -> List[str]:
    """Check code for security violations"""
    executor = RestrictedExecutor()
    return executor._check_code_security(code_str)

if __name__ == "__main__":
    # Test the security framework
    print("🧪 Testing Security Framework...")
    
    executor = RestrictedExecutor()
    
    # Test safe code
    safe_code = """
result = 0
for i in range(10):
    result += i * i
print(f"Result: {result}")
"""
    
    print("Testing safe code execution...")
    result = executor.restricted_exec(safe_code)
    print(f"✅ Safe code result: {result['success']}")
    print(f"Output: {result['output']}")
    
    # Test dangerous code
    dangerous_code = """
import os
os.system('echo dangerous')
"""
    
    print("\nTesting dangerous code detection...")
    result = executor.restricted_exec(dangerous_code)
    print(f"❌ Dangerous code blocked: {not result['success']}")
    print(f"Violations: {result['security_violations']}")
    
    # Test sandboxed execution
    print("\nTesting sandboxed execution...")
    result = executor.sandboxed_exec(safe_code)
    print(f"✅ Sandboxed execution: {result['success']}")
    
    # Get security report
    report = executor.get_security_report()
    print(f"\nSecurity Report: {report}")