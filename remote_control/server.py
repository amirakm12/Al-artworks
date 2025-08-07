import asyncio
import json
import websockets
import logging
import subprocess
import os
import sys
from typing import Dict, Any, Set
from dataclasses import dataclass

log = logging.getLogger("RemoteControl")

@dataclass
class CommandResult:
    stdout: str
    stderr: str
    return_code: int
    success: bool

class SovereignAgent:
    """Full system control agent with command execution capabilities"""
    
    def __init__(self):
        self.running = False
        self.command_history = []
        self.max_history = 100

    async def execute_command(self, command: str) -> CommandResult:
        """Execute system command and return result"""
        log.info(f"Executing command: {command}")
        
        try:
            # Create subprocess
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Wait for completion
            stdout, stderr = await process.communicate()
            
            result = CommandResult(
                stdout=stdout.decode('utf-8', errors='ignore'),
                stderr=stderr.decode('utf-8', errors='ignore'),
                return_code=process.returncode,
                success=process.returncode == 0
            )
            
            # Add to history
            self.command_history.append({
                "command": command,
                "timestamp": asyncio.get_event_loop().time(),
                "result": result
            })
            
            # Keep history size manageable
            if len(self.command_history) > self.max_history:
                self.command_history.pop(0)
            
            log.info(f"Command completed with return code: {result.return_code}")
            return result
            
        except Exception as e:
            log.error(f"Command execution failed: {e}")
            return CommandResult(
                stdout="",
                stderr=str(e),
                return_code=-1,
                success=False
            )

    async def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        import psutil
        
        return {
            "platform": sys.platform,
            "python_version": sys.version,
            "cpu_count": psutil.cpu_count(),
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:\\').percent
        }

    async def get_process_list(self) -> list:
        """Get list of running processes"""
        import psutil
        
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
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
                subprocess.run(["xdg-open", filepath], check=True)
            
            log.info(f"Opened file: {filepath}")
            return True
        except Exception as e:
            log.error(f"Failed to open file {filepath}: {e}")
            return False

    def get_command_history(self) -> list:
        """Get command execution history"""
        return self.command_history

# Global agent instance
agent = SovereignAgent()

# Connected clients
clients: Set[websockets.WebSocketServerProtocol] = set()

async def broadcast_message(message: Dict[str, Any]):
    """Broadcast message to all connected clients"""
    if not clients:
        return
    
    message_json = json.dumps(message)
    disconnected_clients = set()
    
    for client in clients:
        try:
            await client.send(message_json)
        except websockets.exceptions.ConnectionClosed:
            disconnected_clients.add(client)
        except Exception as e:
            log.error(f"Failed to send message to client: {e}")
            disconnected_clients.add(client)
    
    # Remove disconnected clients
    clients.difference_update(disconnected_clients)

async def handle_client(websocket, path):
    """Handle individual client connection"""
    clients.add(websocket)
    client_addr = websocket.remote_address
    log.info(f"Client connected: {client_addr}")
    
    try:
        # Send welcome message
        welcome_msg = {
            "type": "welcome",
            "message": "Connected to ChatGPT+ Remote Control",
            "timestamp": asyncio.get_event_loop().time()
        }
        await websocket.send(json.dumps(welcome_msg))
        
        # Send system info
        system_info = await agent.get_system_info()
        info_msg = {
            "type": "system_info",
            "data": system_info
        }
        await websocket.send(json.dumps(info_msg))
        
        # Handle incoming messages
        async for message in websocket:
            try:
                data = json.loads(message)
                await handle_command(websocket, data)
            except json.JSONDecodeError:
                error_msg = {
                    "type": "error",
                    "message": "Invalid JSON format"
                }
                await websocket.send(json.dumps(error_msg))
            except Exception as e:
                log.error(f"Error handling message from {client_addr}: {e}")
                error_msg = {
                    "type": "error",
                    "message": f"Internal error: {str(e)}"
                }
                await websocket.send(json.dumps(error_msg))
                
    except websockets.exceptions.ConnectionClosed:
        log.info(f"Client disconnected: {client_addr}")
    except Exception as e:
        log.error(f"Unexpected error with client {client_addr}: {e}")
    finally:
        clients.remove(websocket)
        log.info(f"Client {client_addr} removed from active connections")

async def handle_command(websocket, data: Dict[str, Any]):
    """Handle client command"""
    command_type = data.get("type", "unknown")
    
    if command_type == "execute_command":
        command = data.get("command")
        if command:
            result = await agent.execute_command(command)
            response = {
                "type": "command_result",
                "command": command,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.return_code,
                "success": result.success
            }
            await websocket.send(json.dumps(response))
        else:
            error_msg = {
                "type": "error",
                "message": "No command provided"
            }
            await websocket.send(json.dumps(error_msg))
    
    elif command_type == "get_system_info":
        system_info = await agent.get_system_info()
        response = {
            "type": "system_info",
            "data": system_info
        }
        await websocket.send(json.dumps(response))
    
    elif command_type == "get_process_list":
        processes = await agent.get_process_list()
        response = {
            "type": "process_list",
            "data": processes
        }
        await websocket.send(json.dumps(response))
    
    elif command_type == "kill_process":
        pid = data.get("pid")
        if pid is not None:
            success = await agent.kill_process(pid)
            response = {
                "type": "kill_result",
                "pid": pid,
                "success": success
            }
            await websocket.send(json.dumps(response))
        else:
            error_msg = {
                "type": "error",
                "message": "No PID provided"
            }
            await websocket.send(json.dumps(error_msg))
    
    elif command_type == "open_file":
        filepath = data.get("filepath")
        if filepath:
            success = await agent.open_file(filepath)
            response = {
                "type": "open_file_result",
                "filepath": filepath,
                "success": success
            }
            await websocket.send(json.dumps(response))
        else:
            error_msg = {
                "type": "error",
                "message": "No filepath provided"
            }
            await websocket.send(json.dumps(error_msg))
    
    elif command_type == "get_command_history":
        history = agent.get_command_history()
        response = {
            "type": "command_history",
            "data": history
        }
        await websocket.send(json.dumps(response))
    
    else:
        error_msg = {
            "type": "error",
            "message": f"Unknown command type: {command_type}"
        }
        await websocket.send(json.dumps(error_msg))

async def periodic_broadcast():
    """Periodically broadcast system stats to all clients"""
    while True:
        try:
            if clients:
                system_info = await agent.get_system_info()
                broadcast_msg = {
                    "type": "system_update",
                    "data": system_info,
                    "timestamp": asyncio.get_event_loop().time()
                }
                await broadcast_message(broadcast_msg)
        except Exception as e:
            log.error(f"Error in periodic broadcast: {e}")
        
        await asyncio.sleep(5)  # Update every 5 seconds

async def main():
    """Main server function"""
    host = "0.0.0.0"
    port = 8765
    
    log.info(f"Starting remote control WebSocket server on ws://{host}:{port}")
    
    # Start periodic broadcast task
    broadcast_task = asyncio.create_task(periodic_broadcast())
    
    # Start WebSocket server
    server = await websockets.serve(handle_client, host, port)
    
    log.info(f"Remote control server started successfully")
    log.info(f"Listening on ws://{host}:{port}")
    log.info(f"Active connections: {len(clients)}")
    
    try:
        await server.wait_closed()
    except KeyboardInterrupt:
        log.info("Shutting down server...")
    finally:
        broadcast_task.cancel()
        log.info("Server shutdown complete")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Server interrupted by user")
    except Exception as e:
        log.error(f"Server error: {e}")