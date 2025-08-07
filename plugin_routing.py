import asyncio
import logging
import time
import uuid
from enum import Enum, auto
from typing import Dict, Any, Optional, List, Set, Callable
from dataclasses import dataclass, field
from collections import defaultdict

log = logging.getLogger("PluginRouter")

class Role(Enum):
    ADMIN = auto()
    USER = auto()
    GUEST = auto()
    SYSTEM = auto()
    PLUGIN = auto()

class Permission(Enum):
    EXEC_COMMAND = auto()
    READ_DATA = auto()
    WRITE_DATA = auto()
    MODIFY_SETTINGS = auto()
    ACCESS_AI = auto()
    ACCESS_GPU = auto()
    ACCESS_FILESYSTEM = auto()
    ACCESS_NETWORK = auto()
    KILL_PROCESSES = auto()
    INSTALL_PLUGINS = auto()

@dataclass
class SecurityContext:
    """Security context for command execution"""
    user_id: str
    session_id: str
    roles: Set[Role] = field(default_factory=set)
    permissions: Set[Permission] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_time: float = field(default_factory=time.time)
    expires_at: Optional[float] = None

    def has_permission(self, permission: Permission) -> bool:
        """Check if context has specific permission"""
        if self.expires_at and time.time() > self.expires_at:
            return False
        return permission in self.permissions

    def has_role(self, role: Role) -> bool:
        """Check if context has specific role"""
        if self.expires_at and time.time() > self.expires_at:
            return False
        return role in self.roles

    def is_valid(self) -> bool:
        """Check if context is still valid"""
        if self.expires_at and time.time() > self.expires_at:
            return False
        return True

@dataclass
class CommandEvent:
    """Command event with full context and routing information"""
    command_id: str
    command_name: str
    args: Dict[str, Any]
    context: SecurityContext
    source: str  # "voice", "api", "plugin", "system"
    priority: int = 5
    timeout: Optional[float] = None
    created_time: float = field(default_factory=time.time)
    handled: bool = False
    result: Any = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    plugin_handlers: List[str] = field(default_factory=list)

class PluginCommandRouter:
    """Ultra-complex plugin command routing with RBAC and event bubbling"""
    
    def __init__(self):
        self.plugins: List[Any] = []
        self.role_permissions: Dict[Role, Set[Permission]] = {
            Role.ADMIN: {
                Permission.EXEC_COMMAND, Permission.READ_DATA, Permission.WRITE_DATA,
                Permission.MODIFY_SETTINGS, Permission.ACCESS_AI, Permission.ACCESS_GPU,
                Permission.ACCESS_FILESYSTEM, Permission.ACCESS_NETWORK, 
                Permission.KILL_PROCESSES, Permission.INSTALL_PLUGINS
            },
            Role.USER: {
                Permission.READ_DATA, Permission.ACCESS_AI, Permission.ACCESS_GPU,
                Permission.ACCESS_FILESYSTEM
            },
            Role.GUEST: {
                Permission.READ_DATA
            },
            Role.SYSTEM: {
                Permission.EXEC_COMMAND, Permission.READ_DATA, Permission.WRITE_DATA,
                Permission.MODIFY_SETTINGS, Permission.ACCESS_AI, Permission.ACCESS_GPU,
                Permission.ACCESS_FILESYSTEM, Permission.ACCESS_NETWORK
            },
            Role.PLUGIN: {
                Permission.READ_DATA, Permission.ACCESS_AI, Permission.ACCESS_GPU,
                Permission.ACCESS_FILESYSTEM
            }
        }
        
        # Command routing statistics
        self.stats = {
            "commands_processed": 0,
            "commands_handled": 0,
            "commands_failed": 0,
            "permission_denied": 0,
            "total_execution_time": 0.0
        }
        
        # Event history for debugging
        self.event_history: List[CommandEvent] = []
        self.max_history = 1000

    def register_plugin(self, plugin: Any):
        """Register a plugin for command routing"""
        self.plugins.append(plugin)
        log.info(f"Registered plugin for command routing: {plugin.__class__.__name__}")

    def unregister_plugin(self, plugin: Any):
        """Unregister a plugin from command routing"""
        if plugin in self.plugins:
            self.plugins.remove(plugin)
            log.info(f"Unregistered plugin from command routing: {plugin.__class__.__name__}")

    def check_access(self, context: SecurityContext, permission: Permission) -> bool:
        """Check if security context has access to specific permission"""
        if not context.is_valid():
            log.warning(f"Security context expired for user {context.user_id}")
            return False
        
        # Check explicit permissions first
        if context.has_permission(permission):
            return True
        
        # Check role-based permissions
        for role in context.roles:
            if role in self.role_permissions:
                if permission in self.role_permissions[role]:
                    return True
        
        return False

    def get_required_permissions(self, command_name: str) -> Set[Permission]:
        """Get required permissions for a specific command"""
        # This could be extended with a command permission mapping
        permission_map = {
            "system_command": {Permission.EXEC_COMMAND},
            "read_file": {Permission.READ_DATA, Permission.ACCESS_FILESYSTEM},
            "write_file": {Permission.WRITE_DATA, Permission.ACCESS_FILESYSTEM},
            "ai_generate": {Permission.ACCESS_AI},
            "gpu_operation": {Permission.ACCESS_GPU},
            "kill_process": {Permission.KILL_PROCESSES},
            "install_plugin": {Permission.INSTALL_PLUGINS},
            "network_request": {Permission.ACCESS_NETWORK}
        }
        
        return permission_map.get(command_name, {Permission.EXEC_COMMAND})

    async def route_command(self, event: CommandEvent) -> CommandEvent:
        """Route command through plugins with full security and event bubbling"""
        start_time = time.time()
        self.stats["commands_processed"] += 1
        
        log.info(f"Routing command: {event.command_name} from {event.source} "
                f"(user: {event.context.user_id}, roles: {[r.name for r in event.context.roles]})")
        
        # Check permissions
        required_permissions = self.get_required_permissions(event.command_name)
        for permission in required_permissions:
            if not self.check_access(event.context, permission):
                event.error = f"Permission denied: {permission.name}"
                event.handled = True
                self.stats["permission_denied"] += 1
                log.warning(f"Permission denied for command {event.command_name}: {permission.name}")
                return event
        
        # Bubble event through plugins until handled
        for plugin in self.plugins:
            try:
                if hasattr(plugin, 'handle_command'):
                    handled = await plugin.handle_command(event)
                    if handled:
                        event.handled = True
                        event.plugin_handlers.append(plugin.__class__.__name__)
                        log.info(f"Command {event.command_name} handled by plugin: {plugin.__class__.__name__}")
                        break
                        
            except Exception as e:
                log.error(f"Plugin {plugin.__class__.__name__} failed handling command {event.command_name}: {e}")
                event.error = f"Plugin error: {str(e)}"
        
        # If no plugin handled, provide default behavior
        if not event.handled:
            event.result = await self._default_command_handler(event)
            event.handled = True
            log.info(f"No plugin handled command {event.command_name}, using default handler")
        
        # Update statistics
        event.execution_time = time.time() - start_time
        self.stats["total_execution_time"] += event.execution_time
        
        if event.handled and not event.error:
            self.stats["commands_handled"] += 1
        else:
            self.stats["commands_failed"] += 1
        
        # Store in history
        self.event_history.append(event)
        if len(self.event_history) > self.max_history:
            self.event_history.pop(0)
        
        log.info(f"Command {event.command_name} completed in {event.execution_time:.3f}s "
                f"(handled: {event.handled}, error: {event.error is not None})")
        
        return event

    async def _default_command_handler(self, event: CommandEvent) -> Any:
        """Default command handler for unhandled commands"""
        if event.command_name == "echo":
            return f"Echo: {event.args.get('message', '')}"
        elif event.command_name == "status":
            return {"status": "ok", "timestamp": time.time()}
        elif event.command_name == "help":
            return {
                "available_commands": ["echo", "status", "help"],
                "plugins": [p.__class__.__name__ for p in self.plugins]
            }
        else:
            return f"Unknown command: {event.command_name}"

    async def create_command_event(self, command_name: str, args: Dict[str, Any], 
                                 context: SecurityContext, source: str = "api") -> CommandEvent:
        """Create a new command event"""
        command_id = str(uuid.uuid4())
        return CommandEvent(
            command_id=command_id,
            command_name=command_name,
            args=args,
            context=context,
            source=source
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get router statistics"""
        return {
            **self.stats,
            "registered_plugins": len(self.plugins),
            "avg_execution_time": self.stats["total_execution_time"] / max(self.stats["commands_processed"], 1),
            "success_rate": self.stats["commands_handled"] / max(self.stats["commands_processed"], 1)
        }

    def get_event_history(self, limit: int = 100) -> List[CommandEvent]:
        """Get recent command event history"""
        return self.event_history[-limit:]

    def clear_history(self):
        """Clear command event history"""
        self.event_history.clear()
        log.info("Command event history cleared")

# Example plugin interface with async handle_command
class PluginBase:
    """Base class for plugins that can handle commands"""
    
    def __init__(self, name: str):
        self.name = name
        self.supported_commands: Set[str] = set()

    async def handle_command(self, event: CommandEvent) -> bool:
        """Handle a command event - override in subclasses"""
        return False

    def get_supported_commands(self) -> Set[str]:
        """Get list of commands this plugin supports"""
        return self.supported_commands

# Example secure plugin
class SecureEchoPlugin(PluginBase):
    """Example plugin with RBAC support"""
    
    def __init__(self):
        super().__init__("SecureEchoPlugin")
        self.supported_commands = {"echo_secure", "echo_admin"}

    async def handle_command(self, event: CommandEvent) -> bool:
        """Handle secure echo commands with role-based access"""
        
        if event.command_name == "echo_secure":
            # Anyone can use echo_secure
            event.result = f"Secure Echo: {event.args.get('message', '')}"
            return True
            
        elif event.command_name == "echo_admin":
            # Only admins can use echo_admin
            if Role.ADMIN in event.context.roles:
                event.result = f"Admin Echo: {event.args.get('message', '')}"
                return True
            else:
                event.error = "Access denied: Admin role required"
                return True
        
        return False

# Example usage
async def example_plugin_routing():
    """Example of using the plugin command router"""
    logging.basicConfig(level=logging.INFO)
    
    # Create router
    router = PluginCommandRouter()
    
    # Register plugins
    echo_plugin = SecureEchoPlugin()
    router.register_plugin(echo_plugin)
    
    # Create security contexts
    admin_context = SecurityContext(
        user_id="admin_user",
        session_id="session_1",
        roles={Role.ADMIN},
        permissions={Permission.EXEC_COMMAND, Permission.READ_DATA}
    )
    
    user_context = SecurityContext(
        user_id="regular_user",
        session_id="session_2",
        roles={Role.USER},
        permissions={Permission.READ_DATA}
    )
    
    # Route commands
    commands = [
        ("echo_secure", {"message": "Hello World"}, admin_context),
        ("echo_admin", {"message": "Admin Message"}, admin_context),
        ("echo_secure", {"message": "User Message"}, user_context),
        ("echo_admin", {"message": "User Admin Message"}, user_context),
    ]
    
    for command_name, args, context in commands:
        event = await router.create_command_event(command_name, args, context)
        result_event = await router.route_command(event)
        
        log.info(f"Command: {command_name}")
        log.info(f"  Result: {result_event.result}")
        log.info(f"  Error: {result_event.error}")
        log.info(f"  Handled: {result_event.handled}")
        log.info(f"  Execution time: {result_event.execution_time:.3f}s")
        log.info("---")
    
    # Get stats
    stats = router.get_stats()
    log.info(f"Router stats: {stats}")

if __name__ == "__main__":
    asyncio.run(example_plugin_routing())