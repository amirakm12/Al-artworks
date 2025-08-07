import asyncio
import logging
import time
from typing import Dict, Any
from ai_agent_orchestrator import AIAgentOrchestrator, AITask
from gpu_tuning_loops import GPUTuner
from plugin_routing import PluginCommandRouter, SecurityContext, Role, CommandEvent, SecureEchoPlugin

log = logging.getLogger("AIOrchestrationExample")

async def sample_ai_task(context: Dict[str, Any]) -> str:
    """Sample AI task that simulates model inference"""
    task_name = context.get("task_name", "unknown")
    duration = context.get("duration", 1.0)
    model_type = context.get("model_type", "llm")
    
    log.info(f"Starting AI task: {task_name} ({model_type})")
    await asyncio.sleep(duration)
    
    result = f"AI task {task_name} completed successfully using {model_type}"
    log.info(f"Completed AI task: {task_name}")
    return result

async def gpu_intensive_task(context: Dict[str, Any]) -> str:
    """GPU-intensive task that benefits from dynamic batch sizing"""
    task_name = context.get("task_name", "gpu_task")
    batch_size = context.get("batch_size", 8)
    
    log.info(f"Starting GPU-intensive task: {task_name} with batch size {batch_size}")
    await asyncio.sleep(2.0)  # Simulate GPU computation
    
    result = f"GPU task {task_name} completed with batch size {batch_size}"
    log.info(f"Completed GPU task: {task_name}")
    return result

async def voice_processing_task(context: Dict[str, Any]) -> str:
    """Voice processing task with real-time requirements"""
    task_name = context.get("task_name", "voice_task")
    audio_length = context.get("audio_length", 5.0)
    
    log.info(f"Starting voice processing: {task_name} ({audio_length}s audio)")
    await asyncio.sleep(audio_length * 0.1)  # Simulate real-time processing
    
    result = f"Voice processing {task_name} completed for {audio_length}s audio"
    log.info(f"Completed voice processing: {task_name}")
    return result

class AdvancedAIOrchestrator:
    """Combined AI orchestrator with GPU tuning and plugin routing"""
    
    def __init__(self, max_concurrent_tasks: int = 5):
        # Core orchestrator
        self.orchestrator = AIAgentOrchestrator(max_concurrent_tasks=max_concurrent_tasks)
        
        # GPU tuner
        self.gpu_tuner = GPUTuner(target_util=0.75, check_interval=3.0)
        
        # Plugin router
        self.plugin_router = PluginCommandRouter()
        
        # Security contexts
        self.security_contexts = {
            "admin": SecurityContext(
                user_id="admin",
                session_id="admin_session",
                roles={Role.ADMIN},
                permissions=set()  # Will inherit from role
            ),
            "user": SecurityContext(
                user_id="user",
                session_id="user_session", 
                roles={Role.USER},
                permissions=set()
            ),
            "system": SecurityContext(
                user_id="system",
                session_id="system_session",
                roles={Role.SYSTEM},
                permissions=set()
            )
        }
        
        # Task type priorities
        self.task_priorities = {
            "voice": 1,      # Highest priority for real-time voice
            "gpu": 3,        # Medium priority for GPU tasks
            "ai": 5,         # Standard priority for AI tasks
            "background": 8  # Low priority for background tasks
        }
        
        self.running = False

    async def start(self):
        """Start all orchestration components"""
        self.running = True
        log.info("Starting Advanced AI Orchestrator...")
        
        # Register plugins
        echo_plugin = SecureEchoPlugin()
        self.plugin_router.register_plugin(echo_plugin)
        
        # Start GPU tuner
        self.gpu_tuner.add_callback(self._gpu_stats_callback)
        gpu_task = asyncio.create_task(self.gpu_tuner.tuning_loop())
        
        # Start orchestrator
        orchestrator_task = asyncio.create_task(self.orchestrator.run())
        
        # Start monitoring loop
        monitor_task = asyncio.create_task(self._monitoring_loop())
        
        log.info("Advanced AI Orchestrator started successfully")
        
        return gpu_task, orchestrator_task, monitor_task

    async def submit_ai_task(self, task_type: str, task_name: str, 
                           context: Dict[str, Any] = None, priority: int = None) -> str:
        """Submit an AI task with automatic priority and GPU optimization"""
        
        # Determine task function based on type
        if task_type == "voice":
            task_func = voice_processing_task
            priority = priority or self.task_priorities["voice"]
        elif task_type == "gpu":
            task_func = gpu_intensive_task
            priority = priority or self.task_priorities["gpu"]
            # Add current GPU batch size to context
            context = context or {}
            context["batch_size"] = self.gpu_tuner.get_current_batch_size()
        else:
            task_func = sample_ai_task
            priority = priority or self.task_priorities["ai"]
        
        # Add task metadata
        task_context = context or {}
        task_context.update({
            "task_name": task_name,
            "task_type": task_type,
            "submitted_time": time.time()
        })
        
        # Submit to orchestrator
        task_id = await self.orchestrator.submit_task_func(
            task_func, 
            priority=priority,
            context=task_context,
            timeout=60.0  # 60 second timeout
        )
        
        log.info(f"Submitted {task_type} task '{task_name}' with priority {priority}")
        return task_id

    async def route_command(self, command_name: str, args: Dict[str, Any], 
                          user_id: str = "user", source: str = "api") -> CommandEvent:
        """Route a command through the plugin system"""
        
        # Get security context
        context = self.security_contexts.get(user_id, self.security_contexts["user"])
        
        # Create command event
        event = await self.plugin_router.create_command_event(
            command_name, args, context, source
        )
        
        # Route command
        result_event = await self.plugin_router.route_command(event)
        
        return result_event

    async def _gpu_stats_callback(self, stats):
        """Callback for GPU stats updates"""
        log.debug(f"GPU Stats - Util: {stats.utilization_percent:.1f}%, "
                 f"Mem: {stats.memory_allocated_mb:.1f}MB, "
                 f"Batch: {self.gpu_tuner.get_current_batch_size()}")

    async def _monitoring_loop(self):
        """Monitoring loop for system health and performance"""
        while self.running:
            try:
                # Get orchestrator stats
                orchestrator_stats = self.orchestrator.get_stats()
                
                # Get GPU stats
                gpu_stats = self.gpu_tuner.get_average_stats()
                
                # Get router stats
                router_stats = self.plugin_router.get_stats()
                
                # Log system health
                log.info(f"System Health - "
                        f"Tasks: {orchestrator_stats['running_tasks']} running, "
                        f"GPU Util: {gpu_stats.get('avg_utilization', 0):.1f}%, "
                        f"Commands: {router_stats['commands_processed']} processed")
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                log.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)

    async def stop(self):
        """Stop all orchestration components"""
        self.running = False
        log.info("Stopping Advanced AI Orchestrator...")
        
        # Stop GPU tuner
        self.gpu_tuner.stop()
        
        # Stop orchestrator
        self.orchestrator.stop()
        
        # Wait for tasks to complete
        await self.orchestrator.wait_for_all(timeout=30)
        
        log.info("Advanced AI Orchestrator stopped")

    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        return {
            "orchestrator": self.orchestrator.get_stats(),
            "gpu": self.gpu_tuner.get_average_stats(),
            "router": self.plugin_router.get_stats(),
            "device_info": self.gpu_tuner.get_device_info()
        }

# Example usage
async def example_advanced_orchestration():
    """Example of using the advanced AI orchestrator"""
    logging.basicConfig(level=logging.INFO)
    
    # Create orchestrator
    orchestrator = AdvancedAIOrchestrator(max_concurrent_tasks=3)
    
    # Start all components
    gpu_task, orchestrator_task, monitor_task = await orchestrator.start()
    
    try:
        # Submit various types of tasks
        voice_task_id = await orchestrator.submit_ai_task(
            "voice", "voice_recognition", {"audio_length": 3.0}
        )
        
        gpu_task_id = await orchestrator.submit_ai_task(
            "gpu", "image_generation", {"model_type": "stable_diffusion"}
        )
        
        ai_task_id = await orchestrator.submit_ai_task(
            "ai", "text_generation", {"model_type": "gpt"}
        )
        
        background_task_id = await orchestrator.submit_ai_task(
            "background", "data_processing", {"duration": 5.0}
        )
        
        # Route some commands
        commands = [
            ("echo_secure", {"message": "Hello from orchestrator"}),
            ("echo_admin", {"message": "Admin command"}),
            ("status", {}),
            ("help", {})
        ]
        
        for command_name, args in commands:
            result_event = await orchestrator.route_command(command_name, args, "admin")
            log.info(f"Command '{command_name}' result: {result_event.result}")
        
        # Wait for tasks to complete
        await orchestrator.orchestrator.wait_for_all(timeout=60)
        
        # Get final stats
        stats = orchestrator.get_system_stats()
        log.info("Final system stats:")
        for component, component_stats in stats.items():
            log.info(f"  {component}: {component_stats}")
        
    finally:
        # Cleanup
        await orchestrator.stop()
        gpu_task.cancel()
        orchestrator_task.cancel()
        monitor_task.cancel()

# Example plugin that integrates with the orchestrator
class AIOrchestratorPlugin(SecureEchoPlugin):
    """Advanced plugin that can submit tasks to the orchestrator"""
    
    def __init__(self, orchestrator: AdvancedAIOrchestrator):
        super().__init__()
        self.orchestrator = orchestrator
        self.supported_commands.add("submit_task")

    async def handle_command(self, event: CommandEvent) -> bool:
        """Handle commands including task submission"""
        
        if event.command_name == "submit_task":
            # Only admins can submit tasks
            if Role.ADMIN not in event.context.roles:
                event.error = "Access denied: Admin role required to submit tasks"
                return True
            
            task_type = event.args.get("task_type", "ai")
            task_name = event.args.get("task_name", "plugin_task")
            context = event.args.get("context", {})
            
            try:
                task_id = await self.orchestrator.submit_ai_task(task_type, task_name, context)
                event.result = {"task_id": task_id, "status": "submitted"}
                return True
            except Exception as e:
                event.error = f"Failed to submit task: {str(e)}"
                return True
        
        # Handle other commands via parent class
        return await super().handle_command(event)

if __name__ == "__main__":
    asyncio.run(example_advanced_orchestration())