import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

log = logging.getLogger("AITaskManager")

class TaskPriority(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class TaskResult:
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration: Optional[float] = None

class AITask:
    """Represents an AI task with priority and metadata"""
    
    def __init__(self, task_id: str, func: Callable, priority: TaskPriority = TaskPriority.MEDIUM,
                 timeout: Optional[float] = None, context: Optional[Dict[str, Any]] = None):
        self.task_id = task_id
        self.func = func
        self.priority = priority
        self.timeout = timeout
        self.context = context or {}
        self.status = TaskStatus.PENDING
        self.result = None
        self.error = None
        self.start_time = None
        self.end_time = None
        self.created_time = time.time()

    async def execute(self) -> TaskResult:
        """Execute the task asynchronously"""
        self.status = TaskStatus.RUNNING
        self.start_time = time.time()
        
        try:
            if asyncio.iscoroutinefunction(self.func):
                if self.timeout:
                    self.result = await asyncio.wait_for(self.func(self.context), timeout=self.timeout)
                else:
                    self.result = await self.func(self.context)
            else:
                # Run sync function in thread pool
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor() as executor:
                    if self.timeout:
                        self.result = await asyncio.wait_for(
                            loop.run_in_executor(executor, self.func, self.context),
                            timeout=self.timeout
                        )
                    else:
                        self.result = await loop.run_in_executor(executor, self.func, self.context)
            
            self.status = TaskStatus.COMPLETED
            self.end_time = time.time()
            log.info(f"Task {self.task_id} completed successfully")
            
        except asyncio.TimeoutError:
            self.status = TaskStatus.FAILED
            self.error = f"Task {self.task_id} timed out after {self.timeout}s"
            self.end_time = time.time()
            log.error(self.error)
            
        except Exception as e:
            self.status = TaskStatus.FAILED
            self.error = str(e)
            self.end_time = time.time()
            log.error(f"Task {self.task_id} failed: {e}")
        
        return self.get_result()

    def get_result(self) -> TaskResult:
        """Get task result with metadata"""
        duration = None
        if self.start_time and self.end_time:
            duration = self.end_time - self.start_time
        
        return TaskResult(
            task_id=self.task_id,
            status=self.status,
            result=self.result,
            error=self.error,
            start_time=self.start_time,
            end_time=self.end_time,
            duration=duration
        )

    def cancel(self):
        """Cancel the task"""
        self.status = TaskStatus.CANCELLED
        log.info(f"Task {self.task_id} cancelled")

class AITaskManager:
    """Advanced AI task manager with plugin integration and priority queuing"""
    
    def __init__(self, ai_manager=None, plugins: List = None, max_concurrent: int = 5):
        self.ai_manager = ai_manager
        self.plugins = plugins or []
        self.max_concurrent = max_concurrent
        self.running = False
        
        # Priority queues for different task types
        self.task_queues = {
            TaskPriority.CRITICAL: asyncio.Queue(),
            TaskPriority.HIGH: asyncio.Queue(),
            TaskPriority.MEDIUM: asyncio.Queue(),
            TaskPriority.LOW: asyncio.Queue(),
            TaskPriority.BACKGROUND: asyncio.Queue()
        }
        
        # Task tracking
        self.running_tasks: Dict[str, AITask] = {}
        self.completed_tasks: Dict[str, TaskResult] = {}
        self.task_counter = 0
        
        # Statistics
        self.stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "tasks_cancelled": 0,
            "total_runtime": 0.0
        }

    async def submit_task(self, func: Callable, priority: TaskPriority = TaskPriority.MEDIUM,
                         timeout: Optional[float] = None, context: Optional[Dict[str, Any]] = None) -> str:
        """Submit a task for execution"""
        self.task_counter += 1
        task_id = f"task_{self.task_counter}_{int(time.time())}"
        
        task = AITask(task_id, func, priority, timeout, context)
        await self.task_queues[priority].put(task)
        
        self.stats["tasks_submitted"] += 1
        log.debug(f"Submitted task {task_id} with priority {priority.value}")
        
        return task_id

    async def execute_voice_command(self, command: str) -> str:
        """Execute a voice command through plugins or AI manager"""
        async def voice_command_task(context: Dict[str, Any]) -> str:
            command_text = context.get("command", "")
            
            # First, try to handle with plugins
            for plugin in self.plugins:
                if hasattr(plugin, 'can_handle') and plugin.can_handle(command_text):
                    try:
                        if hasattr(plugin, 'handle_async'):
                            result = await plugin.handle_async(command_text)
                        else:
                            result = plugin.handle(command_text)
                        log.info(f"Plugin {plugin.name} handled command: {command_text}")
                        return result
                    except Exception as e:
                        log.error(f"Plugin {plugin.name} failed handling command: {e}")
            
            # Fallback to AI manager
            if self.ai_manager:
                try:
                    if hasattr(self.ai_manager, 'generate_response'):
                        result = await self.ai_manager.generate_response(command_text)
                    else:
                        result = self.ai_manager.generate(command_text)
                    log.info(f"AI manager handled command: {command_text}")
                    return result
                except Exception as e:
                    log.error(f"AI manager failed handling command: {e}")
                    return f"Error processing command: {str(e)}"
            
            return "No handler available for this command."

        # Submit as high priority task
        task_id = await self.submit_task(
            voice_command_task,
            priority=TaskPriority.HIGH,
            timeout=30.0,  # 30 second timeout for voice commands
            context={"command": command}
        )
        
        # Wait for completion
        result = await self.wait_for_task(task_id)
        return result.result if result and result.status == TaskStatus.COMPLETED else "Task failed"

    async def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Optional[TaskResult]:
        """Wait for a specific task to complete"""
        start_time = time.time()
        
        while True:
            if task_id in self.completed_tasks:
                return self.completed_tasks[task_id]
            
            if timeout and (time.time() - start_time) > timeout:
                log.warning(f"Timeout waiting for task {task_id}")
                return None
            
            await asyncio.sleep(0.1)

    async def _task_worker(self, priority: TaskPriority):
        """Worker that processes tasks from a specific priority queue"""
        queue = self.task_queues[priority]
        
        while self.running:
            try:
                # Get task from queue
                task = await asyncio.wait_for(queue.get(), timeout=1.0)
                
                # Check if we have capacity
                if len(self.running_tasks) >= self.max_concurrent:
                    # Put task back and wait
                    await queue.put(task)
                    await asyncio.sleep(0.1)
                    continue
                
                # Execute task
                self.running_tasks[task.task_id] = task
                result = await task.execute()
                
                # Store result
                self.completed_tasks[task.task_id] = result
                if task.task_id in self.running_tasks:
                    del self.running_tasks[task.task_id]
                
                # Update statistics
                if result.status == TaskStatus.COMPLETED:
                    self.stats["tasks_completed"] += 1
                    if result.duration:
                        self.stats["total_runtime"] += result.duration
                elif result.status == TaskStatus.FAILED:
                    self.stats["tasks_failed"] += 1
                elif result.status == TaskStatus.CANCELLED:
                    self.stats["tasks_cancelled"] += 1
                
                log.debug(f"Task {task.task_id} completed with status {result.status}")
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                log.error(f"Error in task worker for priority {priority}: {e}")

    async def start(self):
        """Start the task manager"""
        self.running = True
        log.info(f"Starting AI Task Manager with {self.max_concurrent} max concurrent tasks")
        
        # Start workers for each priority level
        workers = []
        for priority in TaskPriority:
            worker = asyncio.create_task(self._task_worker(priority))
            workers.append(worker)
        
        return workers

    async def stop(self):
        """Stop the task manager"""
        self.running = False
        log.info("AI Task Manager stopped")

    async def shutdown(self):
        """Graceful shutdown - wait for running tasks to complete"""
        log.info("Shutting down AI Task Manager...")
        self.running = False
        
        # Wait for running tasks to complete (with timeout)
        start_time = time.time()
        while self.running_tasks and (time.time() - start_time) < 30:
            await asyncio.sleep(0.1)
        
        # Cancel remaining tasks
        for task in self.running_tasks.values():
            task.cancel()
        
        log.info("AI Task Manager shutdown complete")

    def get_stats(self) -> Dict[str, Any]:
        """Get task manager statistics"""
        return {
            **self.stats,
            "running_tasks": len(self.running_tasks),
            "queued_tasks": sum(q.qsize() for q in self.task_queues.values()),
            "completed_tasks": len(self.completed_tasks),
            "avg_runtime": self.stats["total_runtime"] / max(self.stats["tasks_completed"], 1)
        }

    def get_task_status(self, task_id: str) -> Optional[TaskResult]:
        """Get status of a specific task"""
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id]
        elif task_id in self.running_tasks:
            return self.running_tasks[task_id].get_result()
        return None

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task"""
        if task_id in self.running_tasks:
            task = self.running_tasks[task_id]
            task.cancel()
            self.stats["tasks_cancelled"] += 1
            log.info(f"Cancelled task {task_id}")
            return True
        return False

    def add_plugin(self, plugin):
        """Add a plugin to the task manager"""
        self.plugins.append(plugin)
        log.info(f"Added plugin {plugin.name} to task manager")

    def remove_plugin(self, plugin_name: str):
        """Remove a plugin from the task manager"""
        self.plugins = [p for p in self.plugins if p.name != plugin_name]
        log.info(f"Removed plugin {plugin_name} from task manager")

# Example usage
async def example_usage():
    """Example of using the AI Task Manager"""
    # Create task manager
    task_manager = AITaskManager(max_concurrent=3)
    
    # Start the manager
    workers = await task_manager.start()
    
    # Submit some tasks
    async def sample_task(context: Dict[str, Any]) -> str:
        await asyncio.sleep(1)  # Simulate work
        return f"Task completed with context: {context}"
    
    task_id1 = await task_manager.submit_task(
        sample_task,
        priority=TaskPriority.HIGH,
        context={"data": "high priority task"}
    )
    
    task_id2 = await task_manager.submit_task(
        sample_task,
        priority=TaskPriority.LOW,
        context={"data": "low priority task"}
    )
    
    # Wait for tasks to complete
    result1 = await task_manager.wait_for_task(task_id1)
    result2 = await task_manager.wait_for_task(task_id2)
    
    print(f"Task 1 result: {result1}")
    print(f"Task 2 result: {result2}")
    
    # Get statistics
    stats = task_manager.get_stats()
    print(f"Task manager stats: {stats}")
    
    # Shutdown
    await task_manager.shutdown()
    for worker in workers:
        worker.cancel()

if __name__ == "__main__":
    asyncio.run(example_usage())