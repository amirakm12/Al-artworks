import asyncio
import logging
import time
import uuid
from collections import deque, defaultdict
from enum import Enum, auto
from typing import Dict, Any, Optional, Callable, Coroutine
from dataclasses import dataclass

log = logging.getLogger("AIAgentOrchestrator")

class TaskStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    SUCCESS = auto()
    FAILED = auto()
    RETRYING = auto()
    CANCELLED = auto()

@dataclass
class TaskResult:
    task_id: str
    status: TaskStatus
    result: Any = None
    exception: Optional[Exception] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    retry_count: int = 0
    context: Dict[str, Any] = None

class AITask:
    """Represents an async AI task with priority, retries, and context"""
    
    def __init__(self, task_id: str, coro_func: Callable, priority: int = 5, 
                 max_retries: int = 3, context: Optional[Dict[str, Any]] = None,
                 timeout: Optional[float] = None):
        self.task_id = task_id
        self.coro_func = coro_func  # Async callable
        self.priority = priority    # 1 (highest) to 10 (lowest)
        self.max_retries = max_retries
        self.timeout = timeout
        self.retry_count = 0
        self.context = context or {}
        self.status = TaskStatus.PENDING
        self.result = None
        self.exception = None
        self.start_time = None
        self.end_time = None
        self.created_time = time.time()

    async def run(self) -> Any:
        """Execute the task with retry logic and exponential backoff"""
        self.status = TaskStatus.RUNNING
        self.start_time = time.time()
        
        while self.retry_count <= self.max_retries:
            try:
                if self.timeout:
                    self.result = await asyncio.wait_for(
                        self.coro_func(self.context), 
                        timeout=self.timeout
                    )
                else:
                    self.result = await self.coro_func(self.context)
                
                self.status = TaskStatus.SUCCESS
                self.end_time = time.time()
                duration = self.end_time - self.start_time
                log.info(f"Task {self.task_id} succeeded in {duration:.2f}s")
                return self.result
                
            except asyncio.TimeoutError:
                self.exception = asyncio.TimeoutError(f"Task {self.task_id} timed out after {self.timeout}s")
                self.retry_count += 1
                log.warning(f"Task {self.task_id} timed out, attempt {self.retry_count}/{self.max_retries}")
                
            except Exception as e:
                self.exception = e
                self.retry_count += 1
                log.warning(f"Task {self.task_id} failed attempt {self.retry_count}/{self.max_retries}: {e}")
                
                if self.retry_count > self.max_retries:
                    self.status = TaskStatus.FAILED
                    self.end_time = time.time()
                    log.error(f"Task {self.task_id} failed permanently after {self.max_retries} retries")
                    raise
                
                # Exponential backoff with jitter
                backoff_time = min(2 ** self.retry_count + (time.time() % 1), 60)
                log.info(f"Retrying task {self.task_id} in {backoff_time:.2f}s")
                await asyncio.sleep(backoff_time)

    def cancel(self):
        """Cancel the task"""
        self.status = TaskStatus.CANCELLED
        log.info(f"Task {self.task_id} cancelled")

    def get_result(self) -> TaskResult:
        """Get task result with metadata"""
        return TaskResult(
            task_id=self.task_id,
            status=self.status,
            result=self.result,
            exception=self.exception,
            start_time=self.start_time,
            end_time=self.end_time,
            retry_count=self.retry_count,
            context=self.context
        )

class AIAgentOrchestrator:
    """Advanced AI task orchestrator with priority queues and concurrent execution"""
    
    def __init__(self, max_concurrent_tasks: int = 5, max_queue_size: int = 1000):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.max_queue_size = max_queue_size
        self.task_queue = deque()
        self.running_tasks: Dict[str, AITask] = {}
        self.completed_tasks: Dict[str, TaskResult] = {}
        self.priority_queues = defaultdict(deque)
        self.lock = asyncio.Lock()
        self.running = False
        self.stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "tasks_cancelled": 0,
            "total_runtime": 0.0
        }

    async def submit_task(self, task: AITask) -> bool:
        """Submit a task to the orchestrator"""
        async with self.lock:
            if len(self.completed_tasks) + len(self.running_tasks) >= self.max_queue_size:
                log.warning(f"Task queue full, rejecting task {task.task_id}")
                return False
            
            self.priority_queues[task.priority].append(task)
            self.stats["tasks_submitted"] += 1
            log.debug(f"Submitted task {task.task_id} with priority {task.priority}")
            return True

    async def submit_task_func(self, coro_func: Callable, priority: int = 5, 
                              max_retries: int = 3, context: Optional[Dict[str, Any]] = None,
                              timeout: Optional[float] = None) -> str:
        """Submit a task function and return the task ID"""
        task_id = str(uuid.uuid4())
        task = AITask(task_id, coro_func, priority, max_retries, context, timeout)
        success = await self.submit_task(task)
        if not success:
            raise RuntimeError("Failed to submit task - queue full")
        return task_id

    async def _get_next_task(self) -> Optional[AITask]:
        """Get the next highest priority task"""
        async with self.lock:
            for priority in sorted(self.priority_queues.keys()):
                if self.priority_queues[priority]:
                    return self.priority_queues[priority].popleft()
            return None

    async def _task_runner(self, task: AITask):
        """Run a single task and handle completion"""
        self.running_tasks[task.task_id] = task
        
        try:
            result = await task.run()
            self.completed_tasks[task.task_id] = task.get_result()
            self.stats["tasks_completed"] += 1
            if task.start_time and task.end_time:
                self.stats["total_runtime"] += task.end_time - task.start_time
                
        except Exception as e:
            log.error(f"Task {task.task_id} failed: {e}")
            self.completed_tasks[task.task_id] = task.get_result()
            self.stats["tasks_failed"] += 1
            
        finally:
            if task.task_id in self.running_tasks:
                del self.running_tasks[task.task_id]

    async def run(self):
        """Main orchestrator loop"""
        self.running = True
        log.info(f"Starting AI Agent Orchestrator with {self.max_concurrent_tasks} max concurrent tasks")
        
        while self.running:
            try:
                # Start new tasks if we have capacity
                while len(self.running_tasks) < self.max_concurrent_tasks:
                    task = await self._get_next_task()
                    if task:
                        asyncio.create_task(self._task_runner(task))
                    else:
                        break
                
                # Log stats periodically
                if len(self.running_tasks) > 0 or any(self.priority_queues.values()):
                    log.debug(f"Running: {len(self.running_tasks)}, Queued: {sum(len(q) for q in self.priority_queues.values())}")
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                log.error(f"Error in orchestrator loop: {e}")
                await asyncio.sleep(1)

    async def wait_for_all(self, timeout: Optional[float] = None):
        """Wait for all tasks to complete"""
        start_time = time.time()
        while (self.running_tasks or any(self.priority_queues.values())):
            if timeout and (time.time() - start_time) > timeout:
                log.warning("Timeout waiting for tasks to complete")
                break
            await asyncio.sleep(0.1)

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task"""
        async with self.lock:
            if task_id in self.running_tasks:
                task = self.running_tasks[task_id]
                task.cancel()
                self.stats["tasks_cancelled"] += 1
                log.info(f"Cancelled task {task_id}")
                return True
            return False

    async def cancel_all_tasks(self):
        """Cancel all running tasks"""
        async with self.lock:
            for task in self.running_tasks.values():
                task.cancel()
            self.stats["tasks_cancelled"] += len(self.running_tasks)
            log.info(f"Cancelled {len(self.running_tasks)} tasks")

    def get_task_status(self, task_id: str) -> Optional[TaskResult]:
        """Get status of a specific task"""
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id]
        elif task_id in self.running_tasks:
            return self.running_tasks[task_id].get_result()
        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics"""
        return {
            **self.stats,
            "running_tasks": len(self.running_tasks),
            "queued_tasks": sum(len(q) for q in self.priority_queues.values()),
            "completed_tasks": len(self.completed_tasks),
            "avg_runtime": self.stats["total_runtime"] / max(self.stats["tasks_completed"], 1)
        }

    def stop(self):
        """Stop the orchestrator"""
        self.running = False
        log.info("AI Agent Orchestrator stopped")

    async def shutdown(self):
        """Graceful shutdown - wait for running tasks to complete"""
        log.info("Shutting down AI Agent Orchestrator...")
        self.running = False
        await self.wait_for_all(timeout=30)
        await self.cancel_all_tasks()
        log.info("AI Agent Orchestrator shutdown complete")

# Example usage
async def sample_ai_task(context: Dict[str, Any]) -> str:
    """Example AI task that simulates model inference"""
    task_name = context.get("task_name", "unknown")
    duration = context.get("duration", 1.0)
    
    log.info(f"Starting AI task: {task_name}")
    await asyncio.sleep(duration)
    
    result = f"AI task {task_name} completed successfully"
    log.info(f"Completed AI task: {task_name}")
    return result

async def example_usage():
    """Example of using the AI Agent Orchestrator"""
    orchestrator = AIAgentOrchestrator(max_concurrent_tasks=3)
    
    # Submit tasks with different priorities
    task1_id = await orchestrator.submit_task_func(
        sample_ai_task, 
        priority=1, 
        context={"task_name": "high_priority", "duration": 2.0}
    )
    
    task2_id = await orchestrator.submit_task_func(
        sample_ai_task, 
        priority=5, 
        context={"task_name": "medium_priority", "duration": 1.0}
    )
    
    task3_id = await orchestrator.submit_task_func(
        sample_ai_task, 
        priority=10, 
        context={"task_name": "low_priority", "duration": 3.0}
    )
    
    # Start orchestrator
    orchestrator_task = asyncio.create_task(orchestrator.run())
    
    # Wait for all tasks to complete
    await orchestrator.wait_for_all()
    
    # Get results
    for task_id in [task1_id, task2_id, task3_id]:
        result = orchestrator.get_task_status(task_id)
        if result:
            log.info(f"Task {task_id}: {result.status} - {result.result}")
    
    # Get stats
    stats = orchestrator.get_stats()
    log.info(f"Orchestrator stats: {stats}")
    
    # Shutdown
    orchestrator.stop()
    orchestrator_task.cancel()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(example_usage())