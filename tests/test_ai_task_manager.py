import unittest
import asyncio
import pytest
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List

# Import the task manager classes
from orchestration.ai_task_manager import AITaskManager, AITask, TaskPriority, TaskStatus, TaskResult

# Mock AI Manager for testing
class MockAIManager:
    """Mock AI Manager for testing"""
    
    def __init__(self):
        self.generate_calls = 0
        self.last_prompt = None
        self.responses = {
            "hello": "Hello! How can I help you today?",
            "test": "This is a test response from the AI manager.",
            "error": "Error: Unable to process request."
        }
    
    def generate(self, prompt: str) -> str:
        """Mock text generation"""
        self.generate_calls += 1
        self.last_prompt = prompt
        
        # Return predefined responses for known prompts
        for key, response in self.responses.items():
            if key in prompt.lower():
                return response
        
        # Default response
        return f"AI response to: {prompt}"
    
    async def generate_response(self, prompt: str) -> str:
        """Async version of generate"""
        await asyncio.sleep(0.1)  # Simulate async processing
        return self.generate(prompt)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get mock statistics"""
        return {
            "generate_calls": self.generate_calls,
            "last_prompt": self.last_prompt
        }

# Mock Plugin for testing
class MockPlugin:
    """Mock plugin for testing"""
    
    def __init__(self, name="MockPlugin"):
        self.name = name
        self.commands_handled = 0
    
    def can_handle(self, command: str) -> bool:
        """Mock command handler check"""
        return "hello" in command.lower() or "test" in command.lower()
    
    def handle(self, command: str) -> str:
        """Mock command handler"""
        self.commands_handled += 1
        if "hello" in command.lower():
            return f"Hello from {self.name}!"
        elif "test" in command.lower():
            return f"Test handled by {self.name}"
        return f"Unknown command: {command}"
    
    async def handle_async(self, command: str) -> str:
        """Async version of handle"""
        await asyncio.sleep(0.01)  # Simulate async processing
        return self.handle(command)

class TestAITaskManager(unittest.TestCase):
    """Test suite for AI Task Manager functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.ai_manager = MockAIManager()
        self.plugins = [MockPlugin("TestPlugin1"), MockPlugin("TestPlugin2")]
        self.task_manager = AITaskManager(self.ai_manager, self.plugins, max_concurrent=3)
    
    def test_task_manager_initialization(self):
        """Test task manager initialization"""
        self.assertIsNotNone(self.task_manager)
        self.assertEqual(self.task_manager.ai_manager, self.ai_manager)
        self.assertEqual(self.task_manager.plugins, self.plugins)
        self.assertEqual(self.task_manager.max_concurrent, 3)
        self.assertFalse(self.task_manager.running)
    
    async def test_submit_task(self):
        """Test task submission"""
        async def sample_task(context: Dict[str, Any]) -> str:
            await asyncio.sleep(0.1)
            return f"Task completed with context: {context}"
        
        # Submit task
        task_id = await self.task_manager.submit_task(
            sample_task,
            priority=TaskPriority.HIGH,
            timeout=5.0,
            context={"data": "test"}
        )
        
        self.assertIsInstance(task_id, str)
        self.assertIn("task_", task_id)
        self.assertEqual(self.task_manager.stats["tasks_submitted"], 1)
    
    async def test_execute_voice_command(self):
        """Test voice command execution"""
        # Test with plugin handling
        response = await self.task_manager.execute_voice_command("hello world")
        self.assertIn("Plugin handled", response)
        
        # Test with AI fallback
        response = await self.task_manager.execute_voice_command("unknown command")
        self.assertIn("AI response", response)
        
        # Test with no handlers
        task_manager_no_handlers = AITaskManager()
        response = await task_manager_no_handlers.execute_voice_command("test command")
        self.assertIn("No handler available", response)
    
    async def test_wait_for_task(self):
        """Test waiting for task completion"""
        async def sample_task(context: Dict[str, Any]) -> str:
            await asyncio.sleep(0.1)
            return "Task completed"
        
        # Submit task
        task_id = await self.task_manager.submit_task(sample_task)
        
        # Wait for completion
        result = await self.task_manager.wait_for_task(task_id)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.status, TaskStatus.COMPLETED)
        self.assertEqual(result.result, "Task completed")
    
    async def test_task_priority_queuing(self):
        """Test task priority queuing"""
        async def sample_task(context: Dict[str, Any]) -> str:
            await asyncio.sleep(0.1)
            return f"Task with priority {context.get('priority', 'unknown')}"
        
        # Submit tasks with different priorities
        task_ids = []
        for priority in [TaskPriority.LOW, TaskPriority.HIGH, TaskPriority.MEDIUM]:
            task_id = await self.task_manager.submit_task(
                sample_task,
                priority=priority,
                context={"priority": priority.name}
            )
            task_ids.append(task_id)
        
        # Wait for all tasks to complete
        results = []
        for task_id in task_ids:
            result = await self.task_manager.wait_for_task(task_id)
            results.append(result)
        
        # Check that all tasks completed
        for result in results:
            self.assertEqual(result.status, TaskStatus.COMPLETED)
    
    async def test_task_timeout(self):
        """Test task timeout handling"""
        async def slow_task(context: Dict[str, Any]) -> str:
            await asyncio.sleep(1.0)  # Task takes 1 second
            return "Task completed"
        
        # Submit task with short timeout
        task_id = await self.task_manager.submit_task(
            slow_task,
            timeout=0.1  # 100ms timeout
        )
        
        # Wait for task
        result = await self.task_manager.wait_for_task(task_id)
        
        # Should timeout
        self.assertEqual(result.status, TaskStatus.FAILED)
        self.assertIn("timed out", result.error)
    
    async def test_task_cancellation(self):
        """Test task cancellation"""
        async def long_task(context: Dict[str, Any]) -> str:
            await asyncio.sleep(10.0)  # Very long task
            return "Task completed"
        
        # Submit task
        task_id = await self.task_manager.submit_task(long_task)
        
        # Cancel task
        success = await self.task_manager.cancel_task(task_id)
        self.assertTrue(success)
        
        # Check task status
        result = self.task_manager.get_task_status(task_id)
        self.assertEqual(result.status, TaskStatus.CANCELLED)
    
    async def test_concurrent_task_execution(self):
        """Test concurrent task execution"""
        async def sample_task(context: Dict[str, Any]) -> str:
            await asyncio.sleep(0.1)
            return f"Task {context.get('id', 'unknown')}"
        
        # Submit many tasks
        task_ids = []
        for i in range(10):
            task_id = await self.task_manager.submit_task(
                sample_task,
                context={"id": i}
            )
            task_ids.append(task_id)
        
        # Wait for all tasks to complete
        results = []
        for task_id in task_ids:
            result = await self.task_manager.wait_for_task(task_id)
            results.append(result)
        
        # Check results
        self.assertEqual(len(results), 10)
        completed_tasks = [r for r in results if r.status == TaskStatus.COMPLETED]
        self.assertGreater(len(completed_tasks), 0)
    
    def test_get_stats(self):
        """Test statistics gathering"""
        stats = self.task_manager.get_stats()
        
        self.assertIn("tasks_submitted", stats)
        self.assertIn("tasks_completed", stats)
        self.assertIn("tasks_failed", stats)
        self.assertIn("tasks_cancelled", stats)
        self.assertIn("total_runtime", stats)
        self.assertIn("running_tasks", stats)
        self.assertIn("queued_tasks", stats)
        self.assertIn("completed_tasks", stats)
        self.assertIn("avg_runtime", stats)
    
    async def test_add_remove_plugins(self):
        """Test adding and removing plugins"""
        # Add plugin
        new_plugin = MockPlugin("NewPlugin")
        self.task_manager.add_plugin(new_plugin)
        self.assertEqual(len(self.task_manager.plugins), 3)
        
        # Remove plugin
        self.task_manager.remove_plugin("TestPlugin1")
        self.assertEqual(len(self.task_manager.plugins), 2)
        self.assertNotIn("TestPlugin1", [p.name for p in self.task_manager.plugins])

@pytest.mark.asyncio
class TestAITaskManagerAsync:
    """Async test suite for AI Task Manager"""
    
    @pytest.fixture
    def task_manager(self):
        """Async fixture for task manager"""
        ai_manager = MockAIManager()
        plugins = [MockPlugin("TestPlugin")]
        return AITaskManager(ai_manager, plugins)
    
    async def test_async_task_operations(self, task_manager):
        """Test async task operations"""
        async def sample_task(context: Dict[str, Any]) -> str:
            await asyncio.sleep(0.1)
            return "Async task completed"
        
        # Submit and wait for task
        task_id = await task_manager.submit_task(sample_task)
        result = await task_manager.wait_for_task(task_id)
        
        self.assertEqual(result.status, TaskStatus.COMPLETED)
        self.assertEqual(result.result, "Async task completed")
    
    async def test_concurrent_task_submission(self, task_manager):
        """Test concurrent task submission"""
        async def sample_task(context: Dict[str, Any]) -> str:
            await asyncio.sleep(0.1)
            return f"Task {context.get('id', 'unknown')}"
        
        # Submit many tasks concurrently
        tasks = []
        for i in range(20):
            task = task_manager.submit_task(sample_task, context={"id": i})
            tasks.append(task)
        
        # Wait for all submissions
        task_ids = await asyncio.gather(*tasks)
        
        # Wait for all completions
        results = []
        for task_id in task_ids:
            result = await task_manager.wait_for_task(task_id)
            results.append(result)
        
        # Check results
        self.assertEqual(len(results), 20)
        completed = [r for r in results if r.status == TaskStatus.COMPLETED]
        self.assertGreater(len(completed), 0)
    
    async def test_task_manager_error_handling(self, task_manager):
        """Test error handling in task manager"""
        async def error_task(context: Dict[str, Any]) -> str:
            raise Exception("Test error")
        
        # Submit error task
        task_id = await task_manager.submit_task(error_task)
        result = await task_manager.wait_for_task(task_id)
        
        # Should fail gracefully
        self.assertEqual(result.status, TaskStatus.FAILED)
        self.assertIn("Test error", result.error)

class TestAITask:
    """Test suite for AITask class"""
    
    def test_task_creation(self):
        """Test task creation"""
        async def sample_func(context: Dict[str, Any]) -> str:
            return "test"
        
        task = AITask("test_task", sample_func, TaskPriority.HIGH, 5.0, {"data": "test"})
        
        self.assertEqual(task.task_id, "test_task")
        self.assertEqual(task.priority, TaskPriority.HIGH)
        self.assertEqual(task.timeout, 5.0)
        self.assertEqual(task.context, {"data": "test"})
        self.assertEqual(task.status, TaskStatus.PENDING)
    
    async def test_task_execution(self):
        """Test task execution"""
        async def sample_func(context: Dict[str, Any]) -> str:
            await asyncio.sleep(0.1)
            return f"Result: {context.get('data', 'unknown')}"
        
        task = AITask("test_task", sample_func, TaskPriority.MEDIUM, None, {"data": "test"})
        
        # Execute task
        result = await task.execute()
        
        # Check result
        self.assertEqual(result.status, TaskStatus.COMPLETED)
        self.assertEqual(result.result, "Result: test")
        self.assertIsNotNone(result.start_time)
        self.assertIsNotNone(result.end_time)
        self.assertIsNotNone(result.duration)
    
    async def test_task_timeout(self):
        """Test task timeout"""
        async def slow_func(context: Dict[str, Any]) -> str:
            await asyncio.sleep(1.0)
            return "slow result"
        
        task = AITask("test_task", slow_func, TaskPriority.MEDIUM, 0.1, {})
        
        # Execute task
        result = await task.execute()
        
        # Should timeout
        self.assertEqual(result.status, TaskStatus.FAILED)
        self.assertIn("timed out", result.error)
    
    async def test_task_error_handling(self):
        """Test task error handling"""
        async def error_func(context: Dict[str, Any]) -> str:
            raise Exception("Test error")
        
        task = AITask("test_task", error_func, TaskPriority.MEDIUM, None, {})
        
        # Execute task
        result = await task.execute()
        
        # Should fail
        self.assertEqual(result.status, TaskStatus.FAILED)
        self.assertIn("Test error", result.error)
    
    def test_task_cancellation(self):
        """Test task cancellation"""
        async def sample_func(context: Dict[str, Any]) -> str:
            return "test"
        
        task = AITask("test_task", sample_func, TaskPriority.MEDIUM, None, {})
        
        # Cancel task
        task.cancel()
        
        # Check status
        self.assertEqual(task.status, TaskStatus.CANCELLED)
    
    def test_get_result(self):
        """Test getting task result"""
        async def sample_func(context: Dict[str, Any]) -> str:
            return "test result"
        
        task = AITask("test_task", sample_func, TaskPriority.MEDIUM, None, {"data": "test"})
        
        # Get result before execution
        result = task.get_result()
        self.assertEqual(result.status, TaskStatus.PENDING)
        self.assertIsNone(result.result)

class TestAITaskManagerIntegration:
    """Integration tests for AI Task Manager with real components"""
    
    def test_task_manager_with_real_ai_manager(self):
        """Test task manager with real AI manager"""
        # This would test with real AI models
        # Implementation depends on your actual AI manager
        pass
    
    def test_task_manager_with_real_plugins(self):
        """Test task manager with real plugins"""
        # This would test with real plugin implementations
        # Implementation depends on your actual plugins
        pass

# Performance tests
class TestAITaskManagerPerformance:
    """Performance tests for AI Task Manager"""
    
    async def test_many_tasks_performance(self):
        """Test performance with many tasks"""
        task_manager = AITaskManager(max_concurrent=5)
        
        async def sample_task(context: Dict[str, Any]) -> str:
            await asyncio.sleep(0.01)  # Quick task
            return f"Task {context.get('id', 'unknown')}"
        
        import time
        start_time = time.time()
        
        # Submit many tasks
        task_ids = []
        for i in range(100):
            task_id = await task_manager.submit_task(sample_task, context={"id": i})
            task_ids.append(task_id)
        
        # Wait for all tasks
        results = []
        for task_id in task_ids:
            result = await task_manager.wait_for_task(task_id)
            results.append(result)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete within reasonable time
        self.assertLess(duration, 10.0)  # Less than 10 seconds for 100 tasks
        self.assertEqual(len(results), 100)
    
    async def test_concurrent_task_performance(self):
        """Test concurrent task performance"""
        task_manager = AITaskManager(max_concurrent=10)
        
        async def sample_task(context: Dict[str, Any]) -> str:
            await asyncio.sleep(0.1)
            return f"Task {context.get('id', 'unknown')}"
        
        import time
        start_time = time.time()
        
        # Submit tasks concurrently
        tasks = []
        for i in range(50):
            task = task_manager.submit_task(sample_task, context={"id": i})
            tasks.append(task)
        
        # Wait for all submissions
        task_ids = await asyncio.gather(*tasks)
        
        # Wait for all completions
        results = []
        for task_id in task_ids:
            result = await task_manager.wait_for_task(task_id)
            results.append(result)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete within reasonable time
        self.assertLess(duration, 15.0)  # Less than 15 seconds for 50 tasks
        self.assertEqual(len(results), 50)

# Memory tests
class TestAITaskManagerMemory:
    """Memory usage tests for AI Task Manager"""
    
    async def test_memory_usage_with_many_tasks(self):
        """Test memory usage with many tasks"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        task_manager = AITaskManager(max_concurrent=5)
        
        async def sample_task(context: Dict[str, Any]) -> str:
            await asyncio.sleep(0.01)
            return f"Task {context.get('id', 'unknown')}"
        
        # Submit and complete many tasks
        for i in range(1000):
            task_id = await task_manager.submit_task(sample_task, context={"id": i})
            result = await task_manager.wait_for_task(task_id)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 200MB)
        self.assertLess(memory_increase, 200 * 1024 * 1024)

if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2)