import asyncio
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from plugins.sdk import PluginManager, PluginBase, EventEmitter, PluginContext

@pytest.mark.asyncio
async def test_plugin_load_and_events(tmp_path):
    """Test plugin loading and event handling"""
    plugin_code = """
import asyncio

class Plugin:
    def __init__(self, sdk):
        self.sdk = sdk
        self.loaded = False
        self.unloaded = False
        self.events = []

    async def on_load(self):
        self.loaded = True

    async def on_unload(self):
        self.unloaded = True

    async def on_event(self, event_name, *args, **kwargs):
        self.events.append(event_name)
"""
    plugin_file = tmp_path / "test_plugin.py"
    plugin_file.write_text(plugin_code)

    pm = PluginManager(plugin_dir=str(tmp_path))
    await pm.load_all_plugins()
    
    assert "test_plugin" in pm.plugins
    plugin = pm.plugins["test_plugin"]
    assert plugin.loaded

    await pm.emit_event("test_event")
    assert "test_event" in plugin.events

    await pm.unload_all_plugins()
    assert plugin.unloaded

@pytest.mark.asyncio
async def test_event_emitter():
    """Test event emitter functionality"""
    emitter = EventEmitter()
    events_received = []

    async def test_handler(event_name, *args, **kwargs):
        events_received.append(event_name)

    emitter.on("test_event", test_handler)
    await emitter.emit("test_event")
    
    assert "test_event" in events_received

@pytest.mark.asyncio
async def test_plugin_context():
    """Test plugin context with AI manager and GPU device"""
    mock_ai_manager = Mock()
    mock_device = Mock()
    mock_loop = asyncio.get_event_loop()
    config = {"test": "config"}
    
    context = PluginContext(config, mock_ai_manager, mock_device, mock_loop)
    
    # Test state management
    context.save_state("test_key", "test_value")
    assert context.get_state("test_key") == "test_value"
    assert context.get_state("nonexistent", "default") == "default"

@pytest.mark.asyncio
async def test_plugin_base():
    """Test PluginBase class functionality"""
    mock_context = Mock()
    plugin = PluginBase(mock_context)
    
    # Test default implementations
    await plugin.on_load()
    await plugin.on_unload()
    await plugin.on_event("test_event")

@pytest.mark.asyncio
async def test_plugin_manager_error_handling(tmp_path):
    """Test plugin manager handles loading errors gracefully"""
    # Create a plugin with syntax error
    plugin_code = """
class Plugin:
    def __init__(self, sdk):
        self.sdk = sdk
        # Syntax error
        if True
"""
    plugin_file = tmp_path / "bad_plugin.py"
    plugin_file.write_text(plugin_code)

    pm = PluginManager(plugin_dir=str(tmp_path))
    # Should not raise exception, just log error
    await pm.load_all_plugins()
    
    # Plugin should not be loaded due to syntax error
    assert "bad_plugin" not in pm.plugins

@pytest.mark.asyncio
async def test_plugin_reload(tmp_path):
    """Test plugin reloading functionality"""
    plugin_code = """
class Plugin:
    def __init__(self, sdk):
        self.sdk = sdk
        self.loaded = False
        self.unloaded = False

    async def on_load(self):
        self.loaded = True

    async def on_unload(self):
        self.unloaded = True
"""
    plugin_file = tmp_path / "reload_test_plugin.py"
    plugin_file.write_text(plugin_code)

    pm = PluginManager(plugin_dir=str(tmp_path))
    await pm.load_all_plugins()
    
    plugin = pm.plugins["reload_test_plugin"]
    assert plugin.loaded
    
    # Test reload
    success = pm.reload_plugin("reload_test_plugin")
    assert success
    assert plugin.unloaded  # Old plugin should be unloaded

@pytest.mark.asyncio
async def test_plugin_manager_info():
    """Test plugin manager info retrieval"""
    pm = PluginManager(plugin_dir="nonexistent")
    info = pm.get_plugin_info()
    
    assert "total_plugins" in info
    assert "plugins" in info
    assert "sandbox_enabled" in info
    assert "plugin_directory" in info

@pytest.mark.asyncio
async def test_async_task_execution():
    """Test async task execution in plugin context"""
    mock_context = Mock()
    mock_context.loop = asyncio.get_event_loop()
    
    async def test_coro():
        return "test_result"
    
    result = await mock_context.run_async_task(test_coro())
    assert result == "test_result"

@pytest.mark.asyncio
async def test_plugin_event_handler_registration():
    """Test event handler registration in plugin context"""
    mock_context = Mock()
    mock_context.event_handlers = {}
    
    def test_handler():
        pass
    
    mock_context.register_event_handler("test_event", test_handler)
    assert "test_event" in mock_context.event_handlers
    assert test_handler in mock_context.event_handlers["test_event"]

@pytest.mark.asyncio
async def test_plugin_context_event_triggering():
    """Test event triggering in plugin context"""
    mock_context = Mock()
    mock_context.event_handlers = {"test_event": []}
    events_triggered = []
    
    async def test_handler():
        events_triggered.append("test_event")
    
    mock_context.event_handlers["test_event"].append(test_handler)
    
    await mock_context.trigger_event("test_event")
    assert "test_event" in events_triggered

@pytest.mark.asyncio
async def test_plugin_manager_multiple_plugins(tmp_path):
    """Test loading multiple plugins"""
    # Create multiple test plugins
    plugins = {
        "plugin1.py": """
class Plugin:
    def __init__(self, sdk):
        self.sdk = sdk
        self.name = "plugin1"
    async def on_load(self):
        pass
""",
        "plugin2.py": """
class Plugin:
    def __init__(self, sdk):
        self.sdk = sdk
        self.name = "plugin2"
    async def on_load(self):
        pass
""",
        "plugin3.py": """
class Plugin:
    def __init__(self, sdk):
        self.sdk = sdk
        self.name = "plugin3"
    async def on_load(self):
        pass
"""
    }
    
    for filename, code in plugins.items():
        plugin_file = tmp_path / filename
        plugin_file.write_text(code)
    
    pm = PluginManager(plugin_dir=str(tmp_path))
    await pm.load_all_plugins()
    
    assert len(pm.plugins) == 3
    assert "plugin1" in pm.plugins
    assert "plugin2" in pm.plugins
    assert "plugin3" in pm.plugins

@pytest.mark.asyncio
async def test_plugin_manager_broadcast_event(tmp_path):
    """Test broadcasting events to all plugins"""
    plugin_code = """
class Plugin:
    def __init__(self, sdk):
        self.sdk = sdk
        self.received_events = []
    async def on_event(self, event_name, *args, **kwargs):
        self.received_events.append(event_name)
"""
    
    # Create multiple plugins
    for i in range(3):
        plugin_file = tmp_path / f"plugin{i}.py"
        plugin_file.write_text(plugin_code)
    
    pm = PluginManager(plugin_dir=str(tmp_path))
    await pm.load_all_plugins()
    
    # Broadcast event
    await pm.broadcast_event("test_broadcast")
    
    # Check all plugins received the event
    for plugin in pm.plugins.values():
        assert "test_broadcast" in plugin.received_events

@pytest.mark.asyncio
async def test_plugin_manager_empty_directory():
    """Test plugin manager with empty directory"""
    with tempfile.TemporaryDirectory() as temp_dir:
        pm = PluginManager(plugin_dir=temp_dir)
        await pm.load_all_plugins()
        
        assert len(pm.plugins) == 0

@pytest.mark.asyncio
async def test_plugin_manager_nonexistent_directory():
    """Test plugin manager with nonexistent directory"""
    pm = PluginManager(plugin_dir="nonexistent_directory")
    await pm.load_all_plugins()
    
    # Should handle gracefully without error
    assert len(pm.plugins) == 0

if __name__ == "__main__":
    pytest.main([__file__])