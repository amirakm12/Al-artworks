"""
Agent Orchestrator - Core AI Agent Management System
Manages LLM interactions, tool execution, and coordination between components
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import subprocess
import sys

# LLM and AI imports
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logging.warning("Ollama not available - using fallback LLM")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available")

# Tool imports
from tools.code_executor import CodeExecutor
from tools.web_browser import WebBrowser
from tools.image_editor import ImageEditor
from tools.voice_agent import VoiceAgent

class AgentOrchestrator:
    """Main agent orchestrator for ChatGPT+ Clone"""
    
    def __init__(self, memory_manager):
        self.memory_manager = memory_manager
        self.active_tools = {}
        self.conversation_history = []
        self.current_model = "dolphin-mixtral:8x22b"
        
        # Initialize tools
        self.tools = {
            "code_interpreter": CodeExecutor(),
            "web_browser": WebBrowser(),
            "image_editor": ImageEditor(),
            "voice_agent": VoiceAgent(),
        }
        
        # Initialize LLM
        self.initialize_llm()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def initialize_llm(self):
        """Initialize the language model"""
        try:
            if OLLAMA_AVAILABLE:
                self.llm_client = ollama.Client()
                self.llm_type = "ollama"
                self.logger.info("Using Ollama LLM")
            elif TRANSFORMERS_AVAILABLE:
                self.llm_type = "transformers"
                self.logger.info("Using Transformers LLM")
            else:
                self.llm_type = "mock"
                self.logger.warning("Using mock LLM - no real AI capabilities")
        except Exception as e:
            self.logger.error(f"Error initializing LLM: {e}")
            self.llm_type = "mock"
    
    def process_message(self, message: str) -> str:
        """Process a user message and return response"""
        try:
            # Add to conversation history
            self.conversation_history.append({"role": "user", "content": message})
            
            # Analyze message for tool usage
            tool_analysis = self.analyze_message_for_tools(message)
            
            if tool_analysis["needs_tool"]:
                # Execute tool and get result
                tool_result = self.execute_tool(tool_analysis["tool"], tool_analysis["args"])
                
                # Generate response with tool result
                response = self.generate_response_with_tool(message, tool_result)
            else:
                # Generate direct response
                response = self.generate_response(message)
            
            # Add response to history
            self.conversation_history.append({"role": "assistant", "content": response})
            
            # Save to memory
            self.memory_manager.add_interaction(message, response)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            return f"I encountered an error processing your message: {str(e)}"
    
    def analyze_message_for_tools(self, message: str) -> Dict[str, Any]:
        """Analyze message to determine if tools are needed"""
        message_lower = message.lower()
        
        # Simple keyword-based tool detection
        tool_analysis = {
            "needs_tool": False,
            "tool": None,
            "args": {}
        }
        
        # Code execution detection
        if any(keyword in message_lower for keyword in ["run code", "execute", "python", "script", "program"]):
            tool_analysis["needs_tool"] = True
            tool_analysis["tool"] = "code_interpreter"
            tool_analysis["args"] = {"code": self.extract_code(message)}
        
        # Web search detection
        elif any(keyword in message_lower for keyword in ["search", "find", "look up", "web", "browse"]):
            tool_analysis["needs_tool"] = True
            tool_analysis["tool"] = "web_browser"
            tool_analysis["args"] = {"query": message}
        
        # Image generation/editing detection
        elif any(keyword in message_lower for keyword in ["generate image", "create image", "edit image", "draw", "picture"]):
            tool_analysis["needs_tool"] = True
            tool_analysis["tool"] = "image_editor"
            tool_analysis["args"] = {"prompt": message}
        
        # Voice interaction detection
        elif any(keyword in message_lower for keyword in ["voice", "speak", "listen", "audio"]):
            tool_analysis["needs_tool"] = True
            tool_analysis["tool"] = "voice_agent"
            tool_analysis["args"] = {"action": "voice_interaction"}
        
        return tool_analysis
    
    def extract_code(self, message: str) -> str:
        """Extract code from message"""
        # Simple code extraction - look for code blocks
        if "```" in message:
            start = message.find("```") + 3
            end = message.rfind("```")
            if end > start:
                return message[start:end].strip()
        
        # Look for Python keywords to identify code
        python_keywords = ["def ", "import ", "from ", "class ", "if __name__", "print(", "for ", "while "]
        for keyword in python_keywords:
            if keyword in message:
                # Extract from keyword to end or next sentence
                start = message.find(keyword)
                return message[start:].strip()
        
        return message
    
    def execute_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Execute a specific tool"""
        try:
            if tool_name in self.tools:
                tool = self.tools[tool_name]
                result = tool.execute(args)
                return result
            else:
                return f"Tool '{tool_name}' not found"
        except Exception as e:
            self.logger.error(f"Error executing tool {tool_name}: {e}")
            return f"Error executing tool: {str(e)}"
    
    def generate_response(self, message: str) -> str:
        """Generate a response using the LLM"""
        try:
            if self.llm_type == "ollama":
                return self.generate_ollama_response(message)
            elif self.llm_type == "transformers":
                return self.generate_transformers_response(message)
            else:
                return self.generate_mock_response(message)
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return "I'm having trouble generating a response right now."
    
    def generate_ollama_response(self, message: str) -> str:
        """Generate response using Ollama"""
        try:
            # Prepare conversation context
            messages = []
            for msg in self.conversation_history[-10:]:  # Last 10 messages for context
                messages.append(msg)
            
            # Add system prompt
            system_prompt = """You are ChatGPT+, an advanced AI assistant with access to various tools including:
            - Code interpreter for Python execution
            - Web browser for real-time search
            - Image generation and editing
            - Voice interaction capabilities
            
            Be helpful, accurate, and use tools when appropriate."""
            
            messages.insert(0, {"role": "system", "content": system_prompt})
            
            # Generate response
            response = self.llm_client.chat(
                model=self.current_model,
                messages=messages
            )
            
            return response['message']['content']
            
        except Exception as e:
            self.logger.error(f"Ollama error: {e}")
            return "I'm having trouble connecting to the language model."
    
    def generate_transformers_response(self, message: str) -> str:
        """Generate response using Transformers (fallback)"""
        # This would use a local transformer model
        return f"I understand you said: '{message}'. This is a mock response from the transformers backend."
    
    def generate_mock_response(self, message: str) -> str:
        """Generate mock response when no LLM is available"""
        return f"I understand you said: '{message}'. This is a mock response - please install Ollama or Transformers for full AI capabilities."
    
    def generate_response_with_tool(self, message: str, tool_result: str) -> str:
        """Generate response incorporating tool results"""
        try:
            if self.llm_type == "ollama":
                # Create context with tool result
                context = f"""
                User message: {message}
                Tool result: {tool_result}
                
                Please provide a helpful response that incorporates the tool result.
                """
                
                response = self.llm_client.chat(
                    model=self.current_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful AI assistant. Use the tool result to provide a comprehensive response."},
                        {"role": "user", "content": context}
                    ]
                )
                
                return response['message']['content']
            else:
                return f"Tool executed successfully. Result: {tool_result}"
                
        except Exception as e:
            self.logger.error(f"Error generating response with tool: {e}")
            return f"Tool executed with result: {tool_result}"
    
    def activate_tool(self, tool_name: str):
        """Activate a specific tool"""
        if tool_name in self.tools:
            self.active_tools[tool_name] = self.tools[tool_name]
            self.logger.info(f"Activated tool: {tool_name}")
        else:
            self.logger.warning(f"Tool not found: {tool_name}")
    
    def deactivate_tool(self, tool_name: str):
        """Deactivate a specific tool"""
        if tool_name in self.active_tools:
            del self.active_tools[tool_name]
            self.logger.info(f"Deactivated tool: {tool_name}")
    
    def process_file(self, file_path: str) -> str:
        """Process an uploaded file"""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return f"File not found: {file_path}"
            
            # Determine file type and process accordingly
            if file_path.suffix.lower() in ['.py', '.js', '.java', '.cpp', '.c']:
                # Code file - use code interpreter
                with open(file_path, 'r') as f:
                    code = f.read()
                return self.tools["code_interpreter"].execute({"code": code})
            
            elif file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                # Image file - use image editor
                return self.tools["image_editor"].execute({"image_path": str(file_path)})
            
            elif file_path.suffix.lower() in ['.txt', '.md', '.json', '.csv']:
                # Text file - read and summarize
                with open(file_path, 'r') as f:
                    content = f.read()
                return f"File processed: {file_path.name}\nContent preview: {content[:200]}..."
            
            else:
                return f"Unsupported file type: {file_path.suffix}"
                
        except Exception as e:
            self.logger.error(f"Error processing file: {e}")
            return f"Error processing file: {str(e)}"
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tools"""
        return list(self.tools.keys())
    
    def get_active_tools(self) -> List[str]:
        """Get list of currently active tools"""
        return list(self.active_tools.keys())
    
    def set_model(self, model_name: str):
        """Set the current LLM model"""
        self.current_model = model_name
        self.logger.info(f"Model changed to: {model_name}")
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history"""
        return self.conversation_history.copy()
    
    def clear_conversation(self):
        """Clear conversation history"""
        self.conversation_history.clear()
        self.logger.info("Conversation history cleared")