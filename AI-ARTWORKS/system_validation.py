#!/usr/bin/env python3
"""
AI-ARTWORK System Validation Script
Comprehensive validation of all system components, dependencies, and models
"""

import os
import sys
import asyncio
import importlib
from pathlib import Path
from typing import Dict, List, Any, Optional
import traceback
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

class SystemValidator:
    """Comprehensive system validation"""
    
    def __init__(self):
        self.results = {}
        self.errors = []
        self.warnings = []
        
    def log_result(self, test_name: str, status: str, details: str = ""):
        """Log a test result"""
        self.results[test_name] = {
            "status": status,
            "details": details
        }
        print(f"[{status}] {test_name}: {details}")
        
    def log_error(self, test_name: str, error: str):
        """Log an error"""
        self.errors.append(f"{test_name}: {error}")
        self.log_result(test_name, "FAILED", error)
        
    def log_warning(self, test_name: str, warning: str):
        """Log a warning"""
        self.warnings.append(f"{test_name}: {warning}")
        self.log_result(test_name, "WARNING", warning)
        
    def test_python_environment(self):
        """Test Python environment and basic imports"""
        try:
            import sys
            python_version = sys.version
            self.log_result("Python Version", "PASSED", f"Python {python_version}")
            
            # Test critical imports
            critical_imports = [
                "numpy", "torch", "PIL", "cv2", "transformers", 
                "diffusers", "loguru", "asyncio", "pathlib"
            ]
            
            for module in critical_imports:
                try:
                    importlib.import_module(module)
                    self.log_result(f"Import {module}", "PASSED", "")
                except ImportError as e:
                    self.log_error(f"Import {module}", str(e))
                    
        except Exception as e:
            self.log_error("Python Environment", str(e))
            
    def test_directory_structure(self):
        """Test directory structure"""
        required_dirs = [
            "src", "models", "cache", "logs", "outputs", "temp",
            "src/core", "src/agents", "src/ui", "src/voice", "src/plugins"
        ]
        
        for dir_path in required_dirs:
            path = Path(dir_path)
            if path.exists():
                self.log_result(f"Directory {dir_path}", "PASSED", "")
            else:
                self.log_error(f"Directory {dir_path}", "Missing directory")
                
    def test_model_files(self):
        """Test model files"""
        models_dir = Path("models")
        if not models_dir.exists():
            self.log_error("Models Directory", "Models directory does not exist")
            return
            
        # Check model index
        index_file = models_dir / "model_index.json"
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    index_data = json.load(f)
                self.log_result("Model Index", "PASSED", f"Found {len(index_data.get('models', {}))} model categories")
            except Exception as e:
                self.log_error("Model Index", f"Failed to read index: {e}")
        else:
            self.log_warning("Model Index", "Model index not found")
            
        # Check for downloaded models
        model_files = list(models_dir.rglob("*.pth")) + list(models_dir.rglob("*.safetensors")) + list(models_dir.rglob("*.gguf"))
        if model_files:
            self.log_result("Model Files", "PASSED", f"Found {len(model_files)} model files")
        else:
            self.log_warning("Model Files", "No model files found")
            
    def test_src_imports(self):
        """Test src module imports"""
        try:
            # Test main module import
            from src import AI_ARTWORK, AlArtworks
            self.log_result("Main Module Import", "PASSED", "")
            
            # Test core modules
            core_modules = [
                "src.core.config",
                "src.core.gpu_utils", 
                "src.core.model_manager",
                "src.core.model_zoo"
            ]
            
            for module in core_modules:
                try:
                    importlib.import_module(module)
                    self.log_result(f"Core Module {module}", "PASSED", "")
                except ImportError as e:
                    self.log_error(f"Core Module {module}", str(e))
                    
        except Exception as e:
            self.log_error("Main Module Import", str(e))
            
    def test_agent_imports(self):
        """Test agent module imports"""
        agent_modules = [
            "src.agents.base_agent",
            "src.agents.orchestrator", 
            "src.agents.image_restoration",
            "src.agents.style_aesthetic",
            "src.agents.semantic_editing"
        ]
        
        for module in agent_modules:
            try:
                importlib.import_module(module)
                self.log_result(f"Agent {module}", "PASSED", "")
            except ImportError as e:
                self.log_error(f"Agent {module}", str(e))
                
    def test_ui_imports(self):
        """Test UI module imports (optional)"""
        try:
            import PySide6
            self.log_result("PySide6 Available", "PASSED", "")
            
            try:
                from src.ui.main_window import MainWindow
                self.log_result("Main Window Import", "PASSED", "")
            except ImportError as e:
                self.log_error("Main Window Import", str(e))
                
        except ImportError:
            self.log_warning("PySide6 Available", "PySide6 not available - GUI disabled")
            
    async def test_async_functionality(self):
        """Test async functionality"""
        try:
            # Test basic async operation
            await asyncio.sleep(0.1)
            self.log_result("Async Basic", "PASSED", "")
            
            # Test AI_ARTWORK initialization
            try:
                from src import AI_ARTWORK
                studio = AI_ARTWORK()
                self.log_result("AI_ARTWORK Creation", "PASSED", "")
                
                # Test initialization (with timeout)
                try:
                    await asyncio.wait_for(studio.initialize(), timeout=30.0)
                    self.log_result("AI_ARTWORK Initialize", "PASSED", "")
                except asyncio.TimeoutError:
                    self.log_warning("AI_ARTWORK Initialize", "Initialization timed out")
                except Exception as e:
                    self.log_error("AI_ARTWORK Initialize", str(e))
                    
            except Exception as e:
                self.log_error("AI_ARTWORK Creation", str(e))
                
        except Exception as e:
            self.log_error("Async Basic", str(e))
            
    def test_gpu_availability(self):
        """Test GPU availability"""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
                self.log_result("GPU Available", "PASSED", f"{gpu_count} GPUs, Primary: {gpu_name}")
            else:
                self.log_warning("GPU Available", "No CUDA GPUs available - using CPU")
                
        except Exception as e:
            self.log_error("GPU Available", str(e))
            
    def test_model_loading(self):
        """Test basic model loading"""
        try:
            import torch
            from transformers import AutoTokenizer
            
            # Test a simple model load
            try:
                tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", cache_dir="./cache")
                self.log_result("Model Loading Test", "PASSED", "Successfully loaded test tokenizer")
            except Exception as e:
                self.log_error("Model Loading Test", str(e))
                
        except Exception as e:
            self.log_error("Model Loading Test", str(e))
            
    def generate_report(self):
        """Generate validation report"""
        print("\n" + "="*80)
        print("SYSTEM VALIDATION REPORT")
        print("="*80)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r["status"] == "PASSED")
        failed_tests = sum(1 for r in self.results.values() if r["status"] == "FAILED")
        warning_tests = sum(1 for r in self.results.values() if r["status"] == "WARNING")
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Warnings: {warning_tests}")
        
        if failed_tests > 0:
            print(f"\nFAILED TESTS ({failed_tests}):")
            for test_name, result in self.results.items():
                if result["status"] == "FAILED":
                    print(f"  - {test_name}: {result['details']}")
                    
        if warning_tests > 0:
            print(f"\nWARNINGS ({warning_tests}):")
            for test_name, result in self.results.items():
                if result["status"] == "WARNING":
                    print(f"  - {test_name}: {result['details']}")
                    
        # Overall status
        if failed_tests == 0:
            if warning_tests == 0:
                print(f"\n✅ SYSTEM STATUS: FULLY OPERATIONAL")
            else:
                print(f"\n⚠️  SYSTEM STATUS: OPERATIONAL WITH WARNINGS")
        else:
            print(f"\n❌ SYSTEM STATUS: ISSUES DETECTED - {failed_tests} CRITICAL FAILURES")
            
        # Save report
        report_data = {
            "timestamp": str(Path().cwd()),
            "summary": {
                "total": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "warnings": warning_tests
            },
            "results": self.results,
            "errors": self.errors,
            "warnings": self.warnings
        }
        
        with open("system_validation_report.json", "w") as f:
            json.dump(report_data, f, indent=2)
            
        print(f"\nDetailed report saved to: system_validation_report.json")
        
        return failed_tests == 0

async def main():
    """Main validation function"""
    print("AI-ARTWORK System Validation Starting...")
    print("="*80)
    
    validator = SystemValidator()
    
    # Run all validation tests
    validator.test_python_environment()
    validator.test_directory_structure()
    validator.test_model_files()
    validator.test_src_imports()
    validator.test_agent_imports()
    validator.test_ui_imports()
    validator.test_gpu_availability()
    validator.test_model_loading()
    
    # Run async tests
    await validator.test_async_functionality()
    
    # Generate report
    success = validator.generate_report()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)