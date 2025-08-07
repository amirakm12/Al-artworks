#!/usr/bin/env python3
"""
Enhanced Test Runner for ChatGPT+ Clone
Runs all tests with comprehensive reporting and configuration
"""

import os
import sys
import subprocess
import argparse
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import pytest
import coverage

class TestRunner:
    """Comprehensive test runner with reporting and configuration"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.results = {}
        self.start_time = None
        self.end_time = None
        
        # Ensure we're in the project root
        self.project_root = Path(__file__).parent.parent
        os.chdir(self.project_root)
        
        # Setup paths
        self.tests_dir = self.project_root / "tests"
        self.coverage_dir = self.project_root / "htmlcov"
        self.results_dir = self.project_root / "test_results"
        
        # Create directories
        self.coverage_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
    
    def _default_config(self) -> Dict[str, Any]:
        """Default test configuration"""
        return {
            "test_patterns": [
                "tests/test_*.py",
                "tests/*/test_*.py"
            ],
            "exclude_patterns": [
                "tests/__pycache__/*",
                "tests/*/__pycache__/*"
            ],
            "coverage": {
                "enabled": True,
                "source": ["."],
                "omit": [
                    "*/tests/*",
                    "*/venv/*",
                    "*/env/*",
                    "*/__pycache__/*",
                    "*/build/*",
                    "*/dist/*"
                ]
            },
            "pytest": {
                "verbose": True,
                "tb": "short",
                "maxfail": 5,
                "timeout": 300
            },
            "reporting": {
                "html": True,
                "xml": True,
                "json": True
            }
        }
    
    def discover_tests(self) -> List[str]:
        """Discover all test files"""
        test_files = []
        
        for pattern in self.config["test_patterns"]:
            for test_file in self.project_root.glob(pattern):
                if test_file.is_file() and test_file.suffix == ".py":
                    # Check if file should be excluded
                    excluded = False
                    for exclude_pattern in self.config["exclude_patterns"]:
                        if exclude_pattern in str(test_file):
                            excluded = True
                            break
                    
                    if not excluded:
                        test_files.append(str(test_file))
        
        return sorted(test_files)
    
    def run_unit_tests(self) -> Dict[str, Any]:
        """Run unit tests with pytest"""
        print("🔍 Running unit tests...")
        
        test_files = self.discover_tests()
        if not test_files:
            print("⚠️  No test files found!")
            return {"status": "no_tests", "files": []}
        
        print(f"📁 Found {len(test_files)} test files:")
        for test_file in test_files:
            print(f"   - {test_file}")
        
        # Prepare pytest arguments
        pytest_args = [
            "pytest",
            "--verbose",
            "--tb=short",
            f"--maxfail={self.config['pytest']['maxfail']}",
            "--timeout=300",
            "--junit-xml=test_results/junit.xml",
            "--html=test_results/report.html",
            "--self-contained-html"
        ]
        
        # Add test files
        pytest_args.extend(test_files)
        
        # Run pytest
        try:
            result = subprocess.run(
                pytest_args,
                capture_output=True,
                text=True,
                timeout=self.config["pytest"]["timeout"]
            )
            
            return {
                "status": "success" if result.returncode == 0 else "failed",
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "files": test_files
            }
            
        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "returncode": -1,
                "stdout": "",
                "stderr": "Test execution timed out",
                "files": test_files
            }
        except Exception as e:
            return {
                "status": "error",
                "returncode": -1,
                "stdout": "",
                "stderr": str(e),
                "files": test_files
            }
    
    def run_coverage(self) -> Dict[str, Any]:
        """Run tests with coverage analysis"""
        if not self.config["coverage"]["enabled"]:
            return {"status": "disabled"}
        
        print("📊 Running coverage analysis...")
        
        # Start coverage
        cov = coverage.Coverage(
            source=self.config["coverage"]["source"],
            omit=self.config["coverage"]["omit"]
        )
        cov.start()
        
        # Run tests
        test_result = self.run_unit_tests()
        
        # Stop coverage and generate report
        cov.stop()
        cov.save()
        
        # Generate reports
        cov.html_report(directory=str(self.coverage_dir))
        cov.xml_report(outfile=str(self.results_dir / "coverage.xml"))
        
        # Get coverage statistics
        total = cov.report()
        
        return {
            "status": "success",
            "coverage_percentage": total,
            "test_result": test_result,
            "html_report": str(self.coverage_dir / "index.html"),
            "xml_report": str(self.results_dir / "coverage.xml")
        }
    
    def run_linting(self) -> Dict[str, Any]:
        """Run code linting checks"""
        print("🔍 Running linting checks...")
        
        lint_results = {}
        
        # Run flake8
        try:
            result = subprocess.run(
                ["flake8", ".", "--count", "--select=E9,F63,F7,F82", "--show-source", "--statistics"],
                capture_output=True,
                text=True
            )
            lint_results["flake8_critical"] = {
                "status": "success" if result.returncode == 0 else "failed",
                "output": result.stdout,
                "errors": result.stderr
            }
        except Exception as e:
            lint_results["flake8_critical"] = {
                "status": "error",
                "output": "",
                "errors": str(e)
            }
        
        # Run flake8 with warnings
        try:
            result = subprocess.run(
                ["flake8", ".", "--count", "--exit-zero", "--max-complexity=10", "--max-line-length=127", "--statistics"],
                capture_output=True,
                text=True
            )
            lint_results["flake8_warnings"] = {
                "status": "success",
                "output": result.stdout,
                "errors": result.stderr
            }
        except Exception as e:
            lint_results["flake8_warnings"] = {
                "status": "error",
                "output": "",
                "errors": str(e)
            }
        
        # Run black check
        try:
            result = subprocess.run(
                ["black", "--check", "--diff", "."],
                capture_output=True,
                text=True
            )
            lint_results["black"] = {
                "status": "success" if result.returncode == 0 else "failed",
                "output": result.stdout,
                "errors": result.stderr
            }
        except Exception as e:
            lint_results["black"] = {
                "status": "error",
                "output": "",
                "errors": str(e)
            }
        
        # Run isort check
        try:
            result = subprocess.run(
                ["isort", "--check-only", "--diff", "."],
                capture_output=True,
                text=True
            )
            lint_results["isort"] = {
                "status": "success" if result.returncode == 0 else "failed",
                "output": result.stdout,
                "errors": result.stderr
            }
        except Exception as e:
            lint_results["isort"] = {
                "status": "error",
                "output": "",
                "errors": str(e)
            }
        
        return lint_results
    
    def run_type_checking(self) -> Dict[str, Any]:
        """Run type checking with mypy"""
        print("🔍 Running type checking...")
        
        try:
            result = subprocess.run(
                ["mypy", "--ignore-missing-imports", "."],
                capture_output=True,
                text=True
            )
            
            return {
                "status": "success" if result.returncode == 0 else "failed",
                "output": result.stdout,
                "errors": result.stderr
            }
            
        except Exception as e:
            return {
                "status": "error",
                "output": "",
                "errors": str(e)
            }
    
    def run_security_checks(self) -> Dict[str, Any]:
        """Run security checks"""
        print("🔒 Running security checks...")
        
        security_results = {}
        
        # Run bandit
        try:
            result = subprocess.run(
                ["bandit", "-r", ".", "-f", "json", "-o", "test_results/bandit-report.json"],
                capture_output=True,
                text=True
            )
            security_results["bandit"] = {
                "status": "success",
                "output": result.stdout,
                "errors": result.stderr
            }
        except Exception as e:
            security_results["bandit"] = {
                "status": "error",
                "output": "",
                "errors": str(e)
            }
        
        # Run safety check
        try:
            result = subprocess.run(
                ["safety", "check", "--json", "--output", "test_results/safety-report.json"],
                capture_output=True,
                text=True
            )
            security_results["safety"] = {
                "status": "success",
                "output": result.stdout,
                "errors": result.stderr
            }
        except Exception as e:
            security_results["safety"] = {
                "status": "error",
                "output": "",
                "errors": str(e)
            }
        
        return security_results
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        report = {
            "timestamp": time.time(),
            "duration": self.end_time - self.start_time if self.end_time else 0,
            "results": self.results,
            "summary": {}
        }
        
        # Generate summary
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        
        if "unit_tests" in self.results:
            unit_result = self.results["unit_tests"]
            if unit_result["status"] == "success":
                # Parse pytest output to count tests
                lines = unit_result["stdout"].split("\n")
                for line in lines:
                    if "passed" in line and "failed" in line:
                        # Extract numbers from line like "5 passed, 2 failed"
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part.isdigit() and i + 1 < len(parts):
                                if parts[i + 1] == "passed":
                                    passed_tests = int(part)
                                elif parts[i + 1] == "failed":
                                    failed_tests = int(part)
                        break
        
        total_tests = passed_tests + failed_tests
        
        report["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            "overall_status": "success" if failed_tests == 0 else "failed"
        }
        
        return report
    
    def save_report(self, report: Dict[str, Any], filename: str = "test_report.json"):
        """Save test report to file"""
        report_file = self.results_dir / filename
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"📄 Test report saved to: {report_file}")
    
    def print_summary(self, report: Dict[str, Any]):
        """Print test summary"""
        summary = report["summary"]
        
        print("\n" + "="*60)
        print("📊 TEST SUMMARY")
        print("="*60)
        print(f"⏱️  Duration: {report['duration']:.2f} seconds")
        print(f"🧪 Total Tests: {summary['total_tests']}")
        print(f"✅ Passed: {summary['passed_tests']}")
        print(f"❌ Failed: {summary['failed_tests']}")
        print(f"📈 Success Rate: {summary['success_rate']:.1f}%")
        print(f"🎯 Overall Status: {summary['overall_status'].upper()}")
        
        if "coverage" in self.results and self.results["coverage"]["status"] == "success":
            coverage_pct = self.results["coverage"]["coverage_percentage"]
            print(f"📊 Coverage: {coverage_pct:.1f}%")
        
        print("="*60)
        
        # Print detailed results
        for test_type, result in self.results.items():
            status_emoji = "✅" if result.get("status") == "success" else "❌"
            print(f"{status_emoji} {test_type.replace('_', ' ').title()}: {result.get('status', 'unknown')}")
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and generate comprehensive report"""
        print("🚀 Starting comprehensive test suite...")
        self.start_time = time.time()
        
        # Run different types of tests
        self.results["unit_tests"] = self.run_unit_tests()
        self.results["coverage"] = self.run_coverage()
        self.results["linting"] = self.run_linting()
        self.results["type_checking"] = self.run_type_checking()
        self.results["security"] = self.run_security_checks()
        
        self.end_time = time.time()
        
        # Generate and save report
        report = self.generate_report()
        self.save_report(report)
        
        # Print summary
        self.print_summary(report)
        
        return report

def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(description="Enhanced Test Runner for ChatGPT+ Clone")
    parser.add_argument("--config", help="Path to test configuration file")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage analysis")
    parser.add_argument("--linting", action="store_true", help="Run linting checks only")
    parser.add_argument("--type-checking", action="store_true", help="Run type checking only")
    parser.add_argument("--security", action="store_true", help="Run security checks only")
    parser.add_argument("--unit-tests", action="store_true", help="Run unit tests only")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--output", help="Output file for results")
    
    args = parser.parse_args()
    
    # Load configuration
    config = None
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Create test runner
    runner = TestRunner(config)
    
    # Run specific tests based on arguments
    if args.linting:
        print("🔍 Running linting checks only...")
        results = runner.run_linting()
        print("Linting results:", results)
    elif args.type_checking:
        print("🔍 Running type checking only...")
        results = runner.run_type_checking()
        print("Type checking results:", results)
    elif args.security:
        print("🔒 Running security checks only...")
        results = runner.run_security_checks()
        print("Security results:", results)
    elif args.unit_tests:
        print("🧪 Running unit tests only...")
        results = runner.run_unit_tests()
        print("Unit test results:", results)
    else:
        # Run all tests
        report = runner.run_all_tests()
        
        # Exit with appropriate code
        if report["summary"]["overall_status"] == "success":
            sys.exit(0)
        else:
            sys.exit(1)

if __name__ == "__main__":
    main()