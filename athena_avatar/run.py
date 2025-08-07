#!/usr/bin/env python3
"""
Athena 3D Avatar - Run Script
Simple execution script for the cosmic AI companion
"""

import sys
import os
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Main entry point for Athena 3D Avatar"""
    try:
        # Import and run the main application
        from main import AthenaApp
        
        # Create and run the application
        app = AthenaApp()
        
        if app.initialize():
            print("🌟 Athena 3D Avatar - Cosmic AI Companion")
            print("🚀 Initializing cosmic experience...")
            print("✨ Optimized for 12GB RAM with <250ms latency")
            print("🎭 20+ animations, 12+ voice tones, divine emotions")
            print("🎨 Advanced 3D rendering with NeRF technology")
            print("📊 Real-time performance monitoring")
            print("🎮 Ready for cosmic interaction!")
            print()
            
            # Run the application
            exit_code = app.run()
            
            print("🌟 Athena session completed. Farewell, cosmic traveler!")
            return exit_code
        else:
            print("❌ Failed to initialize Athena application")
            return 1
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")
        return 1
        
    except Exception as e:
        print(f"❌ Critical error: {e}")
        print("💡 Check the logs for more details")
        return 1

if __name__ == "__main__":
    sys.exit(main())