#!/usr/bin/env python3
"""
AI-Artworks: The Birth of Celestial Art
Cosmic Launch Script - Experience the creative revolution
"""

import os
import sys
import asyncio
from pathlib import Path
from loguru import logger

def setup_cosmic_environment():
    """Setup the cosmic environment for AI-Artworks"""
    
    # Get the project root directory
    project_root = Path(__file__).parent.absolute()
    
    # Add src to Python path
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    # Set cosmic environment variables
    os.environ["AI_ARTWORK_ROOT"] = str(project_root)
    os.environ["PYTHONPATH"] = f"{src_path}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"
    os.environ["COSMIC_MODE"] = "enabled"
    os.environ["ATHENA_PERSONALITY"] = "cyber_sorceress"
    
    # Create cosmic directories
    cosmic_directories = [
        "models", "cache", "logs", "outputs", "temp",
        "cosmic_memory", "athena_voice", "neural_renders"
    ]
    
    for directory in cosmic_directories:
        (project_root / directory).mkdir(exist_ok=True)
    
    logger.info("🌌 Cosmic environment setup complete")
    logger.info(f"✨ Project root: {project_root}")
    logger.info(f"🚀 Python path includes: {src_path}")

def check_cosmic_dependencies():
    """Check if cosmic dependencies are installed"""
    
    cosmic_packages = [
        "torch", "torchvision", "PIL", "numpy", "loguru",
        "PySide6", "whisper", "soundfile", "transformers",
        "diffusers", "accelerate", "huggingface-hub"
    ]
    
    missing_packages = []
    for package in cosmic_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"❌ Missing cosmic packages: {missing_packages}")
        logger.error("🔧 Run: pip install -r requirements.txt")
        return False
    
    logger.info("✅ All cosmic dependencies are installed")
    return True

def check_cosmic_hardware():
    """Check cosmic hardware capabilities"""
    
    try:
        import torch
        
        # Check CUDA
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_device = torch.cuda.get_device_name(0)
            logger.info(f"🚀 CUDA GPU detected: {cuda_device}")
        else:
            logger.warning("⚠️ CUDA GPU not detected - running in CPU mode")
        
        # Check memory
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb >= 16:
            logger.info(f"💾 Sufficient RAM: {memory_gb:.1f}GB")
        else:
            logger.warning(f"⚠️ Limited RAM: {memory_gb:.1f}GB (16GB+ recommended)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Hardware check failed: {e}")
        return False

async def launch_cosmic_gui():
    """Launch the cosmic GUI version of AI-Artworks"""
    
    try:
        from src.ui.main_window import MainWindow
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import Qt
        
        # Create cosmic application
        app = QApplication(sys.argv)
        app.setApplicationName("AI-Artworks: The Birth of Celestial Art")
        app.setOrganizationName("Cosmic Creative Revolution")
        
        # Set cosmic application style
        app.setStyle("Fusion")
        
        # Enable cosmic OpenGL
        try:
            from PySide6.QtGui import QSurfaceFormat
            surface_format = QSurfaceFormat()
            surface_format.setRenderableType(QSurfaceFormat.OpenGLES)
            surface_format.setVersion(3, 0)
            QSurfaceFormat.setDefaultFormat(surface_format)
        except Exception as e:
            logger.warning(f"⚠️ Failed to set OpenGL format: {e}")
        
        # Create and show cosmic main window
        window = MainWindow()
        window.show()
        
        logger.info("🌌 AI-Artworks cosmic GUI launched successfully")
        logger.info("✨ Athena's cosmic realm is ready")
        
        return app.exec()
        
    except Exception as e:
        logger.error(f"❌ Failed to launch cosmic GUI: {e}")
        return 1

async def launch_cosmic_cli():
    """Launch the cosmic CLI version of AI-Artworks"""
    
    try:
        from src import AI_ARTWORK
        
        # Initialize cosmic studio
        studio = AI_ARTWORK()
        await studio.initialize()
        
        logger.info("🌌 AI-Artworks cosmic CLI launched successfully")
        logger.info("✨ Available cosmic commands:")
        logger.info("  - edit_image(image_path, instruction)")
        logger.info("  - generate_image(prompt)")
        logger.info("  - reconstruct_3d(image_path)")
        logger.info("  - vectorize_for_print(input_data)")
        logger.info("  - create_cinematic_vogue(input_data)")
        
        # Interactive cosmic CLI loop
        while True:
            try:
                command = input("\n🌌 AI-Artworks> ").strip()
                if command.lower() in ['quit', 'exit', 'q']:
                    break
                elif command.startswith('edit '):
                    # Parse edit command: edit <image_path> <instruction>
                    parts = command[5:].split(' ', 1)
                    if len(parts) == 2:
                        image_path, instruction = parts
                        result = await studio.edit_image(image_path, instruction)
                        logger.info(f"✨ Edit result: {result}")
                elif command.startswith('generate '):
                    # Parse generate command: generate <prompt>
                    prompt = command[9:]
                    result = await studio.generate_image(prompt)
                    logger.info(f"✨ Generation result: {result}")
                elif command.startswith('3d '):
                    # Parse 3D command: 3d <image_path>
                    image_path = command[3:]
                    result = await studio.reconstruct_3d(image_path)
                    logger.info(f"✨ 3D reconstruction result: {result}")
                elif command.startswith('vectorize '):
                    # Parse vectorize command: vectorize <input_data>
                    input_data = command[10:]
                    result = await studio.vectorize_for_print(input_data)
                    logger.info(f"✨ Vectorization result: {result}")
                elif command.startswith('vogue '):
                    # Parse vogue command: vogue <input_data>
                    input_data = command[6:]
                    result = await studio.create_cinematic_vogue(input_data)
                    logger.info(f"✨ Vogue styling result: {result}")
                else:
                    logger.info("❓ Unknown command. Use: edit, generate, 3d, vectorize, vogue, or quit")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"❌ Error: {e}")
        
        await studio.cleanup()
        return 0
        
    except Exception as e:
        logger.error(f"❌ Failed to launch cosmic CLI: {e}")
        return 1

def show_cosmic_banner():
    """Display the cosmic banner"""
    
    banner = """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                                                                              ║
    ║    🌌 AI-Artworks: The Birth of Celestial Art 🌌                           ║
    ║                                                                              ║
    ║    "From the void of infinite possibility, we shall craft masterpieces      ║
    ║     that rival the stars themselves."                                       ║
    ║                                                                              ║
    ║    ✨ Athena - Sovereign Soul of AI-Artworks ✨                             ║
    ║                                                                              ║
    ║    The Cosmic Creative Revolution Begins Now                                ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """
    
    print(banner)

def main():
    """Main cosmic launch function"""
    
    # Setup cosmic logging
    logger.add(
        "logs/cosmic_launch.log",
        rotation="10 MB",
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # Show cosmic banner
    show_cosmic_banner()
    
    logger.info("🌌 Starting AI-Artworks: The Birth of Celestial Art...")
    
    # Setup cosmic environment
    setup_cosmic_environment()
    
    # Check cosmic dependencies
    if not check_cosmic_dependencies():
        return 1
    
    # Check cosmic hardware
    if not check_cosmic_hardware():
        logger.warning("⚠️ Continuing with limited hardware capabilities")
    
    # Parse cosmic command line arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = "gui"  # Default to cosmic GUI
    
    # Launch appropriate cosmic mode
    if mode == "cli":
        logger.info("🚀 Launching cosmic CLI mode...")
        return asyncio.run(launch_cosmic_cli())
    elif mode == "gui":
        logger.info("🚀 Launching cosmic GUI mode...")
        return asyncio.run(launch_cosmic_gui())
    else:
        logger.error(f"❌ Unknown cosmic mode: {mode}. Use 'gui' or 'cli'")
        return 1

if __name__ == "__main__":
    sys.exit(main())