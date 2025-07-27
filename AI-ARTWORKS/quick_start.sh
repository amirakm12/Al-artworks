#!/bin/bash

# AI-ARTWORK Quick Start Script
# Automatically activates environment and launches the system

echo "🚀 AI-ARTWORK Quick Start"
echo "=========================="

# Check if we're in the right directory
if [ ! -f "launch.py" ]; then
    echo "❌ Error: Please run this script from the AI-ARTWORKS directory"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "../venv" ]; then
    echo "❌ Error: Virtual environment not found at ../venv"
    echo "Please run the system validation first to set up the environment"
    exit 1
fi

echo "✅ Activating virtual environment..."
source ../venv/bin/activate

echo "✅ Environment activated"
echo ""

# Ask user which mode they want
echo "Choose launch mode:"
echo "1) CLI (Command Line Interface)"
echo "2) GUI (Graphical User Interface)"
echo "3) Validation (Run system tests)"
echo ""
read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        echo "🖥️  Launching AI-ARTWORK CLI..."
        echo "Available commands:"
        echo "  - edit <image_path> <instruction>"
        echo "  - generate <prompt>"
        echo "  - 3d <image_path>"
        echo "  - quit/exit/q"
        echo ""
        python launch.py cli
        ;;
    2)
        echo "🖼️  Launching AI-ARTWORK GUI..."
        python launch.py gui
        ;;
    3)
        echo "🔍 Running system validation..."
        python system_validation.py
        ;;
    *)
        echo "❌ Invalid choice. Please run the script again."
        exit 1
        ;;
esac

echo ""
echo "✅ AI-ARTWORK session completed"
echo "Thank you for using AI-ARTWORK!"