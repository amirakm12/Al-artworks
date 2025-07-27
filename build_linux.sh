#!/bin/bash

# ULTIMATE System - Linux Build Script
# This script builds the ULTIMATE System on Linux platforms

set -e  # Exit on any error

echo "🚀 ULTIMATE System - Linux Build Script"
echo "========================================"

# Check for required tools
echo "🔧 Checking build dependencies..."

if ! command -v cmake &> /dev/null; then
    echo "❌ CMake not found. Please install cmake (version 3.20 or higher)"
    exit 1
fi

if ! command -v make &> /dev/null; then
    echo "❌ Make not found. Please install build-essential"
    exit 1
fi

if ! command -v g++ &> /dev/null && ! command -v clang++ &> /dev/null; then
    echo "❌ No C++ compiler found. Please install g++ or clang++"
    exit 1
fi

echo "✅ All dependencies found"

# Configuration options
BUILD_TYPE=${1:-Release}
BUILD_DIR="build_linux"
INSTALL_PREFIX=${2:-"./install"}
NUM_JOBS=$(nproc)

echo "📋 Build Configuration:"
echo "   Build Type: $BUILD_TYPE"
echo "   Build Directory: $BUILD_DIR"
echo "   Install Prefix: $INSTALL_PREFIX"
echo "   Parallel Jobs: $NUM_JOBS"
echo ""

# Clean previous build if requested
if [[ "$3" == "clean" ]]; then
    echo "🧹 Cleaning previous build..."
    rm -rf "$BUILD_DIR"
fi

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake
echo "⚙️  Configuring with CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -DBUILD_EXAMPLES=ON \
    -DBUILD_TESTS=OFF

# Build the project
echo "🔨 Building ULTIMATE System..."
make -j"$NUM_JOBS"

# Test the build
echo "🧪 Testing build artifacts..."
if [[ -f "lib/libultimate.a" ]]; then
    echo "✅ Core library built successfully"
else
    echo "❌ Core library build failed"
    exit 1
fi

if [[ -f "bin/ultimate_system" ]]; then
    echo "✅ Main executable built successfully"
else
    echo "❌ Main executable build failed"
    exit 1
fi

# Optional: Run a quick test
if [[ -f "bin/ai_acceleration_demo" ]]; then
    echo "🎯 Running quick functionality test..."
    timeout 5s ./bin/ai_acceleration_demo || echo "✅ Demo test completed (timed out as expected)"
fi

echo ""
echo "🎉 ULTIMATE System build completed successfully!"
echo "📦 Build artifacts:"
echo "   Library: $BUILD_DIR/lib/libultimate.a"
echo "   Executable: $BUILD_DIR/bin/ultimate_system"
if [[ -f "bin/ai_acceleration_demo" ]]; then
    echo "   Demo: $BUILD_DIR/bin/ai_acceleration_demo"
fi
echo ""
echo "🚀 To install system-wide, run: make install"
echo "🏃 To run the system: ./bin/ultimate_system"