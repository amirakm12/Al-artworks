cmake_minimum_required(VERSION 3.20)

# Project configuration
project(AI-Artworks 
    VERSION 1.0.0 
    LANGUAGES CXX C
    DESCRIPTION "AI-Powered Artwork Generation and Processing System"
)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build configuration
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Windows-specific settings
if(WIN32)
    add_definitions(-D_WIN32_WINNT=0x0A00)  # Windows 10+
    add_definitions(-DWIN32_LEAN_AND_MEAN)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_definitions(-DNOMINMAX)  # Prevent min/max macro conflicts
    
    # Set Windows subsystem
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /SUBSYSTEM:CONSOLE")
    
    # Enable large address awareness for 32-bit builds
    if(CMAKE_SIZEOF_VOID_P EQUAL 4)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LARGEADDRESSAWARE")
    endif()
endif()

# Compiler-specific optimizations
if(MSVC)
    # MSVC compiler flags
    add_compile_options(
        /W4                 # Warning level 4
        /permissive-        # Disable non-conforming code
        /Zc:__cplusplus     # Enable correct __cplusplus macro
        /MP                 # Multi-processor compilation
        /bigobj             # Large object files
        /wd4100             # Unreferenced parameter
        /wd4996             # Deprecated functions
    )
    
    # Optimization flags
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(
            /O2                 # Maximum optimization
            /Ob2                # Inline expansion
            /Oi                 # Intrinsic functions
            /Ot                 # Favor speed
            /GL                 # Whole program optimization
        )
        add_link_options(/LTCG)  # Link time code generation
    elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(/Od /Zi /RTC1)
        add_definitions(-DDEBUG=1)
    endif()
    
    # AVX2 support for performance
    add_compile_options(/arch:AVX2)
    
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # GCC/Clang compiler flags
    add_compile_options(
        -Wall -Wextra -Wpedantic
        -Wno-unused-parameter
        -mavx2 -mfma
    )
    
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3 -march=native -mtune=native)
    elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-O0 -g3)
        add_definitions(-DDEBUG=1)
    endif()
endif()

# Find required packages
find_package(PkgConfig QUIET)

# vcpkg integration
if(DEFINED CMAKE_TOOLCHAIN_FILE)
    message(STATUS "Using vcpkg toolchain: ${CMAKE_TOOLCHAIN_FILE}")
endif()

# Find packages with error handling
macro(find_package_with_fallback PACKAGE_NAME)
    find_package(${PACKAGE_NAME} ${ARGN} QUIET)
    if(NOT ${PACKAGE_NAME}_FOUND)
        message(STATUS "Package ${PACKAGE_NAME} not found via find_package, trying pkg-config...")
        if(PkgConfig_FOUND)
            pkg_check_modules(${PACKAGE_NAME} ${PACKAGE_NAME})
        endif()
        if(NOT ${PACKAGE_NAME}_FOUND)
            message(WARNING "Package ${PACKAGE_NAME} not found. Some features may be disabled.")
        endif()
    else()
        message(STATUS "Found ${PACKAGE_NAME}")
    endif()
endmacro()

# Core dependencies
find_package_with_fallback(fmt CONFIG)
find_package_with_fallback(spdlog CONFIG)
find_package_with_fallback(Eigen3 CONFIG)
find_package_with_fallback(benchmark CONFIG)
find_package_with_fallback(GTest CONFIG)

# Graphics dependencies
find_package_with_fallback(glfw3 CONFIG)
find_package(OpenGL QUIET)
find_package(Vulkan QUIET)

# Audio dependencies
find_package_with_fallback(portaudio CONFIG)

# Threading
find_package(Threads REQUIRED)

# OpenMP for parallel processing
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP")
else()
    message(STATUS "OpenMP not found - parallel processing will be limited")
endif()

# Intel TBB for advanced threading
find_package_with_fallback(TBB CONFIG)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/core
    ${CMAKE_CURRENT_SOURCE_DIR}/include/ai
    ${CMAKE_CURRENT_SOURCE_DIR}/include/graphics
    ${CMAKE_CURRENT_SOURCE_DIR}/include/audio
    ${CMAKE_CURRENT_SOURCE_DIR}/include/utils
)

# Source files organization
file(GLOB_RECURSE CORE_SOURCES
    "src/core/*.cpp"
    "src/core/*.c"
)

file(GLOB_RECURSE AI_SOURCES
    "src/ai/*.cpp"
    "src/ai/*.c"
)

file(GLOB_RECURSE GRAPHICS_SOURCES
    "src/graphics/*.cpp"
    "src/graphics/*.c"
)

file(GLOB_RECURSE AUDIO_SOURCES
    "src/audio/*.cpp"
    "src/audio/*.c"
)

file(GLOB_RECURSE UTILS_SOURCES
    "src/utils/*.cpp"
    "src/utils/*.c"
)

# Main executable
add_executable(ai-artworks
    src/main.cpp
    ${CORE_SOURCES}
    ${AI_SOURCES}
    ${GRAPHICS_SOURCES}
    ${AUDIO_SOURCES}
    ${UTILS_SOURCES}
)

# Set target properties
set_target_properties(ai-artworks PROPERTIES
    OUTPUT_NAME "ai-artworks"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Compile definitions
target_compile_definitions(ai-artworks PRIVATE
    AI_ARTWORKS_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
    AI_ARTWORKS_VERSION_MINOR=${PROJECT_VERSION_MINOR}
    AI_ARTWORKS_VERSION_PATCH=${PROJECT_VERSION_PATCH}
    AI_ARTWORKS_VERSION_STRING="${PROJECT_VERSION}"
)

# Link libraries with error handling
target_link_libraries(ai-artworks PRIVATE Threads::Threads)

# Core utilities
if(fmt_FOUND)
    target_link_libraries(ai-artworks PRIVATE fmt::fmt)
    target_compile_definitions(ai-artworks PRIVATE HAS_FMT=1)
endif()

if(spdlog_FOUND)
    target_link_libraries(ai-artworks PRIVATE spdlog::spdlog)
    target_compile_definitions(ai-artworks PRIVATE HAS_SPDLOG=1)
endif()

if(Eigen3_FOUND)
    target_link_libraries(ai-artworks PRIVATE Eigen3::Eigen)
    target_compile_definitions(ai-artworks PRIVATE HAS_EIGEN=1)
endif()

# Graphics libraries
if(glfw3_FOUND)
    target_link_libraries(ai-artworks PRIVATE glfw)
    target_compile_definitions(ai-artworks PRIVATE HAS_GLFW=1)
endif()

if(OpenGL_FOUND)
    target_link_libraries(ai-artworks PRIVATE OpenGL::GL)
    target_compile_definitions(ai-artworks PRIVATE HAS_OPENGL=1)
endif()

if(Vulkan_FOUND)
    target_link_libraries(ai-artworks PRIVATE Vulkan::Vulkan)
    target_compile_definitions(ai-artworks PRIVATE HAS_VULKAN=1)
endif()

# Audio libraries
if(portaudio_FOUND)
    target_link_libraries(ai-artworks PRIVATE portaudio)
    target_compile_definitions(ai-artworks PRIVATE HAS_PORTAUDIO=1)
endif()

# Parallel processing
if(OpenMP_CXX_FOUND)
    target_link_libraries(ai-artworks PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(ai-artworks PRIVATE HAS_OPENMP=1)
endif()

if(TBB_FOUND)
    target_link_libraries(ai-artworks PRIVATE TBB::tbb)
    target_compile_definitions(ai-artworks PRIVATE HAS_TBB=1)
endif()

# Windows-specific libraries
if(WIN32)
    target_link_libraries(ai-artworks PRIVATE
        kernel32 user32 gdi32 winspool shell32 ole32 oleaut32 uuid
        comdlg32 advapi32 ws2_32 winmm imm32 oleacc version
        dwmapi uxtheme comctl32 d3d11 dxgi
    )
endif()

# Custom target for copying assets
add_custom_target(copy_assets ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/assets
        ${CMAKE_BINARY_DIR}/bin/assets
    COMMENT "Copying assets to build directory"
)

add_dependencies(ai-artworks copy_assets)

# Custom target for copying models
add_custom_target(copy_models ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/bin/models
    COMMAND ${CMAKE_COMMAND} -E echo "Models directory created. Place GGUF files here."
    COMMENT "Creating models directory"
)

add_dependencies(ai-artworks copy_models)

# Installation
install(TARGETS ai-artworks
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY assets/
    DESTINATION bin/assets
    FILES_MATCHING PATTERN "*"
)

install(DIRECTORY ${CMAKE_BINARY_DIR}/bin/models/
    DESTINATION bin/models
    FILES_MATCHING PATTERN "*.gguf"
    OPTIONAL
)

# Testing (optional)
option(BUILD_TESTS "Build test suite" OFF)
if(BUILD_TESTS AND GTest_FOUND)
    enable_testing()
    
    add_executable(ai-artworks-tests
        tests/test_main.cpp
        tests/test_core.cpp
        tests/test_ai.cpp
        ${CORE_SOURCES}
        ${AI_SOURCES}
        ${UTILS_SOURCES}
    )
    
    target_link_libraries(ai-artworks-tests PRIVATE
        GTest::gtest_main
        Threads::Threads
    )
    
    if(fmt_FOUND)
        target_link_libraries(ai-artworks-tests PRIVATE fmt::fmt)
    endif()
    
    include(GoogleTest)
    gtest_discover_tests(ai-artworks-tests)
endif()

# Benchmarking (optional)
option(BUILD_BENCHMARKS "Build benchmark suite" OFF)
if(BUILD_BENCHMARKS AND benchmark_FOUND)
    add_executable(ai-artworks-benchmarks
        benchmarks/benchmark_main.cpp
        benchmarks/benchmark_ai.cpp
        ${CORE_SOURCES}
        ${AI_SOURCES}
        ${UTILS_SOURCES}
    )
    
    target_link_libraries(ai-artworks-benchmarks PRIVATE
        benchmark::benchmark
        Threads::Threads
    )
    
    if(fmt_FOUND)
        target_link_libraries(ai-artworks-benchmarks PRIVATE fmt::fmt)
    endif()
endif()

# Print build information
message(STATUS "")
message(STATUS "AI-Artworks Build Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Platform: ${CMAKE_SYSTEM_NAME}")
message(STATUS "  Architecture: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "")
message(STATUS "Dependencies Found:")
message(STATUS "  fmt: ${fmt_FOUND}")
message(STATUS "  spdlog: ${spdlog_FOUND}")
message(STATUS "  Eigen3: ${Eigen3_FOUND}")
message(STATUS "  GLFW: ${glfw3_FOUND}")
message(STATUS "  OpenGL: ${OpenGL_FOUND}")
message(STATUS "  Vulkan: ${Vulkan_FOUND}")
message(STATUS "  PortAudio: ${portaudio_FOUND}")
message(STATUS "  OpenMP: ${OpenMP_CXX_FOUND}")
message(STATUS "  TBB: ${TBB_FOUND}")
message(STATUS "  GTest: ${GTest_FOUND}")
message(STATUS "  benchmark: ${benchmark_FOUND}")
message(STATUS "")

# Warnings for missing critical dependencies
if(NOT fmt_FOUND)
    message(WARNING "fmt library not found. Logging may be limited.")
endif()

if(NOT spdlog_FOUND)
    message(WARNING "spdlog library not found. Advanced logging disabled.")
endif()

if(NOT glfw3_FOUND AND NOT WIN32)
    message(WARNING "GLFW not found. Graphics rendering may be limited.")
endif()