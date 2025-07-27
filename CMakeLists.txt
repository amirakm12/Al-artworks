cmake_minimum_required(VERSION 3.20)

# Set the project name and version
project(ULTIMATE_System VERSION 1.0.0 LANGUAGES C CXX)

# Set C and C++ standards
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Platform detection
if(WIN32)
    set(ULTIMATE_PLATFORM "Windows")
    add_definitions(-D_WIN32_WINNT=0x0601)  # Windows 7 or later
    add_definitions(-DWIN32_LEAN_AND_MEAN)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    
    # Set Windows subsystem
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /SUBSYSTEM:CONSOLE")
    
    # Windows libraries
    set(PLATFORM_LIBRARIES
        kernel32 user32 gdi32 winspool shell32 ole32 oleaut32 uuid
        comdlg32 advapi32 ws2_32 winmm imm32 oleacc version
    )
elseif(UNIX AND NOT APPLE)
    set(ULTIMATE_PLATFORM "Linux")
    add_definitions(-DULTIMATE_LINUX=1)
    
    # Linux libraries
    set(PLATFORM_LIBRARIES
        pthread
        dl
        m
    )
    
    # Find required packages for Linux
    find_package(Threads REQUIRED)
    if(Threads_FOUND)
        list(APPEND PLATFORM_LIBRARIES ${CMAKE_THREAD_LIBS_INIT})
    endif()
    
elseif(APPLE)
    set(ULTIMATE_PLATFORM "macOS")
    add_definitions(-DULTIMATE_MACOS=1)
    set(PLATFORM_LIBRARIES pthread dl m)
endif()

# OpenMP support (optional but recommended for performance)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found - enabling parallel processing")
    add_definitions(-DULTIMATE_OPENMP_ENABLED=1)
else()
    message(STATUS "OpenMP not found - parallel processing will be limited")
endif()

# Compiler-specific options
if(MSVC)
    add_compile_options(
        /W4
        /wd4100  # Unreferenced parameter
        /wd4201  # Nameless struct/union
        /wd4996  # Deprecated functions
    )
    
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(/Od /Zi /RTC1)
        add_definitions(-DDEBUG=1)
    else()
        add_compile_options(/O2 /Ob2 /Oi /Ot)
        add_definitions(-DNDEBUG=1)
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        -Wall
        -Wextra
        -Wno-unused-parameter
        -Wno-implicit-const-int-float-conversion  # Suppress RAND_MAX warnings
    )
    
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-O0 -g3)
        add_definitions(-DDEBUG=1)
    else()
        add_compile_options(-O2 -g1)
        add_definitions(-DNDEBUG=1)
    endif()
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/core
)

# Source files
file(GLOB_RECURSE CORE_SOURCES
    "src/core/*.c"
    "src/core/*.cpp"
)

file(GLOB_RECURSE HARDWARE_SOURCES
    "src/hardware/*.c"
    "src/hardware/*.cpp"
)

file(GLOB_RECURSE NEURAL_SOURCES
    "src/neural/*.c"
    "src/neural/*.cpp"
)

file(GLOB_RECURSE AI_SOURCES
    "src/ai/*.c"
    "src/ai/*.cpp"
)

# Add main application sources
set(APP_SOURCES
    src/main_minimal.cpp
)

# Create the main library
add_library(ultimate_core STATIC
    ${CORE_SOURCES}
    ${HARDWARE_SOURCES}
    ${NEURAL_SOURCES}
    ${AI_SOURCES}
)

# Target properties
set_target_properties(ultimate_core PROPERTIES
    OUTPUT_NAME "ultimate"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# Target compile definitions
target_compile_definitions(ultimate_core PRIVATE
    ULTIMATE_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
    ULTIMATE_VERSION_MINOR=${PROJECT_VERSION_MINOR}
    ULTIMATE_VERSION_PATCH=${PROJECT_VERSION_PATCH}
    ULTIMATE_PLATFORM="${ULTIMATE_PLATFORM}"
)

# Link platform libraries
target_link_libraries(ultimate_core PRIVATE ${PLATFORM_LIBRARIES})

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(ultimate_core PRIVATE OpenMP::OpenMP_CXX)
endif()

# Create main executable
add_executable(ultimate_system ${APP_SOURCES})
target_link_libraries(ultimate_system PRIVATE ultimate_core ${PLATFORM_LIBRARIES})

if(OpenMP_CXX_FOUND)
    target_link_libraries(ultimate_system PRIVATE OpenMP::OpenMP_CXX)
endif()

set_target_properties(ultimate_system PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Installation
install(TARGETS ultimate_core ultimate_system
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

# Optional: Create example executables
option(BUILD_EXAMPLES "Build example applications" ON)
if(BUILD_EXAMPLES AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples")
    add_subdirectory(examples)
endif()

# Optional: Create tests
option(BUILD_TESTS "Build test suite" OFF)
if(BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests")
    enable_testing()
    add_subdirectory(tests)
endif()

# Print build information
message(STATUS "ULTIMATE System Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C Compiler: ${CMAKE_C_COMPILER_ID}")
message(STATUS "  CXX Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  Platform: ${ULTIMATE_PLATFORM}")
message(STATUS "  OpenMP Support: ${OpenMP_CXX_FOUND}")
message(STATUS "  Build Examples: ${BUILD_EXAMPLES}")
message(STATUS "  Build Tests: ${BUILD_TESTS}") 