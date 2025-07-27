# Examples CMakeLists.txt
cmake_minimum_required(VERSION 3.20)

# Add executable for image processing demo
add_executable(image_processing_demo
    image_processing_demo.c
)

# Link with the main library
target_link_libraries(image_processing_demo
    ultimate_core
)

# Set target properties
set_target_properties(image_processing_demo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Add math library for Linux
if(NOT WIN32)
    target_link_libraries(image_processing_demo m)
endif()

# Install the demo
install(TARGETS image_processing_demo
    RUNTIME DESTINATION bin
) 