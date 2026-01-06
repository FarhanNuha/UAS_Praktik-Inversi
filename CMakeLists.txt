cmake_minimum_required(VERSION 3.21)
project(bismillah LANGUAGES CXX)

# Check for CUDA
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CUDA_AVAILABLE ON)
    message(STATUS "CUDA support enabled")
else()
    set(CUDA_AVAILABLE OFF)
    message(STATUS "CUDA not found - GPU acceleration disabled")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CUDA_AVAILABLE)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Enable automatic MOC, RCC, and UIC
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# Find Qt6 with required components
find_package(Qt6 REQUIRED COMPONENTS 
    Core 
    Gui 
    Widgets
)

# Project sources - organized by type
set(PROJECT_HEADERS
    include/MainWindow.h
    include/MapViewer2D.h
    include/MapViewer3D.h
    include/CalculatingConditionWidget.h
    include/MethodWidget.h
    include/VelocityModelWidget.h
    include/DataInputWidget.h
    include/ResultWidget.h
    include/ComputationEngine.h
    include/SharedTypes.h
)

set(PROJECT_SOURCES
    src/main.cpp
    src/MainWindow.cpp
    src/MapViewer2D.cpp
    src/MapViewer3D.cpp
    src/CalculatingConditionWidget.cpp
    src/MethodWidget.cpp
    src/VelocityModelWidget.cpp
    src/DataInputWidget.cpp
    src/ResultWidget.cpp
    src/ComputationEngine.cpp
)

# Add CUDA sources if available
if(CUDA_AVAILABLE)
    set(CUDA_SOURCES
        src/GPUKernels.cu
    )
    set(CUDA_HEADERS
        include/GPUKernels.cuh
    )
    list(APPEND PROJECT_SOURCES ${CUDA_SOURCES})
    list(APPEND PROJECT_HEADERS ${CUDA_HEADERS})
endif()

set(PROJECT_RESOURCES
    app.qrc
)

# Create executable
add_executable(${PROJECT_NAME} 
    ${PROJECT_SOURCES}
    ${PROJECT_HEADERS}
    ${PROJECT_RESOURCES}
)

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link Qt6 libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
)

# Add CUDA compilation flags and definitions
if(CUDA_AVAILABLE)
    target_compile_definitions(${PROJECT_NAME} PRIVATE CUDA_ENABLED)
    
    # Set CUDA architectures (adjust based on your GPU)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "60;70;75;80;86;89;90"  # Support wide range of GPUs
    )
    
    # CUDA specific flags
    if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
        target_compile_options(${PROJECT_NAME} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:
                --use_fast_math
                --expt-relaxed-constexpr
            >
        )
    endif()
endif()

# Platform-specific settings
if(WIN32)
    # Windows-specific settings
    set_target_properties(${PROJECT_NAME} PROPERTIES
        WIN32_EXECUTABLE ON
    )
    
    # Copy Qt DLLs to build directory (for easier debugging)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "Deploying Qt libraries..."
        COMMAND Qt6::windeployqt --no-compiler-runtime --no-translations $<TARGET_FILE:${PROJECT_NAME}>
        COMMENT "Running windeployqt..."
    )
elseif(UNIX AND NOT APPLE)
    # Linux-specific settings
    target_link_libraries(${PROJECT_NAME} PRIVATE
        pthread
        m
    )
elseif(APPLE)
    # macOS-specific settings
    set_target_properties(${PROJECT_NAME} PROPERTIES
        MACOSX_BUNDLE ON
    )
endif()

# Copy resource directories to build folder
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/maps
    $<TARGET_FILE_DIR:${PROJECT_NAME}>/maps
    COMMENT "Copying maps directory to build folder"
)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/examples
    $<TARGET_FILE_DIR:${PROJECT_NAME}>/examples
    COMMENT "Copying examples directory to build folder"
)

# Compiler warnings and optimizations
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE 
        /W4           # Warning level 4
        /MP           # Multi-processor compilation
        /permissive-  # Standards conformance
    )
    
    # Release optimizations
    target_compile_options(${PROJECT_NAME} PRIVATE 
        $<$<CONFIG:Release>:/O2 /GL>
    )
    target_link_options(${PROJECT_NAME} PRIVATE
        $<$<CONFIG:Release>:/LTCG>
    )
else()
    target_compile_options(${PROJECT_NAME} PRIVATE 
        -Wall 
        -Wextra 
        -pedantic
    )
    
    # Release optimizations
    target_compile_options(${PROJECT_NAME} PRIVATE 
        $<$<CONFIG:Release>:-O3 -march=native>
    )
endif()

# Debug/Release configurations
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "=== Build Configuration Summary ===")
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Qt6 Version: ${Qt6_VERSION}")
message(STATUS "Qt6 Directory: ${Qt6_DIR}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "Generator: ${CMAKE_GENERATOR}")
if(CUDA_AVAILABLE)
    message(STATUS "CUDA: ENABLED")
    message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
    message(STATUS "CUDA Version: ${CMAKE_CUDA_COMPILER_VERSION}")
    message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
else()
    message(STATUS "CUDA: DISABLED (CPU-only mode)")
endif()
message(STATUS "===================================")
message(STATUS "")
