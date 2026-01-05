cmake_minimum_required(VERSION 3.21)
project(bismillah LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# Find Qt6
find_package(Qt6 COMPONENTS 
    Core 
    Gui 
    Widgets 
    REQUIRED
)

# Project sources
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
    src/LocalInversionMethods.cpp
    src/GlobalInversionMethods.cpp
    src/GeneticAlgorithm.cpp
    include/MainWindow.h
    include/MapViewer2D.h
    include/MapViewer3D.h
    include/CalculatingConditionWidget.h
    include/MethodWidget.h
    include/VelocityModelWidget.h
    include/DataInputWidget.h
    include/ResultWidget.h
    include/InversionMethods.h
    include/CUDAInversionWrapper.h
    app.qrc
)

# Add CUDA wrapper if CUDA is available
if(CMAKE_CUDA_COMPILER)
    list(APPEND PROJECT_SOURCES src/CUDAInversionWrapper.cpp)
endif()

add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE include)

target_link_libraries(${PROJECT_NAME}
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
)

# Enable OpenMP for parallel processing
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP found - parallel processing enabled")
endif()

# Copy resources to build folder
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/maps
    $<TARGET_FILE_DIR:${PROJECT_NAME}>/maps
    COMMENT "Copying maps directory"
)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/examples
    $<TARGET_FILE_DIR:${PROJECT_NAME}>/examples
    COMMENT "Copying examples directory"
)

# Optional: Check for CUDA for GPU acceleration
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    
    # CUDA sources - separable compilation
    set(CUDA_SOURCES
        src/cuda/GridSearchKernel.cu
        src/cuda/SimpleSAKernel.cu
        src/cuda/MetropolisSAKernel.cu
        src/cuda/CauchySAKernel.cu
        src/cuda/StandardGAKernel.cu
        src/cuda/SteadyStateGAKernel.cu
        src/cuda/SPEA2Kernel.cu
    )
    
    # Enable separable compilation for CUDA
    set_property(TARGET inversion_cuda PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET inversion_cuda PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    
    # Create CUDA library
    add_library(inversion_cuda STATIC ${CUDA_SOURCES})
    target_include_directories(inversion_cuda PRIVATE include)
    
    # Link CUDA library to main executable
    target_link_libraries(${PROJECT_NAME} inversion_cuda)
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_CUDA)
    
    message(STATUS "CUDA found - GPU acceleration enabled")
else()
    message(STATUS "CUDA not found - using CPU only")
endif()
