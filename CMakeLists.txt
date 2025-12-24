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
    include/MainWindow.h
    include/MapViewer2D.h
    include/MapViewer3D.h
    include/CalculatingConditionWidget.h
    include/MethodWidget.h
    include/VelocityModelWidget.h
    include/DataInputWidget.h
    include/ResultWidget.h
    app.qrc
)

add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE include)

target_link_libraries(${PROJECT_NAME}
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
)

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
