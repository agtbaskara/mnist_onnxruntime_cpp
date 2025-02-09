cmake_minimum_required(VERSION 3.10)
project(app CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Enable optimization flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

message(STATUS "Workspace Directory: ${CMAKE_SOURCE_DIR}")

# Set OpenCV paths manually
set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/lib/opencv-4.8.0/build")
find_package(OpenCV REQUIRED)

# Set ONNX Runtime paths manually
set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/lib/onnxruntime-linux-x64-gpu-1.20.1")

# Add executable
add_executable(app src/main.cpp)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_DIR}/include
)

include_directories(
    ${ONNXRUNTIME_DIR}/lib
)

target_link_libraries(app
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so
)
