cmake_minimum_required(VERSION 3.5)
project(RealSenseReader LANGUAGES CXX CUDA)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(CUDA REQUIRED)
if(CUDA_FOUND)
  message("CUDA version is ${CUDA_VERSION}")
endif()

find_package(realsense2 REQUIRED)
if(realsense2_FOUND)
  message("realsense2 version is ${realsense2_VERSION}")
endif()
find_package(OpenCV 4.0 QUIET)
################################################################################
# Build
################################################################################
if($ENV{BUILD_TYPE} MATCHES "Release")
  set(CMAKE_BUILD_TYPE Release)
  message("Build type is Release")
else()
  set(CMAKE_BUILD_TYPE Debug)
  message("Build type is Debug")
endif()

include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${realsense2_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIRS}
)

add_executable(
  ${PROJECT_NAME}
  ${PROJECT_SOURCE_DIR}/src/main.cpp
  ${PROJECT_SOURCE_DIR}/src/image_handler.cpp
  ${PROJECT_SOURCE_DIR}/src/unit_test.cpp)

target_link_libraries(
  ${PROJECT_NAME}
  ${OpenCV_LIBS}
  realsense2::realsense2
)
################################################################################
# Install
################################################################################

################################################################################
# Macro for ament package
################################################################################
