set(CUDA_USE_STATIC_CUDA_RUNTIME FALSE)

cmake_minimum_required(VERSION 2.8.7)

project(MyFYP)

find_package(OpenCV REQUIRED)

#cuda
find_package(CUDA)
message(STATUS "    version: ${OpenCV_VERSION}")
message("-- OPENCV include:   " ${OpenCV_INCLUDE_DIRS})
#message("-- OPENCV libs:   " ${OpenCV_LIBS} )
include_directories(${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
#add directory for libnvbuf-utils to program
link_directories(/usr/lib/aarch64-linux-gnu/tegra)

add_executable(image_stitching_SURF_GPU image_stitching_SURF_GPU.cpp )

target_link_libraries(image_stitching_SURF_GPU ${OpenCV_LIBS} ${CUDA_LIBRARIES})

 
