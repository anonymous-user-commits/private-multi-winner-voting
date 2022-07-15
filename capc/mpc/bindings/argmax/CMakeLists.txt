cmake_minimum_required(VERSION 2.8.11)
set(CMAKE_CXX_STANDARD 11)
project (argmax_cpp)
set(NAME "argmax_cpp")
set(ENABLE_FLOAT ON)

find_path(CMAKE_FOLDER NAMES cmake/emp-tool-config.cmake)
include(${CMAKE_FOLDER}/cmake/common.cmake)
include(${CMAKE_FOLDER}/cmake/enable_rdseed.cmake)
include(${CMAKE_FOLDER}/cmake/enable_float.cmake)

find_package(emp-sh2pc REQUIRED)
find_package(emp-tool REQUIRED)
include_directories(${EMP-SH2PC_INCLUDE_DIRS})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
add_library(argmax_cpp SHARED argmax_cython.hpp argmax_cython.cpp)
target_link_libraries(argmax_cpp ${EMP-SH2PC_LIBRARIES})
