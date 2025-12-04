cmake_minimum_required(VERSION 3.10)

# Project name and C++ standard
project(ApproxKMedianClustering CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# If you want some optimization by default (like -O2)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Paths
set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(CLASSES_DIR ${PROJECT_ROOT}/classes)
set(HEADER_DIR  ${CLASSES_DIR}/header)
set(SOURCE_DIR  ${CLASSES_DIR}/source)

# Include directory for headers
include_directories(${HEADER_DIR})

# Source files
set(SOURCES
    ${PROJECT_ROOT}/main.cpp
    ${SOURCE_DIR}/RangeCountingOracle.cpp
    ${SOURCE_DIR}/Representative.cpp
    ${SOURCE_DIR}/k_median++.cpp
    # 만약 point.cpp를 나중에 만들었다면 여기에 추가:
    # ${SOURCE_DIR}/point.cpp
)

# Executable target
add_executable(exp_sublinear ${SOURCES})
