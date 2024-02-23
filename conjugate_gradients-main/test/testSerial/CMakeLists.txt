cmake_minimum_required(VERSION 3.20)

project(Serial_Implementation VERSION 1.0.0 DESCRIPTION "Serial Version of CG Implementation")

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)

# Default to Release if not specified
if(NOT CMAKE_BUILD_TYPE OR "${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

# Configure flags for Debug and Release builds
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-DBUILD_TYPE_DEBUG)
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -march=native")
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../include)

add_library(conjugate-gradient STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/CGSolver.cpp
)

add_executable(mainSerial main.cpp)

target_include_directories(mainSerial PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)

target_link_libraries(mainSerial PRIVATE conjugate-gradient)
