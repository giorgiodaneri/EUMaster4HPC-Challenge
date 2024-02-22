cmake_minimum_required(VERSION 3.20)

# Default to Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)

project(OpenMP_Implementation)

# Compiler options
if(CMAKE_BUILD_TYPE MATCHES Debug)
    # Debug settings
    add_compile_options(-g -O0 -Wall -DDEBUG)
else()
    # Release settings
    add_compile_options(-O3 -Wall -march=native)
endif()

find_package(OpenMP REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../include)

add_library(conjugate-gradient STATIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/CGSolver.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/CGSolverOMP.cpp
)

set_target_properties(conjugate-gradient PROPERTIES VERSION ${PROJECT_VERSION})

add_executable(mainOmp mainOmp.cpp)

target_include_directories(mainOmp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)

target_link_libraries(conjugate-gradient PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(mainOmp PRIVATE conjugate-gradient)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}, OpenMP version: ${OpenMP_VERSION}")