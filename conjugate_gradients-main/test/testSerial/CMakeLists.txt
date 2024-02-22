cmake_minimum_required(VERSION 3.20)

# Default to Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)

project(Serial_Implementation)

# Compiler options
if(CMAKE_BUILD_TYPE MATCHES Debug)
    # Debug settings
    add_compile_options(-g -O0 -Wall -DDEBUG)
else()
    # Release settings
    add_compile_options(-O3 -Wall -march=native)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../include)

add_library(conjugate-gradient STATIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/CGSolver.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/CGSolverSerial.cpp
)

set_target_properties(conjugate-gradient PROPERTIES VERSION ${PROJECT_VERSION})

add_executable(main main.cpp)

target_include_directories(main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)

target_link_libraries(main PRIVATE conjugate-gradient)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}, Serial version")
