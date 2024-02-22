cmake_minimum_required(VERSION 3.20)

# Default to Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)

project(MPI_Implementation)

# Compiler options
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # Debug settings
    add_compile_options(-g -O0 -Wall -DDEBUG)
else()
    # Release settings
    add_compile_options(-O3 -Wall -march=native)
endif()

find_package(OpenACC REQUIRED)
if(OpenACC_CXX_FOUND)
    set(OPENACC_FLAGS "-fopenacc -foffload-options=nvptx-none=-march=sm_80")
    add_compile_options(${OpenACC_CXX_FLAGS})
endif()

include_directories(${PROJECT_SOURCE_DIR}/../../include)

# Library files
add_library(conjugate-gradient STATIC
        ${PROJECT_SOURCE_DIR}/../../src/CGSolver.cpp
        ${PROJECT_SOURCE_DIR}/../../src/CGSolverMPI.cpp
)

set_target_properties(conjugate-gradient PROPERTIES VERSION ${PROJECT_VERSION})

# Executable
add_executable(mainOpenACC
        mainOpenACC.cpp
)

target_compile_options(conjugate-gradient PRIVATE ${OPENACC_OPTIONS})
target_compile_options(mainOpenACC PRIVATE ${OPENACC_OPTIONS})

target_link_libraries(conjugate-gradient OpenACC::OpenACC_CXX ${OPENACC_FLAGS})
target_link_libraries(mainOpenACC conjugate-gradient OpenACC::OpenACC_CXX ${OPENACC_FLAGS})

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}, OpenACC version: ${OpenACC_VERSION}")
