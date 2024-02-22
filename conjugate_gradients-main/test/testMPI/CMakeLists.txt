cmake_minimum_required(VERSION 3.20)

# Default to Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)

project(MPI_Implementation)

# Compiler options
if(CMAKE_BUILD_TYPE MATCHES Debug)
    # Debug settings
    add_compile_options(-g -O0 -Wall -DDEBUG)
else()
    # Release settings
    add_compile_options(-O3 -Wall -march=native)
endif()

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_definitions(-DOMPI_SKIP_MPICXX)

include_directories(${MPI_INCLUDE_PATH})
include_directories(${PROJECT_SOURCE_DIR}/../../include)

add_library(conjugate-gradient STATIC
        ${PROJECT_SOURCE_DIR}/../../src/CGSolver.cpp
        ${PROJECT_SOURCE_DIR}/../../src/CGSolverMPI.cpp
)

set_target_properties(conjugate-gradient PROPERTIES VERSION ${PROJECT_VERSION})

add_executable(mainMPI_OpenMP main.cpp)

target_include_directories(conjugate-gradient PUBLIC ${PROJECT_SOURCE_DIR}/../../include)
target_include_directories(mainMPI_OpenMP PRIVATE ${PROJECT_SOURCE_DIR}/../../include ${MPI_INCLUDE_PATH})

target_link_libraries(mainMPI_OpenMP PRIVATE conjugate-gradient ${MPI_CXX_LIBRARIES} OpenMP::OpenMP_CXX)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}, OpenMP version: ${OpenMP_VERSION}, MPI version: ${MPI_VERSION}")