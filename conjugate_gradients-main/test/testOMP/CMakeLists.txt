cmake_minimum_required(VERSION 3.20)

project(MPI_Implementation VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)

# Set the default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Configure flags for Debug and Release builds
if(CMAKE_BUILD_TYPE MATCHES Debug)
    add_compile_options(-g -O0 -Wall -DDEBUG)
else()
    add_compile_options(-O3 -Wall -march=native)
endif()

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    set(OPENMP_FLAGS "-fopenmp -fopenmp-simd")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../include)

add_library(conjugate-gradient STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/CGSolver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/CGSolverOMP.cpp
)

add_executable(mainOmp mainOmp.cpp)

target_include_directories(mainOmp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)

target_link_libraries(conjugate-gradient PRIVATE OpenMP::OpenMP_CXX ${OPENMP_FLAGS})
target_link_libraries(mainOmp PRIVATE conjugate-gradient OpenMP::OpenMP_CXX ${OPENMP_FLAGS})