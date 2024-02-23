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

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_definitions(-DOMPI_SKIP_MPICXX)

include_directories(${MPI_INCLUDE_PATH})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../include)

add_executable(mainMPI_OpenMP main.cpp)

target_include_directories(mainMPI_OpenMP PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include ${MPI_INCLUDE_PATH})

target_link_libraries(mainMPI_OpenMP PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)
