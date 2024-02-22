# Usage:
#   -DDEBUG_MODE=ON/OFF : Enable or disable debug mode (default: OFF)
#   -DBUILD_MPI=ON/OFF : Build the MPI version
#   -DBUILD_OMP=ON/OFF : Build the OpenMP version
#   -DBUILD_OPENACC=ON/OFF : Build the OpenACC version
#   -DBUILD_SERIAL=ON/OFF : Build the serial version
#
# Example:
#   cmake -S . -B build -DDEBUG_MODE=ON -DBUILD_MPI=ON
cmake_minimum_required(VERSION 3.10)

project(ConjugateGradients VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

option(DEBUG_MODE "Enable Debug Mode" OFF)

if(DEBUG_MODE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type" FORCE)
else()
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

option(BUILD_MPI "Build the MPI version" OFF)
option(BUILD_OMP "Build the OpenMP version" OFF)
option(BUILD_OPENACC "Build the OpenACC version" OFF)
option(BUILD_SERIAL "Build the serial version" OFF)

if(BUILD_MPI)
    add_subdirectory(conjugate_gradients-main/test/testMPI)
endif()

if(BUILD_OMP)
    add_subdirectory(conjugate_gradients-main/test/testOMP)
endif()

if(BUILD_OPENACC)
    add_subdirectory(conjugate_gradients-main/test/testOpenACC)
endif()

if(BUILD_SERIAL)
    add_subdirectory(conjugate_gradients-main/test/testSerial)
endif()

if(NOT BUILD_MPI AND NOT BUILD_OMP AND NOT BUILD_OPENACC AND NOT BUILD_SERIAL)
    message(WARNING "No version selected. Please enable at least one version (BUILD_MPI, BUILD_OMP, BUILD_OPENACC, BUILD_SERIAL) to build.")
endif()
