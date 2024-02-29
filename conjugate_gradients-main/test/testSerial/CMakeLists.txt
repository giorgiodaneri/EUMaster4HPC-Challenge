cmake_minimum_required(VERSION 3.20)
project(conjugate-gradient)


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

IF (MY_COMPILER_OPTION STREQUAL "Intel")
  MESSAGE(STATUS "** Compiling with Intel settings **")
  SET(CMAKE_CXX_COMPILER "icpx")
#   SET(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -xHost -ipo -march=native -axarch -no-prec-div -fp-model fast=2 -qopt-zmm-usage=high")
  SET(CMAKE_CXX_FLAGS_RELEASE "-O2 -Wall")
  SET(CMAKE_CXX_FLAGS_DEBUG "-g -O3 -Wall -xHost -ipo -march=native -axarch -no-prec-div -fp-model fast=2 -qopt-zmm-usage=high")
  MESSAGE(STATUS "CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}")
ENDIF ()

# Project declaration
project(conjugate-gradient VERSION 1.0.0 DESCRIPTION "Accelerated version of CG iterative method")


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../include)

add_library(conjugate-gradient STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/CGSolver.cpp
)

add_executable(mainSerial main.cpp)

target_include_directories(mainSerial PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)

target_link_libraries(mainSerial PRIVATE conjugate-gradient)
