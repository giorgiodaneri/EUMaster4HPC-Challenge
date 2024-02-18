cmake_minimum_required(VERSION 3.20)
project(N-body)

set(CMAKE-CXX-STANDARD 20)
set(CMAKE-CXX-FLAGS "${CMAKE-CXX-FLAGS} -Wall")

# Set default build type to Release.
if(NOT CMAKE_BUILD_TYPE OR "${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    add_definitions(-DBUILD_TYPE_DEBUG)
endif()
message(STATUS)
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS)
SET(CMAKE_CXX_FLAGS"-O3 -Wall -march=native")

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
set(OPENMP_FLAGS "-fopenmp -fopenmp-simd")
endif()

# Project declaration
project(conjugate-gradient VERSION 1.0.0 DESCRIPTION "Accelerated version of CG iterative method")

include_directories(../../include)

# Library files
add_library(conjugate-gradient STATIC
    ../../src/CGSolver.cpp
    ../../src/CGSolverOMP.cpp
)

# Version property
set_target_properties(conjugate-gradient PROPERTIES VERSION ${PROJECT_VERSION})

add_executable(mainOmp
mainOmp.cpp
${HEADER_FILES}
)

# Include directory
target_include_directories(conjugate-gradient PRIVATE ../../include)
target_include_directories(mainOmp PRIVATE ../../include)

# target_link_libraries
target_link_libraries(conjugate-gradient OpenMP::OpenMP_CXX ${OPENMP_FLAGS})
target_link_libraries(mainOmp conjugate-gradient OpenMP::OpenMP_CXX ${OPENMP_FLAGS})
