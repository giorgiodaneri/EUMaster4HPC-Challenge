cmake_minimum_required(VERSION 3.20)
project(N-body)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

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
set(CMAKE_CXX_FLAGS_DEBUG "-g -O3 -Wall -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -march=native -mtune=native -mavx2 -mfma -funroll-loops") 

# IF (MY_COMPILER_OPTION STREQUAL "Intel")
#   MESSAGE(STATUS "** Compiling with Intel settings **")
#   SET(CMAKE_CXX_COMPILER "icpx")
#   SET(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -march=native")
#   SET(CMAKE_CXX_FLAGS_DEBUG "-g")
# ENDIF ()

find_package(OpenACC REQUIRED)
if(OpenACC_CXX_FOUND)
    set(OPENACC_FLAGS "-fopenacc -foffload-options=nvptx-none=-march=sm_80")
    message(STATUS "OpenACC_CXX_FLAGS: ${OpenACC_CXX_FLAGS}")
endif()

# Project declaration
project(conjugate-gradient VERSION 1.0.0 DESCRIPTION "Accelerated version of CG iterative method")

include_directories(../../include)

# Library files
add_library(conjugate-gradient STATIC
    ../../src/CGSolver.cpp
    ../../src/CGSolverACC.cpp
)

set(MKL_LIBRARIES mkl_intel_lp64 mkl_sequential mkl_core pthread m dl)

# Version property
set_target_properties(conjugate-gradient PROPERTIES VERSION ${PROJECT_VERSION})

add_executable(mainOpenACC
    mainOpenACC.cpp
    ${HEADER_FILES}
)

target_compile_options(conjugate-gradient PRIVATE ${OPENACC_OPTIONS})
target_compile_options(mainOpenACC PRIVATE ${OPENACC_OPTIONS})

# Add profiling flags for gprof
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")

# target_link_libraries
target_link_libraries(conjugate-gradient OpenACC::OpenACC_CXX ${OPENACC_FLAGS} ${MKL_LIBRARIES})
target_link_libraries(mainOpenACC conjugate-gradient OpenACC::OpenACC_CXX ${OPENACC_FLAGS} ${MKL_LIBRARIES})
