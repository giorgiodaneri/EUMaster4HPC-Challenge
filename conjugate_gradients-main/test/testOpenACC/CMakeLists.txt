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
set(CMAKE_CXX_FLAGS_DEBUG "-g -O3 -Wall -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -march=native")

find_package(OpenACC REQUIRED)
if(OpenACC_CXX_FOUND)
set(OPENACC_FLAGS "-fopenacc -foffload-options=nvptx-none=-march=sm_80")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenACC_CXX_FLAGS}")
endif()

# Project declaration
project(conjugate-gradient VERSION 1.0.0 DESCRIPTION "Accelerated version of CG iterative method")

include_directories(../../include)

# Library files
add_library(conjugate-gradient STATIC
    ../../src/CGSolver.cpp
    ../../src/CGSolverACC.cpp
)

# Version property
set_target_properties(conjugate-gradient PROPERTIES VERSION ${PROJECT_VERSION})

add_executable(mainOpenACC
mainOpenACC.cpp
${HEADER_FILES}
)

target_compile_options(conjugate-gradient PRIVATE ${OPENACC_OPTIONS})
target_compile_options(mainOpenACC PRIVATE ${OPENACC_OPTIONS})

# target_link_libraries
target_link_libraries(conjugate-gradient OpenACC::OpenACC_CXX ${OPENACC_FLAGS})
target_link_libraries(mainOpenACC conjugate-gradient OpenACC::OpenACC_CXX ${OPENACC_FLAGS})
