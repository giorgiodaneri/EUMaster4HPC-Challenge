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
set(CMAKE_CXX_FLAGS_DEBUG "-O3 -Wall -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -march=native")

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

include_directories(../../include)

# Library files
add_library(conjugate-gradient STATIC
    ../../src/CGSolver.cpp
)

# Version property
set_target_properties(conjugate-gradient PROPERTIES VERSION ${PROJECT_VERSION})

add_executable(main
main.cpp
${HEADER_FILES}
)

# Include directory
target_include_directories(conjugate-gradient PRIVATE ../../include)
target_include_directories(main PRIVATE ../../include)

# target_link_libraries
target_link_libraries(conjugate-gradient)
target_link_libraries(main conjugate-gradient)
