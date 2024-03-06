cmake_minimum_required(VERSION 3.20)
project(conjugate-gradient)

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

message(STATUS)
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS)
# add_definitions(-DNO_AFFINITY=1)
# add_compile_options(-DNO_AFFINITY=1)
SET(CMAKE_CXX_FLAGS"-Ofast -Wall -mavx2 -flto -lopenblas -lpthread -march=znver2 -mtune=znver2 -qopt-zmm-usage=high -qopt-prefetch -qopt-dynamic-align")

IF (MY_COMPILER_OPTION STREQUAL "Intel")
  MESSAGE(STATUS "** Compiling with Intel settings **")
  SET(CMAKE_CXX_COMPILER "icpx")
  SET(CMAKE_CXX_FLAGS_RELEASE "-Wall -Ofast -xHost -ipo -march=znver2 -mtune=znver2 -axarch -no-prec-div -fp-model fast=2 -qopt-zmm-usage=high -qopt-prefetch -qopt-dynamic-align")
  SET(CMAKE_CXX_FLAGS_DEBUG "-g -Wall")
  MESSAGE(STATUS "CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}")
ENDIF ()

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
# set(OPENMP_FLAGS "-qopenmp -qopenmp-simd")
# if compiling with GCC
set(OPENMP_FLAGS "-fopenmp -Wopenmp-simd -foffload=nvptx-none -foffload-options=nvptx-none=-march=sm_80")
endif()

include_directories(../../include)

# Library files
add_library(conjugate-gradient STATIC
    ../../src/CGSolver.cpp
    ../../src/CGSolverOMP.cpp
)

add_executable(mainOmp
mainOmp.cpp
${HEADER_FILES}
)

# Include directory
target_include_directories(conjugate-gradient PRIVATE ../../include)
target_include_directories(mainOmp PRIVATE ../../include)

# target_link_libraries
target_link_libraries(conjugate-gradient OpenMP::OpenMP_CXX ${OPENMP_FLAGS} openblas)
target_link_libraries(mainOmp conjugate-gradient OpenMP::OpenMP_CXX ${OPENMP_FLAGS} openblas)

target_link_libraries(mainOmp conjugate-gradient OpenMP::OpenMP_CXX ${OPENMP_FLAGS} openblas)

