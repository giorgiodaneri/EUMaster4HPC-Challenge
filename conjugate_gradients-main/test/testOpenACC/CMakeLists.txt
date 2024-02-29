cmake_minimum_required(VERSION 3.20)
project(conjugate-gradient)


# Set the C++ standard and compile options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

# Set the default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Configure flags for Debug and Release builds
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(BUILD_TYPE_DEBUG)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
endif()
message(STATUS)
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS)
set(CMAKE_CXX_FLAGS_DEBUG "-g -O3 -Wall -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -funroll-loops -finline-functions -Wall")


# Find the OpenACC package
find_package(OpenACC REQUIRED)

# Assuming OpenACC is found, set OpenACC compile flags
set(OPENACC_FLAGS "-fopenacc -foffload=nvptx-none -march=sm_80")

# Include directories
target_include_directories(conjugate-gradient PRIVATE ../../include)

# Library files
add_library(conjugate-gradient STATIC
    ../../src/CGSolver.cpp
    ../../src/CGSolverACC.cpp
)

add_executable(mainOpenACC mainOpenACC.cpp)

# Apply OpenACC flags to targets
target_compile_options(conjugate-gradient PRIVATE ${OPENACC_FLAGS})
target_compile_options(mainOpenACC PRIVATE ${OPENACC_FLAGS})

# target_link_libraries
target_link_libraries(conjugate-gradient OpenACC::OpenACC_CXX ${OPENACC_FLAGS})
target_link_libraries(mainOpenACC conjugate-gradient OpenACC::OpenACC_CXX ${OPENACC_FLAGS})

