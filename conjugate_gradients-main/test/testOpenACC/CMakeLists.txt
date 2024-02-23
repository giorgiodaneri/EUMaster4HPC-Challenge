cmake_minimum_required(VERSION 3.20)
project(OpenACC_Implementation VERSION 1.0.0 DESCRIPTION "OpenACC Implementation")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

# Set the default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Configure flags for Debug and Release builds
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    add_definitions(-DBUILD_TYPE_DEBUG)
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -march=native")
endif()


find_package(OpenACC REQUIRED)
if(OpenACC_CXX_FOUND)
    set(OPENACC_FLAGS "-fopenacc")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenACC_CXX_FLAGS} -foffload-options=nvptx-none=-march=sm_80")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../include)

add_library(conjugate-gradient STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/CGSolver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/CGSolverACC.cpp
)

add_executable(mainOpenACC mainOpenACC.cpp)

target_include_directories(mainOpenACC PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)

target_compile_options(conjugate-gradient PRIVATE ${OPENACC_FLAGS})
target_compile_options(mainOpenACC PRIVATE ${OPENACC_FLAGS})


target_link_libraries(conjugate-gradient PRIVATE OpenACC::OpenACC_CXX ${OPENACC_FLAGS})
target_link_libraries(mainOpenACC PRIVATE conjugate-gradient OpenACC::OpenACC_CXX ${OPENACC_FLAGS})
