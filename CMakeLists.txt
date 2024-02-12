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

add_compile_options(-O3 -march=native -I/include)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
set(OPENMP_FLAGS "-fopenmp")
endif()

# Project declaration
project(conjugate-gradient VERSION 1.0.0 DESCRIPTION "Accelerated version of CG iterative method")

# Library files
add_library(conjugate-gradient STATIC
    src/conjugate_gradients.cpp
)

# Version property
set_target_properties(conjugate-gradient PROPERTIES VERSION ${PROJECT_VERSION})

add_executable(conjugateGradient
conjugate_gradients.cpp
${HEADER_FILES}
)

# Include directory
target_include_directories(conjugate-gradient PRIVATE include)
target_include_directories(conjugateGradient PRIVATE 
    include/
)

# target_link_libraries
target_link_libraries(conjugateGradient OpenMP::OpenMP_CXX)
target_link_libraries(conjugate-gradient OpenMP::OpenMP_CXX)






