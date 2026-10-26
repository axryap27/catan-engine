cmake_minimum_required(VERSION 3.24)
project(catan_engine CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(CATAN_CHECKED "Enable checked assertions and extra validation" OFF)
option(CATAN_BUILD_TESTS "Build unit tests" ON)

add_subdirectory(core)
add_subdirectory(belief)
add_subdirectory(tools)

if(CATAN_BUILD_TESTS)
    enable_testing()
    add_subdirectory(core/tests)
    add_subdirectory(belief/tests)  # reuses Catch2 and catan_test_support from core/tests
endif()
