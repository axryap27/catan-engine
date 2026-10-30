cmake_minimum_required(VERSION 3.24)

# The Python module must be built for the interpreter's architecture, which on a Mac need not be
# the host tool's: an Intel (x86_64) CMake would otherwise compile x86_64 and an arm64 Python
# could not load the result. Ask the interpreter (CMAKE_OSX_ARCHITECTURES must be set before
# project()). An explicit -DCMAKE_OSX_ARCHITECTURES wins.
if(CATAN_BUILD_PYTHON AND CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin"
   AND NOT CMAKE_OSX_ARCHITECTURES AND Python3_EXECUTABLE)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import platform; print(platform.machine())"
        OUTPUT_VARIABLE catan_python_arch
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE catan_python_arch_status
        ERROR_QUIET)
    if(catan_python_arch_status EQUAL 0 AND catan_python_arch)
        set(CMAKE_OSX_ARCHITECTURES "${catan_python_arch}" CACHE STRING
            "Architecture of the Python interpreter that will load catan_engine")
    endif()
endif()

project(catan_engine CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(CATAN_CHECKED "Enable checked assertions and extra validation" OFF)
option(CATAN_BUILD_TESTS "Build unit tests" ON)
option(CATAN_BUILD_PYTHON "Build the catan_engine Python module" OFF)

if(CATAN_BUILD_PYTHON)
    # The module is a shared library that statically links the engine libraries.
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

# Before the subdirectories, so tests they register (python_bind) are kept.
if(CATAN_BUILD_TESTS)
    enable_testing()
endif()

add_subdirectory(core)
add_subdirectory(belief)
add_subdirectory(encode)
add_subdirectory(eval)
add_subdirectory(bind)
add_subdirectory(tools)

if(CATAN_BUILD_TESTS)
    add_subdirectory(core/tests)
    add_subdirectory(belief/tests)  # reuses Catch2 and catan_test_support from core/tests
    add_subdirectory(encode/tests)
    add_subdirectory(eval/tests)
    add_subdirectory(bind/tests)
endif()
