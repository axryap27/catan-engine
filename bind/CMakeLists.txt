# Text views (C++ only, used by the CLI tools and the Python module).
add_library(catan_text src/text.cpp)
target_include_directories(catan_text PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(catan_text PUBLIC catan::core)
target_compile_options(catan_text PRIVATE -Wall -Wextra -Wpedantic -Werror)
add_library(catan::text ALIAS catan_text)

# The Python extension module `catan_engine` (python/module.cpp), built with
# -DCATAN_BUILD_PYTHON=ON or `cmake --preset python`. It links the engine libraries statically
# (position-independent code is switched on in the root CMakeLists).
if(CATAN_BUILD_PYTHON)
    find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter Development.Module)

    include(FetchContent)
    FetchContent_Declare(pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v3.1.0
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(pybind11)

    pybind11_add_module(catan_engine python/module.cpp)
    target_link_libraries(catan_engine PRIVATE catan::text catan::encode catan::eval)
    target_compile_options(catan_engine PRIVATE -Wall -Wextra -Wpedantic -Werror)

    # Type stubs next to the built module, so editors and type checkers find them.
    add_custom_command(TARGET catan_engine POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${CMAKE_CURRENT_SOURCE_DIR}/python/catan_engine.pyi
                $<TARGET_FILE_DIR:catan_engine>/catan_engine.pyi
        VERBATIM)

    # Needs numpy and pytest in the interpreter (requirements-dev.txt).
    if(CATAN_BUILD_TESTS)
        # Run from the tests directory rather than passing its path: pytest rejects a path
        # whose letter case differs from the disk's (macOS), and getcwd returns the real case.
        add_test(NAME python_bind
            COMMAND ${Python3_EXECUTABLE} -m pytest -q -p no:cacheprovider
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)
        # The tests import the module from its build directory and run play.py as a script.
        set(catan_python_env
            "PYTHONPATH=$<TARGET_FILE_DIR:catan_engine>"
            "CATAN_PY_PLAY=${CMAKE_CURRENT_SOURCE_DIR}/python/play.py")
        set_tests_properties(python_bind PROPERTIES ENVIRONMENT "${catan_python_env}")
    endif()
endif()
