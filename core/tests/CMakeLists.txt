include(FetchContent)
FetchContent_Declare(Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.7.1
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(Catch2)
include(Catch)

# Fixtures and the independent reference rules shared by the suites.
add_library(catan_test_support STATIC support/fixtures.cpp support/reference.cpp)
target_include_directories(catan_test_support PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(catan_test_support PUBLIC catan::core)
target_compile_options(catan_test_support PRIVATE -Wall -Wextra -Wpedantic -Werror)

# One binary per area so each suite builds and runs on its own.
foreach(name topology board state rng rules turn robber devcards awards invariants fuzz golden)
    add_executable(test_${name} test_${name}.cpp)
    target_link_libraries(test_${name} PRIVATE catan_test_support Catch2::Catch2WithMain)
    target_compile_options(test_${name} PRIVATE -Wall -Wextra -Wpedantic -Werror)
    catch_discover_tests(test_${name})
endforeach()
