include(FetchContent)
FetchContent_Declare(Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.7.1
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(Catch2)
include(Catch)

# One binary per area so each suite builds and runs on its own.
foreach(name topology board state rng rules)
    add_executable(test_${name} test_${name}.cpp)
    target_link_libraries(test_${name} PRIVATE catan::core Catch2::Catch2WithMain)
    target_compile_options(test_${name} PRIVATE -Wall -Wextra -Wpedantic -Werror)
    catch_discover_tests(test_${name})
endforeach()
