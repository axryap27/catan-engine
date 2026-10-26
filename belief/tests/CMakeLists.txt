add_executable(test_belief test_belief.cpp)
target_link_libraries(test_belief PRIVATE catan::belief catan_test_support Catch2::Catch2WithMain)
target_compile_options(test_belief PRIVATE -Wall -Wextra -Wpedantic -Werror)
catch_discover_tests(test_belief)
