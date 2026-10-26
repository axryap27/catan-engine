add_executable(test_eval test_eval.cpp)
target_link_libraries(test_eval PRIVATE catan::eval catan::belief catan_test_support
                      Catch2::Catch2WithMain)
target_compile_options(test_eval PRIVATE -Wall -Wextra -Wpedantic -Werror)
catch_discover_tests(test_eval)
