add_executable(test_text test_text.cpp)
target_link_libraries(test_text PRIVATE catan::text catan_test_support Catch2::Catch2WithMain)
target_compile_options(test_text PRIVATE -Wall -Wextra -Wpedantic -Werror)
catch_discover_tests(test_text)
