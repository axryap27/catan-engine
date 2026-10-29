add_executable(test_encode test_encode.cpp)
target_link_libraries(test_encode PRIVATE catan::encode catan_test_support Catch2::Catch2WithMain)
target_compile_options(test_encode PRIVATE -Wall -Wextra -Wpedantic -Werror)
catch_discover_tests(test_encode)
