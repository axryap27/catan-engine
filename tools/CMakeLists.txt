add_executable(catan_sim catan_sim.cpp)
target_link_libraries(catan_sim PRIVATE catan::core)
target_compile_options(catan_sim PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_executable(catan_bench catan_bench.cpp)
target_link_libraries(catan_bench PRIVATE catan::core)
target_compile_options(catan_bench PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_executable(catan_match catan_match.cpp)
target_link_libraries(catan_match PRIVATE catan::eval)
target_compile_options(catan_match PRIVATE -Wall -Wextra -Wpedantic -Werror)
