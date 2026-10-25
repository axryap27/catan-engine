add_executable(catan_sim catan_sim.cpp)
target_link_libraries(catan_sim PRIVATE catan::core)
target_compile_options(catan_sim PRIVATE -Wall -Wextra -Wpedantic -Werror)
