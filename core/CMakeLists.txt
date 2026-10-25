add_library(catan_core
    src/topology.cpp
    src/board.cpp
    src/state.cpp
    src/mask.cpp
    src/chance.cpp
    src/rules.cpp
    src/setup.cpp
    src/turn.cpp
    src/robber.cpp
)

target_include_directories(catan_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_compile_options(catan_core PRIVATE
    -Wall -Wextra -Wpedantic -Werror
)

# GCC treats C99's complex catan() as a builtin and rejects `namespace catan` beside it.
target_compile_options(catan_core PUBLIC $<$<CXX_COMPILER_ID:GNU>:-fno-builtin-catan>)

if(CATAN_CHECKED)
    target_compile_definitions(catan_core PUBLIC CATAN_CHECKED=1)
endif()

add_library(catan::core ALIAS catan_core)
