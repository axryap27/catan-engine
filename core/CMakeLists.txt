add_library(catan_core
    src/topology.cpp
    src/board.cpp
    src/state.cpp
    src/mask.cpp
    src/chance.cpp
    src/rules.cpp
)

target_include_directories(catan_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_compile_options(catan_core PRIVATE
    -Wall -Wextra -Wpedantic -Werror
)

if(CATAN_CHECKED)
    target_compile_definitions(catan_core PUBLIC CATAN_CHECKED=1)
endif()

add_library(catan::core ALIAS catan_core)
