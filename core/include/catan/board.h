#pragma once
#include <array>
#include <cstdint>

#include "catan/state.h"
#include "catan/types.h"

// The immutable per-game layout: which resource sits on each hex, its dice token, and which
// harbor sits in each port slot. It is NOT part of GameState (search copies GameState
// constantly); one Board is shared by every state of a game.
//
// Hex ids and port slots follow the numbering documented in topology.h.

namespace catan {

struct Board {
    std::array<HexType, NUM_HEXES> hex_type;
    std::array<uint8_t, NUM_HEXES> token;       // dice number 2..12, 0 on the desert
    std::array<PortType, NUM_PORTS> port_type;  // indexed by topology().port_slots
    uint8_t desert_hex;

    // Derived lookups, filled when a board is built; never edited directly.
    std::array<uint8_t, NUM_VERTICES> vertex_port;          // PortType as uint8_t, NONE if no harbor
    std::array<std::array<uint8_t, 2>, 13> hexes_for_roll;  // hexes producing on roll r, NONE-padded
};

// The official beginner layout.
Board standard_board();

// A random legal layout, fully determined by `seed`. With forbid_adjacent_red, no two
// neighboring hexes both carry a 6 or an 8.
Board random_board(uint64_t seed, bool forbid_adjacent_red = true);

// Fresh game on this board (the robber starts on the desert).
GameState initial_state(const Board& board);

}  // namespace catan
