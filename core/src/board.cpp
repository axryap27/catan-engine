#include "catan/board.h"

#include <utility>

#include "catan/rng.h"
#include "catan/topology.h"

using namespace std;

namespace catan {
namespace {

// Fills desert_hex, vertex_port and hexes_for_roll from hex_type, token and port_type.
void finalize(Board& board) {
    const Topology& topo = topology();

    board.desert_hex = NONE;
    board.vertex_port.fill(static_cast<uint8_t>(NONE));
    for (auto& hexes : board.hexes_for_roll) hexes.fill(NONE);

    for (int h = 0; h < NUM_HEXES; ++h) {
        if (board.hex_type[h] == HexType::DESERT) {
            board.desert_hex = static_cast<uint8_t>(h);
            continue;
        }
        auto& hexes = board.hexes_for_roll[board.token[h]];
        // Each number is used at most twice, so there is always a free slot.
        (hexes[0] == NONE ? hexes[0] : hexes[1]) = static_cast<uint8_t>(h);
    }

    for (int p = 0; p < NUM_PORTS; ++p) {
        const PortSlot& slot = topo.port_slots[p];
        const auto type = static_cast<uint8_t>(board.port_type[p]);
        board.vertex_port[slot.v0] = type;
        board.vertex_port[slot.v1] = type;
    }
}

// Fisher-Yates shuffle driven by the BOARD stream; `index` advances once per draw.
template <typename T, size_t N>
void shuffle(array<T, N>& items, uint64_t seed, uint64_t& index) {
    for (size_t i = N - 1; i > 0; --i) {
        const auto j = random_below(seed, Stream::BOARD, index++, static_cast<uint32_t>(i + 1));
        swap(items[i], items[j]);
    }
}

bool is_red(uint8_t token) { return token == 6 || token == 8; }

bool has_adjacent_red(const Board& board) {
    const Topology& topo = topology();
    for (int h = 0; h < NUM_HEXES; ++h) {
        if (!is_red(board.token[h])) continue;
        for (uint8_t n : topo.hex_neighbors[h]) {
            if (n != NONE && is_red(board.token[n])) return true;
        }
    }
    return false;
}

// Writes `tokens` onto the non-desert hexes in hex-id order and 0 onto the desert.
void place_tokens(Board& board, const array<uint8_t, 18>& tokens) {
    size_t next = 0;
    for (int h = 0; h < NUM_HEXES; ++h) {
        board.token[h] = (board.hex_type[h] == HexType::DESERT) ? 0 : tokens[next++];
    }
}

}  // namespace

Board standard_board() {
    constexpr HexType BRICK = HexType::BRICK, LUMBER = HexType::LUMBER, WOOL = HexType::WOOL,
                      GRAIN = HexType::GRAIN, ORE = HexType::ORE, DESERT = HexType::DESERT;
    constexpr PortType ANY = PortType::GENERIC;

    Board board{};
    board.hex_type = {
        ORE,   WOOL,   LUMBER,                // row 0
        GRAIN, BRICK,  WOOL,   BRICK,         // row 1
        GRAIN, LUMBER, DESERT, LUMBER, ORE,   // row 2
        LUMBER, ORE,   GRAIN,  WOOL,          // row 3
        BRICK, GRAIN,  WOOL,                  // row 4
    };
    board.token = {
        10, 2, 9,
        12, 6, 4, 10,
        9, 11, 0, 3, 8,
        8, 3, 4, 5,
        5, 6, 11,
    };

    // Rulebook "Starting Map for Beginners" (5th ed., Illustration A). Slots run clockwise
    // from the top-left.
    board.port_type = {
        ANY,                 // slot 0: hex 0 NW
        PortType::GRAIN,     // slot 1: hex 1 NE
        PortType::ORE,       // slot 2: hex 6 NE
        ANY,                 // slot 3: hex 11 E
        PortType::WOOL,      // slot 4: hex 15 SE
        ANY,                 // slot 5: hex 17 SE
        ANY,                 // slot 6: hex 16 SW
        PortType::BRICK,     // slot 7: hex 12 W
        PortType::LUMBER,    // slot 8: hex 3 W
    };

    finalize(board);
    return board;
}

Board random_board(uint64_t seed, bool forbid_adjacent_red) {
    constexpr HexType BRICK = HexType::BRICK, LUMBER = HexType::LUMBER, WOOL = HexType::WOOL,
                      GRAIN = HexType::GRAIN, ORE = HexType::ORE, DESERT = HexType::DESERT;

    uint64_t index = 0;  // running position in the BOARD stream, shared by every shuffle
    Board board{};

    board.hex_type = {
        LUMBER, LUMBER, LUMBER, LUMBER,
        WOOL,   WOOL,   WOOL,   WOOL,
        GRAIN,  GRAIN,  GRAIN,  GRAIN,
        BRICK,  BRICK,  BRICK,
        ORE,    ORE,    ORE,
        DESERT,
    };
    shuffle(board.hex_type, seed, index);

    array<uint8_t, 18> tokens = {2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12};
    do {
        shuffle(tokens, seed, index);
        place_tokens(board, tokens);
    } while (forbid_adjacent_red && has_adjacent_red(board));

    using P = PortType;
    board.port_type = {P::BRICK, P::LUMBER, P::WOOL, P::GRAIN, P::ORE,
                       P::GENERIC, P::GENERIC, P::GENERIC, P::GENERIC};
    shuffle(board.port_type, seed, index);

    finalize(board);
    return board;
}

GameState initial_state(const Board& board) {
    return initial_state(board.desert_hex);
}

}  // namespace catan
