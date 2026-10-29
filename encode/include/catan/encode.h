#pragma once
#include <array>
#include <cstdint>

#include "catan/belief.h"
#include "catan/board.h"
#include "catan/decision.h"
#include "catan/state.h"

// Turns a state, from one player's point of view, into the network's input: a fixed-size
// uint8 buffer (converted to float on the GPU), plus a separate legal-move mask buffer.
//
// The observer sees only public state, its belief row, and its own hidden cards. The private
// part is passed as a PrivateView, so the encoder cannot read another player's hand or the
// deck even by mistake. Invariant: encode(resample_hidden(s, belief, o), o) == encode(s, o).
//
// Seats are relative to the observer: rel 0 is the observer, rel k is the seat k places
// clockwise after it. Hexes, vertices and edges keep their absolute ids (the board is never
// rotated). Rotating every seat of a state by one and the observer with it leaves the
// encoding unchanged.
//
// Every value is a small non-negative count or a 0/1 flag; "one-hot" groups have exactly one
// byte set unless noted "or none". Layout changes bump ENCODE_VERSION.

namespace catan {

inline constexpr uint32_t ENCODE_VERSION = 1;

// Dice tokens in one-hot order: 2..6 then 8..12 (7 is never a token).
inline constexpr std::array<uint8_t, 10> TOKEN_VALUES = {2, 3, 4, 5, 6, 8, 9, 10, 11, 12};

// ---- Tile block: NUM_HEXES rows of TILE_FEATURES, row h is hex h --------------------------
//
// Corner i and side i follow topology.h (corners clockwise from the top; side i joins corners
// i and i + 1). A building or road appears in every tile it touches (up to 3 and 2 times).
// In the standard frame each tile borders at most one harbor.
inline constexpr int TILE_HEX_TYPE = 0;      // one-hot 6: HexType (BRICK..ORE, DESERT)
inline constexpr int TILE_TOKEN = 6;         // one-hot 10 or none (desert): TOKEN_VALUES order
inline constexpr int TILE_PIPS = 16;         // 0..5: dice combinations producing (6 and 8: 5)
inline constexpr int TILE_ROBBER = 17;       // flag
inline constexpr int TILE_CORNERS = 18;      // 6 corners x 4 rel owners x {settlement, city}:
                                             //   byte TILE_CORNERS + 8 * corner + 2 * rel + city
inline constexpr int TILE_ROADS = 66;        // 6 sides x 4 rel owners: TILE_ROADS + 4 * side + rel
inline constexpr int TILE_PORT_TYPE = 90;    // one-hot 6 or none: PortType of this tile's harbor
inline constexpr int TILE_PORT_SIDE = 96;    // one-hot 6 or none: which side the harbor is on
inline constexpr int TILE_FEATURES = 102;

// ---- Self block: the observer's own hidden cards ------------------------------------------
inline constexpr int HAND_BUCKETS = 8;  // 0, 1, 2, 3, 4, 5, 6-7, 8+
inline constexpr int SELF_HAND_BUCKET = 0;   // 5 resources x one-hot 8: + 8 * res + bucket
inline constexpr int SELF_HAND = 40;         // 5: exact count per resource
inline constexpr int SELF_DEV_HELD = 45;     // 5: per DevCard (includes cards bought this turn)
inline constexpr int SELF_DEV_BOUGHT = 50;   // 5: per DevCard bought this turn (0 unless own turn)
inline constexpr int SELF_TOTAL_VP = 55;     // public VP + VP cards held
inline constexpr int SELF_FEATURES = 56;

// ---- Player blocks: NUM_PLAYERS rows of PLAYER_FEATURES, row k is relative seat k ----------
// Public facts plus the observer's belief. Row 0 is the observer, whose bounds are exact.
inline constexpr int PLAYER_HAND_MIN = 0;     // 5: fewest cards of each resource possible
inline constexpr int PLAYER_HAND_MAX = 5;     // 5: most possible (hand_max)
inline constexpr int PLAYER_HAND_SIZE = 10;
inline constexpr int PLAYER_DEV_COUNT = 11;
inline constexpr int PLAYER_KNIGHTS = 12;
inline constexpr int PLAYER_PUBLIC_VP = 13;
inline constexpr int PLAYER_ROAD_LEN = 14;
inline constexpr int PLAYER_PIECES_LEFT = 15;   // 3: PIECE_* order
inline constexpr int PLAYER_PENDING_DISCARD = 18;
inline constexpr int PLAYER_LONGEST_ROAD = 19;  // flag: holds the award
inline constexpr int PLAYER_LARGEST_ARMY = 20;  // flag: holds the award
inline constexpr int PLAYER_IS_CURRENT = 21;    // flag: it is this player's turn
inline constexpr int PLAYER_VP_AT_TURN_END = 22;  // belief: public VP after their last turn
inline constexpr int PLAYER_PORTS = 23;         // 6 flags: owns a building on a PortType harbor
// Cards received per resource per roll from buildings (settlement 1, city 2), excluding the
// hex under the robber: + 10 * res + roll index (TOKEN_VALUES order).
inline constexpr int PLAYER_PRODUCTION = 29;
inline constexpr int PLAYER_FEATURES = 79;

// ---- Global block --------------------------------------------------------------------------
inline constexpr int GLOBAL_PHASE = 0;          // one-hot 11: Phase
inline constexpr int GLOBAL_DICE = 11;          // one-hot 11 or none: last roll 2..12 this turn
inline constexpr int GLOBAL_BANK = 22;          // 5
inline constexpr int GLOBAL_DEV_DECK = 27;      // cards left in the deck
inline constexpr int GLOBAL_DEV_PLAYED = 28;    // 5: per DevCard, all players (belief)
inline constexpr int GLOBAL_BOUGHT_THIS_TURN = 33;  // dev cards the current player bought
inline constexpr int GLOBAL_PLAYED_DEV = 34;    // flag: current player played a dev card
inline constexpr int GLOBAL_FREE_ROADS = 35;
inline constexpr int GLOBAL_YOP_REMAINING = 36;
inline constexpr int GLOBAL_EXCHANGE_GIVE = 37;  // one-hot 5 or none (only in EXCHANGE_RECV)
inline constexpr int GLOBAL_DISCARD_FLOOR = 42;  // 0..4 (only meaningful in DISCARD)
inline constexpr int GLOBAL_SETUP_STEP = 43;     // 0..15 during setup, 16 after
inline constexpr int GLOBAL_TURN = 44;           // min(turn, 255)
inline constexpr int GLOBAL_FEATURES = 45;

// ---- Whole observation ---------------------------------------------------------------------
inline constexpr int OBS_TILES = 0;
inline constexpr int OBS_SELF = OBS_TILES + NUM_HEXES * TILE_FEATURES;          // 1938
inline constexpr int OBS_PLAYERS = OBS_SELF + SELF_FEATURES;                    // 1994
inline constexpr int OBS_GLOBAL = OBS_PLAYERS + NUM_PLAYERS * PLAYER_FEATURES;  // 2310
inline constexpr int OBS_SIZE = OBS_GLOBAL + GLOBAL_FEATURES;                   // 2355

using Observation = std::array<uint8_t, OBS_SIZE>;

// ---- Legal-move mask: one flag per head entry ----------------------------------------------
// Vertex ids for BUILD_SETTLEMENT and BUILD_CITY have separate masks (the vertex head takes the
// type as input). The resource mask belongs to whichever type uses the resource head in this
// phase. All zero for a non-PLAYER decision. Bits beyond a head's size are ignored.
inline constexpr int MASK_TYPE = 0;                              // 13: ActionType
inline constexpr int MASK_SETTLEMENT = MASK_TYPE + 13;           // 54: vertex
inline constexpr int MASK_CITY = MASK_SETTLEMENT + NUM_VERTICES; // 54: vertex
inline constexpr int MASK_EDGE = MASK_CITY + NUM_VERTICES;       // 72
inline constexpr int MASK_TILE = MASK_EDGE + NUM_EDGES;          // 19
inline constexpr int MASK_DEVCARD = MASK_TILE + NUM_HEXES;       // 5
inline constexpr int MASK_PLAYER = MASK_DEVCARD + NUM_DEV;       // 3: relative to the decider
inline constexpr int MASK_RESOURCE = MASK_PLAYER + 3;            // 5
inline constexpr int MASK_SIZE = MASK_RESOURCE + NUM_RES;        // 225

using MaskBuffer = std::array<uint8_t, MASK_SIZE>;

// Section table for Python and debugging: name, byte offset, rows, bytes per row.
struct ObsSection {
    const char* name;
    uint16_t offset;
    uint16_t rows;
    uint16_t width;
};
inline constexpr std::array<ObsSection, 4> OBS_SECTIONS = {{
    {"tiles", OBS_TILES, NUM_HEXES, TILE_FEATURES},
    {"self", OBS_SELF, 1, SELF_FEATURES},
    {"players", OBS_PLAYERS, NUM_PLAYERS, PLAYER_FEATURES},
    {"global", OBS_GLOBAL, 1, GLOBAL_FEATURES},
}};

// The only hidden data an observer may see: its own cards.
struct PrivateView {
    std::array<uint8_t, NUM_RES> hand;
    std::array<uint8_t, NUM_DEV> dev_held;
    std::array<uint8_t, NUM_DEV> dev_bought_this_turn;  // zero unless the observer is current
};

PrivateView private_view(const GameState& state, uint8_t observer);

// Writes the observation of `observer` into `out` (every byte is written). Pure; reads the
// board, public state, the observer's belief row and `own` only.
void encode(const Board& board, const PublicState& pub, const BeliefState& belief,
            const PrivateView& own, uint8_t observer, Observation& out);

// Convenience: encode(board, state.pub, belief, private_view(state, observer), observer, out).
void encode(const Board& board, const GameState& state, const BeliefState& belief,
            uint8_t observer, Observation& out);

// Expands a decision's mask into one flag per head entry (every byte is written).
void encode_mask(const Decision& decision, MaskBuffer& out);

}  // namespace catan
