#pragma once

#include <array>
#include <cstdint>
#include <type_traits>

#include "catan/types.h"

namespace catan {

// Game constants.
inline constexpr int RESOURCE_PER_TYPE = 19;
inline constexpr int ROADS_PER_PLAYER = 15;
inline constexpr int SETTLEMENTS_PER_PLAYER = 5;
inline constexpr int CITIES_PER_PLAYER = 4;

// Development deck composition, indexed by DevCard.
inline constexpr std::array<uint8_t, NUM_DEV> DEV_DECK_COUNTS = {
    14,  // KNIGHT
    5,   // VICTORY_POINT
    2,   // ROAD_BUILDING
    2,   // YEAR_OF_PLENTY
    2,   // MONOPOLY
};
inline constexpr int DEV_DECK_SIZE = 25;

// Index into PublicState::pieces_left.
inline constexpr int PIECE_ROAD = 0;
inline constexpr int PIECE_SETTLEMENT = 1;
inline constexpr int PIECE_CITY = 2;

// Vertex byte: bit0 occupied, bit1 city, bits2-3 owner. Empty vertex is 0.
constexpr bool vertex_occupied(uint8_t v) { return (v & 1) != 0; }
constexpr bool vertex_is_city(uint8_t v) { return (v & 2) != 0; }
constexpr uint8_t vertex_owner(uint8_t v) {
    return vertex_occupied(v) ? static_cast<uint8_t>((v >> 2) & 3) : NONE;
}
constexpr uint8_t make_settlement(uint8_t player) {
    return static_cast<uint8_t>(1 | (player << 2));
}
constexpr uint8_t make_city(uint8_t player) {
    return static_cast<uint8_t>(3 | (player << 2));
}

// Everything every player can observe.
struct PublicState {
    std::array<EdgeSet, NUM_PLAYERS> roads;  // 72-bit edge set per player
    uint16_t ply;
    uint16_t turn;
    std::array<uint8_t, NUM_VERTICES> vertex;  // see vertex_* helpers
    std::array<uint8_t, NUM_RES> bank;
    std::array<uint8_t, NUM_PLAYERS> hand_size;  // count only
    std::array<uint8_t, NUM_PLAYERS> dev_count;
    std::array<uint8_t, NUM_PLAYERS> knights_played;
    std::array<uint8_t, NUM_PLAYERS> public_vp;
    std::array<uint8_t, NUM_PLAYERS> road_len;         // every player's longest road
    std::array<uint8_t, NUM_PLAYERS> pending_discard;  // cards still owed after a 7
    std::array<std::array<uint8_t, 3>, NUM_PLAYERS> pieces_left;  // PIECE_* order
    uint8_t dev_deck_remaining;
    uint8_t robber_hex;
    uint8_t current_player;
    Phase phase;
    uint8_t setup_step;           // 0..15 during setup (snake order), 16 after
    uint8_t dice;
    uint8_t longest_road_holder;  // NONE if unclaimed
    uint8_t largest_army_holder;  // NONE if unclaimed
    uint8_t played_dev_this_turn;
    uint8_t free_roads;       // remaining Road Building placements
    uint8_t yop_remaining;    // remaining Year of Plenty picks
    uint8_t exchange_give;    // resource chosen in step 1 of a bank exchange
    uint8_t discard_floor;    // min resource index for the next discard

    // Reserved for trading.
    uint8_t trade_target;
    uint8_t trades_this_turn;
    std::array<int8_t, NUM_RES> open_offer;

    uint8_t pending_chance;  // ChanceKind awaiting apply_chance, 0 (NONE) if none
    uint8_t steal_victim;    // seat being robbed while a STEAL chance is pending, else NONE

    uint8_t reserved_[7]{};  // explicit padding to a multiple of 8
};

// Information only the owning player (or the deck) knows.
struct HiddenState {
    std::array<std::array<uint8_t, NUM_RES>, NUM_PLAYERS> hands;
    std::array<std::array<uint8_t, NUM_DEV>, NUM_PLAYERS> dev_held;
    std::array<uint8_t, NUM_DEV> dev_deck;
    std::array<uint8_t, NUM_DEV> dev_bought_this_turn;  // current player, per card type

    uint8_t reserved_[6]{};  // explicit padding so GameState has no tail padding
};

struct GameState {
    PublicState pub;
    HiddenState hid;
};

// Layout: PublicState = 64 (roads) + 4 (ply, turn) + 54 + 5 + 6*4 + 12 + 4 + 9
// + 2 + 5 + 2 (pending_chance, steal_victim) + 7 reserved = 192. HiddenState = 20 + 20 + 5 + 5 + 6 reserved = 56.
static_assert(sizeof(PublicState) == 192);
static_assert(sizeof(HiddenState) == 56);
static_assert(sizeof(GameState) == 248);
static_assert(sizeof(GameState) <= 512);
static_assert(std::is_trivially_copyable_v<PublicState>);
static_assert(std::is_trivially_copyable_v<HiddenState>);
static_assert(std::is_trivially_copyable_v<GameState>);
static_assert(std::has_unique_object_representations_v<PublicState>);
static_assert(std::has_unique_object_representations_v<HiddenState>);
static_assert(std::has_unique_object_representations_v<GameState>);

// Fresh game at the start of setup; robber_hex is the desert hex.
GameState initial_state(uint8_t robber_hex);

// 64-bit FNV-1a over the state's bytes (no padding, so equal states hash equal). Golden replays
// pin these values; any rules change that alters play changes them.
uint64_t state_hash(const GameState& state);

}  // namespace catan
