#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <cstring>
#include <numeric>

#include "catan/state.h"

using namespace std;
using namespace catan;

TEST_CASE("state: initial_state(7)", "[state]") {
    GameState gs = initial_state(7);
    PublicState& p = gs.pub;

    // Check bank: 19 of each resource
    for (int i = 0; i < NUM_RES; ++i) {
        REQUIRE(p.bank[i] == RESOURCE_PER_TYPE);
    }

    // Check all hand sizes are 0
    for (int i = 0; i < NUM_PLAYERS; ++i) {
        REQUIRE(p.hand_size[i] == 0);
    }

    // Check hand_size is the same as the global constant
    REQUIRE(accumulate(p.hand_size.begin(), p.hand_size.end(), 0) == 0);

    // Check pieces for each player: 15 roads, 5 settlements, 4 cities
    for (int i = 0; i < NUM_PLAYERS; ++i) {
        REQUIRE(p.pieces_left[i][PIECE_ROAD] == ROADS_PER_PLAYER);
        REQUIRE(p.pieces_left[i][PIECE_SETTLEMENT] == SETTLEMENTS_PER_PLAYER);
        REQUIRE(p.pieces_left[i][PIECE_CITY] == CITIES_PER_PLAYER);
    }

    // Check dev deck
    for (int i = 0; i < NUM_DEV; ++i) {
        REQUIRE(gs.hid.dev_deck[i] == DEV_DECK_COUNTS[i]);
    }
    // Verify it sums to DEV_DECK_SIZE
    uint8_t dev_sum = accumulate(gs.hid.dev_deck.begin(), gs.hid.dev_deck.end(), 0u);
    REQUIRE(dev_sum == DEV_DECK_SIZE);

    // Check dev_deck_remaining
    REQUIRE(p.dev_deck_remaining == DEV_DECK_SIZE);

    // Check phase and setup_step
    REQUIRE(p.phase == Phase::SETUP);
    REQUIRE(p.setup_step == 0);

    // Check current_player and ply/turn
    REQUIRE(p.current_player == 0);
    REQUIRE(p.ply == 0);
    REQUIRE(p.turn == 0);

    // Check robber_hex
    REQUIRE(p.robber_hex == 7);

    // Check longest_road_holder and largest_army_holder are NONE
    REQUIRE(p.longest_road_holder == NONE);
    REQUIRE(p.largest_army_holder == NONE);

    // Check exchange_give and trade_target are NONE
    REQUIRE(p.exchange_give == NONE);
    REQUIRE(p.trade_target == NONE);

    // Check pending_chance
    REQUIRE(p.pending_chance == 0);

    // Check all vertices are empty (0)
    for (int i = 0; i < NUM_VERTICES; ++i) {
        REQUIRE(p.vertex[i] == 0);
    }

    // Check all roads are empty
    for (int i = 0; i < NUM_PLAYERS; ++i) {
        REQUIRE(p.roads[i][0] == 0);
        REQUIRE(p.roads[i][1] == 0);
    }

    // Check public_vp for all players is 0
    for (int i = 0; i < NUM_PLAYERS; ++i) {
        REQUIRE(p.public_vp[i] == 0);
    }
}

TEST_CASE("state: vertex helpers round-trip", "[state]") {
    // Test for each player
    for (uint8_t player = 0; player < NUM_PLAYERS; ++player) {
        // Test settlement
        uint8_t settlement_byte = make_settlement(player);
        REQUIRE(vertex_occupied(settlement_byte) == true);
        REQUIRE(vertex_is_city(settlement_byte) == false);
        REQUIRE(vertex_owner(settlement_byte) == player);

        // Test city
        uint8_t city_byte = make_city(player);
        REQUIRE(vertex_occupied(city_byte) == true);
        REQUIRE(vertex_is_city(city_byte) == true);
        REQUIRE(vertex_owner(city_byte) == player);
    }

    // Test empty vertex
    uint8_t empty = 0;
    REQUIRE(vertex_occupied(empty) == false);
    REQUIRE(vertex_is_city(empty) == false);
    REQUIRE(vertex_owner(empty) == NONE);
}

TEST_CASE("state: memcpy equality of initial states", "[state]") {
    GameState gs1 = initial_state(7);
    GameState gs2 = initial_state(7);

    // Compare using memcmp (checking unique object representations)
    int cmp_result = memcmp(&gs1, &gs2, sizeof(GameState));
    REQUIRE(cmp_result == 0);
}
