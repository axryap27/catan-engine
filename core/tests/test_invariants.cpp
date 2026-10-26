#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

#include "catan/invariants.h"
#include "catan/rules.h"
#include "catan/topology.h"
#include "support/fixtures.h"

using namespace std;
using namespace catan;
using namespace catan::test;

namespace {

// A valid position a few dozen decisions into a random game, put into MAIN for seat 0.
struct Mid {
    Board board;
    GameState s;
    explicit Mid(uint64_t seed = 7, int steps = 40) : board(random_board(seed)) {
        s = random_setup(board, seed);
        play_random(board, s, seed, steps);
        set_main(s, 0);
        REQUIRE(invariant_violation(board, s) == nullptr);
    }
};

// Asserts the state is rejected with exactly `msg`.
void expect(const Board& board, const GameState& s, const char* msg) {
    const char* why = invariant_violation(board, s);
    REQUIRE(why != nullptr);
    CHECK(string(why) == msg);
}

// Edges of a simple vertex path with n edges from vertex `from`, found by DFS.
bool find_path(int v, int n, vector<int>& visited, vector<uint8_t>& edges) {
    if (n == 0) return true;
    const Topology& topo = topology();
    visited.push_back(v);
    for (uint8_t e : topo.vertex_edges[v]) {
        if (e == NONE) continue;
        const int w = topo.edge_vertices[e][0] == v ? topo.edge_vertices[e][1]
                                                    : topo.edge_vertices[e][0];
        bool seen = false;
        for (int u : visited) seen = seen || u == w;
        if (seen) continue;
        edges.push_back(e);
        if (find_path(w, n - 1, visited, edges)) return true;
        edges.pop_back();
    }
    visited.pop_back();
    return false;
}

}  // namespace

TEST_CASE("fresh states are consistent", "[invariants]") {
    for (uint64_t seed = 0; seed < 50; ++seed) {
        const Board board = random_board(seed);
        REQUIRE(invariant_violation(board, initial_state(board)) == nullptr);
    }
}

TEST_CASE("random games never violate an invariant", "[invariants]") {
    for (uint64_t seed = 0; seed < 50; ++seed) {
        const Board board = random_board(seed);
        GameState s = initial_state(board);
        int steps = 0;
        play_random(board, s, seed, 1 << 30,
                    [&](const GameState&, const Decision&, const GameState& after,
                        const EventList&) {
                        ++steps;
                        const char* why = invariant_violation(board, after);
                        INFO("seed " << seed << " ply " << after.pub.ply << " "
                                     << (why ? why : ""));
                        REQUIRE(why == nullptr);
                    });
        REQUIRE(steps > 0);
    }
}

TEST_CASE("resource violations", "[invariants]") {
    Mid m;
    SECTION("bank without hand") {
        m.s.pub.bank[0] = static_cast<uint8_t>(m.s.pub.bank[0] - 1);
        expect(m.board, m.s, "resource conservation");
    }
    SECTION("card from nowhere") {
        m.s.hid.hands[1][2] = static_cast<uint8_t>(m.s.hid.hands[1][2] + 1);
        m.s.pub.hand_size[1] = static_cast<uint8_t>(m.s.pub.hand_size[1] + 1);
        expect(m.board, m.s, "resource conservation");
    }
    SECTION("hand_size off by one") {
        m.s.pub.hand_size[2] = static_cast<uint8_t>(m.s.pub.hand_size[2] + 1);
        expect(m.board, m.s, "hand_size mismatch");
    }
}

TEST_CASE("dev card violations", "[invariants]") {
    Mid m;
    SECTION("deck count only") {
        m.s.pub.dev_deck_remaining = static_cast<uint8_t>(m.s.pub.dev_deck_remaining - 1);
        expect(m.board, m.s, "dev_deck_remaining mismatch");
    }
    SECTION("played non-knight cards may leave circulation") {
        --m.s.hid.dev_deck[2];
        --m.s.pub.dev_deck_remaining;
        REQUIRE(invariant_violation(m.board, m.s) == nullptr);
    }
    SECTION("dev_count off") {
        ++m.s.pub.dev_count[3];
        expect(m.board, m.s, "dev_count mismatch");
    }
    SECTION("knight vanishes") {
        REQUIRE(m.s.hid.dev_deck[0] > 0);
        --m.s.hid.dev_deck[0];
        --m.s.pub.dev_deck_remaining;
        expect(m.board, m.s, "knight conservation");
    }
    SECTION("extra victory point card") {
        ++m.s.hid.dev_deck[1];
        ++m.s.pub.dev_deck_remaining;
        expect(m.board, m.s, "dev card conservation");
    }
    SECTION("bought more than held") {
        int c = 0;
        while (m.s.hid.dev_held[0][c] != 0) ++c;
        m.s.hid.dev_bought_this_turn[c] = 1;
        expect(m.board, m.s, "dev_bought_this_turn");
    }
    SECTION("bought by a non-current seat is fine") {
        m.s.hid.dev_bought_this_turn.fill(0);
        REQUIRE(invariant_violation(m.board, m.s) == nullptr);
    }
}

TEST_CASE("piece and board violations", "[invariants]") {
    Mid m;
    SECTION("road piece") {
        --m.s.pub.pieces_left[1][PIECE_ROAD];
        expect(m.board, m.s, "road pieces");
    }
    SECTION("settlement piece") {
        ++m.s.pub.pieces_left[2][PIECE_SETTLEMENT];
        expect(m.board, m.s, "settlement pieces");
    }
    SECTION("city piece") {
        --m.s.pub.pieces_left[3][PIECE_CITY];
        expect(m.board, m.s, "city pieces");
    }
    SECTION("adjacent settlements") {
        int v = 0;
        while (!vertex_occupied(m.s.pub.vertex[v])) ++v;
        uint8_t n = NONE;
        for (uint8_t w : topology().vertex_vertices[v]) {
            if (w != NONE && !vertex_occupied(m.s.pub.vertex[w])) n = w;
        }
        REQUIRE(n != NONE);
        place_settlement(m.s, 0, n);
        expect(m.board, m.s, "distance rule");
    }
    SECTION("road on two players") {
        int e = 0;
        while (!edge_set_has(m.s.pub.roads[0], e)) ++e;
        place_road(m.s, 1, e);
        expect(m.board, m.s, "road owned twice");
    }
    SECTION("edge out of range") {
        m.s.pub.roads[0][1] |= uint64_t{1} << 63;
        expect(m.board, m.s, "edge out of range");
    }
    SECTION("malformed vertex") {
        int v = 0;
        while (vertex_occupied(m.s.pub.vertex[v])) ++v;
        m.s.pub.vertex[v] = 4;  // owner bits without the occupied bit
        expect(m.board, m.s, "malformed vertex");
    }
}

TEST_CASE("victory point violations", "[invariants]") {
    Mid m;
    SECTION("public_vp too high") {
        ++m.s.pub.public_vp[0];
        expect(m.board, m.s, "public_vp mismatch");
    }
    SECTION("public_vp too low") {
        --m.s.pub.public_vp[3];
        expect(m.board, m.s, "public_vp mismatch");
    }
    SECTION("city counted once") {
        int v = 0;
        while (!vertex_occupied(m.s.pub.vertex[v]) || vertex_is_city(m.s.pub.vertex[v])) ++v;
        const uint8_t p = vertex_owner(m.s.pub.vertex[v]);
        m.s.pub.vertex[v] = make_city(p);
        --m.s.pub.pieces_left[p][PIECE_CITY];
        ++m.s.pub.pieces_left[p][PIECE_SETTLEMENT];
        expect(m.board, m.s, "public_vp mismatch");
    }
}

TEST_CASE("award violations", "[invariants]") {
    Mid m;
    SECTION("road_len stale") {
        ++m.s.pub.road_len[0];
        expect(m.board, m.s, "road_len stale");
    }
    SECTION("longest road holder below 5") {
        clear_board(m.s);
        m.s.pub.longest_road_holder = 2;
        m.s.pub.public_vp[2] = static_cast<uint8_t>(m.s.pub.public_vp[2] + 2);
        expect(m.board, m.s, "longest road holder below 5");
    }
    SECTION("longest road holder out of range") {
        m.s.pub.longest_road_holder = 9;
        expect(m.board, m.s, "longest road holder out of range");
    }
    SECTION("longest road holder not longest") {
        clear_board(m.s);
        vector<int> visited;
        vector<uint8_t> path;
        REQUIRE(find_path(0, 11, visited, path));
        for (int i = 0; i < 11; ++i) place_road(m.s, i < 5 ? 0 : 1, path[i]);
        refresh_awards(m.s);
        REQUIRE(m.s.pub.road_len[0] == 5);
        REQUIRE(m.s.pub.road_len[1] == 6);
        REQUIRE(m.s.pub.longest_road_holder == 1);
        REQUIRE(invariant_violation(m.board, m.s) == nullptr);
        m.s.pub.longest_road_holder = 0;
        m.s.pub.public_vp[1] = static_cast<uint8_t>(m.s.pub.public_vp[1] - 2);
        m.s.pub.public_vp[0] = static_cast<uint8_t>(m.s.pub.public_vp[0] + 2);
        expect(m.board, m.s, "longest road holder not longest");
    }
    SECTION("unheld longest road with a long road is allowed") {
        clear_board(m.s);
        vector<int> visited;
        vector<uint8_t> path;
        REQUIRE(find_path(0, 10, visited, path));
        for (int i = 0; i < 10; ++i) place_road(m.s, i < 5 ? 0 : 1, path[i]);
        refresh_awards(m.s);  // tie at 5: set aside
        REQUIRE(m.s.pub.longest_road_holder == NONE);
        REQUIRE(invariant_violation(m.board, m.s) == nullptr);
    }
    SECTION("tied longest road holder is allowed") {
        clear_board(m.s);
        vector<int> visited;
        vector<uint8_t> path;
        REQUIRE(find_path(0, 10, visited, path));
        for (int i = 0; i < 10; ++i) place_road(m.s, i < 5 ? 0 : 1, path[i]);
        refresh_awards(m.s);
        m.s.pub.longest_road_holder = 0;
        m.s.pub.public_vp[0] = static_cast<uint8_t>(m.s.pub.public_vp[0] + 2);
        REQUIRE(invariant_violation(m.board, m.s) == nullptr);
    }
    SECTION("largest army holder with 2 knights") {
        REQUIRE(m.s.hid.dev_deck[0] >= 2);
        m.s.hid.dev_deck[0] = static_cast<uint8_t>(m.s.hid.dev_deck[0] - 2);
        m.s.pub.dev_deck_remaining = static_cast<uint8_t>(m.s.pub.dev_deck_remaining - 2);
        m.s.pub.knights_played[0] = 2;
        m.s.pub.largest_army_holder = 0;
        m.s.pub.public_vp[0] = static_cast<uint8_t>(m.s.pub.public_vp[0] + 2);
        expect(m.board, m.s, "largest army holder below 3 knights");
    }
    SECTION("3 knights but nobody holds the army") {
        REQUIRE(m.s.hid.dev_deck[0] >= 3);
        m.s.hid.dev_deck[0] = static_cast<uint8_t>(m.s.hid.dev_deck[0] - 3);
        m.s.pub.dev_deck_remaining = static_cast<uint8_t>(m.s.pub.dev_deck_remaining - 3);
        m.s.pub.knights_played[1] = 3;
        expect(m.board, m.s, "largest army unclaimed");
    }
    SECTION("largest army holder outranked") {
        REQUIRE(m.s.hid.dev_deck[0] >= 7);
        m.s.hid.dev_deck[0] = static_cast<uint8_t>(m.s.hid.dev_deck[0] - 7);
        m.s.pub.dev_deck_remaining = static_cast<uint8_t>(m.s.pub.dev_deck_remaining - 7);
        m.s.pub.knights_played[0] = 3;
        m.s.pub.knights_played[1] = 4;
        m.s.pub.largest_army_holder = 0;
        m.s.pub.public_vp[0] = static_cast<uint8_t>(m.s.pub.public_vp[0] + 2);
        expect(m.board, m.s, "largest army holder not largest");
    }
}

TEST_CASE("phase bookkeeping violations", "[invariants]") {
    Mid m;
    SECTION("pending_discard outside DISCARD") {
        m.s.pub.pending_discard[1] = 1;
        expect(m.board, m.s, "pending_discard outside DISCARD");
    }
    SECTION("DISCARD with nobody owing") {
        m.s.pub.phase = Phase::DISCARD;
        m.s.pub.dice = 7;
        expect(m.board, m.s, "DISCARD with nobody owing");
    }
    SECTION("free_roads outside ROAD_BUILDING") {
        m.s.pub.free_roads = 1;
        expect(m.board, m.s, "free_roads outside ROAD_BUILDING");
    }
    SECTION("yop_remaining outside YOP_PICK") {
        m.s.pub.yop_remaining = 1;
        expect(m.board, m.s, "yop_remaining outside YOP_PICK");
    }
    SECTION("exchange_give set in MAIN") {
        m.s.pub.exchange_give = 0;
        expect(m.board, m.s, "exchange_give mismatch");
    }
    SECTION("EXCHANGE_RECV without exchange_give") {
        m.s.pub.phase = Phase::EXCHANGE_RECV;
        expect(m.board, m.s, "exchange_give mismatch");
    }
    SECTION("steal_victim with no chance") {
        m.s.pub.steal_victim = 1;
        expect(m.board, m.s, "steal_victim mismatch");
    }
    SECTION("STEAL chance without victim") {
        m.s.pub.pending_chance = static_cast<uint8_t>(ChanceKind::STEAL);
        expect(m.board, m.s, "steal_victim mismatch");
    }
    SECTION("pending_chance out of range") {
        m.s.pub.pending_chance = 9;
        expect(m.board, m.s, "pending_chance out of range");
    }
    SECTION("current_player out of range") {
        m.s.pub.current_player = 4;
        expect(m.board, m.s, "current_player out of range");
    }
    SECTION("robber_hex out of range") {
        m.s.pub.robber_hex = NUM_HEXES;
        expect(m.board, m.s, "robber_hex out of range");
    }
}

TEST_CASE("dice violations", "[invariants]") {
    Mid m;
    SECTION("MAIN without a roll") {
        m.s.pub.dice = 0;
        expect(m.board, m.s, "dice out of range");
    }
    SECTION("dice above 12") {
        m.s.pub.dice = 13;
        expect(m.board, m.s, "dice out of range");
    }
    SECTION("ROLL with dice set") {
        m.s.pub.phase = Phase::ROLL;
        m.s.pub.dice = 5;
        expect(m.board, m.s, "dice set before roll");
    }
    SECTION("SETUP with dice set") {
        const GameState fresh = initial_state(m.board);
        GameState s = fresh;
        s.pub.dice = 3;
        expect(m.board, s, "dice set during SETUP");
    }
    SECTION("setup_step out of range") {
        GameState s = initial_state(m.board);
        s.pub.setup_step = 16;
        expect(m.board, s, "setup_step out of range");
    }
    SECTION("MOVE_ROBBER before the roll is fine") {
        m.s.pub.phase = Phase::MOVE_ROBBER;
        m.s.pub.dice = 0;
        REQUIRE(invariant_violation(m.board, m.s) == nullptr);
    }
}

TEST_CASE("chance and liveness violations", "[invariants]") {
    Mid m;
    SECTION("STEAL with nobody to rob") {
        clear_board(m.s);
        m.s.pub.phase = Phase::STEAL;
        expect(m.board, m.s, "no legal action");
    }
    SECTION("terminal states skip the liveness check") {
        clear_board(m.s);
        m.s.pub.phase = Phase::TERMINAL;
        REQUIRE(invariant_violation(m.board, m.s) == nullptr);
    }
    SECTION("STEAL chance on an empty hand") {
        clear_board(m.s);
        clear_hands(m.s);
        m.s.pub.phase = Phase::STEAL;
        m.s.pub.pending_chance = static_cast<uint8_t>(ChanceKind::STEAL);
        m.s.pub.steal_victim = 1;
        expect(m.board, m.s, "steal from empty hand");
    }
    SECTION("dev draw from an empty deck") {
        m.s.pub.pending_chance = static_cast<uint8_t>(ChanceKind::DEV_DRAW);
        m.s.pub.dev_deck_remaining = 0;
        m.s.hid.dev_deck.fill(0);
        m.s.hid.dev_held[1] = {14, 5, 2, 2, 2};
        m.s.pub.dev_count[1] = 25;
        m.s.pub.knights_played.fill(0);
        expect(m.board, m.s, "dev draw from empty deck");
    }
}
