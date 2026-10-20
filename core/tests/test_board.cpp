#include <algorithm>
#include <array>
#include <cstdint>
#include <set>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "catan/board.h"
#include "catan/state.h"
#include "catan/topology.h"
#include "catan/types.h"

using namespace std;
using namespace catan;

namespace {

bool same_board(const Board& a, const Board& b) {
    return a.hex_type == b.hex_type && a.token == b.token && a.port_type == b.port_type &&
           a.desert_hex == b.desert_hex && a.vertex_port == b.vertex_port &&
           a.hexes_for_roll == b.hexes_for_roll;
}

bool has_adjacent_reds(const Board& b) {
    const Topology& topo = topology();
    auto red = [&](int h) { return b.token[h] == 6 || b.token[h] == 8; };
    for (int h = 0; h < NUM_HEXES; ++h) {
        if (!red(h)) continue;
        for (uint8_t n : topo.hex_neighbors[h]) {
            if (n != NONE && red(n)) return true;
        }
    }
    return false;
}

void check_hex_counts(const Board& b) {
    array<int, 6> count{};
    for (HexType t : b.hex_type) ++count[static_cast<int>(t)];
    CHECK(count[static_cast<int>(HexType::LUMBER)] == 4);
    CHECK(count[static_cast<int>(HexType::WOOL)] == 4);
    CHECK(count[static_cast<int>(HexType::GRAIN)] == 4);
    CHECK(count[static_cast<int>(HexType::BRICK)] == 3);
    CHECK(count[static_cast<int>(HexType::ORE)] == 3);
    CHECK(count[static_cast<int>(HexType::DESERT)] == 1);

    REQUIRE(b.desert_hex < NUM_HEXES);
    CHECK(b.hex_type[b.desert_hex] == HexType::DESERT);
    CHECK(b.token[b.desert_hex] == 0);
}

void check_tokens(const Board& b) {
    vector<int> tokens;
    for (int h = 0; h < NUM_HEXES; ++h) {
        if (b.hex_type[h] != HexType::DESERT) tokens.push_back(b.token[h]);
    }
    sort(tokens.begin(), tokens.end());
    const vector<int> expected = {2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12};
    CHECK(tokens.size() == 18);
    CHECK(tokens == expected);
}

void check_ports(const Board& b) {
    array<int, 6> count{};
    for (PortType p : b.port_type) ++count[static_cast<int>(p)];
    CHECK(count[static_cast<int>(PortType::GENERIC)] == 4);
    CHECK(count[static_cast<int>(PortType::BRICK)] == 1);
    CHECK(count[static_cast<int>(PortType::LUMBER)] == 1);
    CHECK(count[static_cast<int>(PortType::WOOL)] == 1);
    CHECK(count[static_cast<int>(PortType::GRAIN)] == 1);
    CHECK(count[static_cast<int>(PortType::ORE)] == 1);
}

void check_lookups(const Board& b) {
    const Topology& topo = topology();

    array<uint8_t, NUM_VERTICES> expected_port;
    expected_port.fill(NONE);
    for (int p = 0; p < NUM_PORTS; ++p) {
        const PortSlot& s = topo.port_slots[p];
        expected_port[s.v0] = static_cast<uint8_t>(b.port_type[p]);
        expected_port[s.v1] = static_cast<uint8_t>(b.port_type[p]);
    }
    int marked = 0;
    for (int v = 0; v < NUM_VERTICES; ++v) {
        CHECK(b.vertex_port[v] == expected_port[v]);
        if (b.vertex_port[v] != NONE) ++marked;
    }
    CHECK(marked == 18);

    for (int r = 0; r < 13; ++r) {
        vector<uint8_t> want;
        for (int h = 0; h < NUM_HEXES; ++h) {
            if (b.token[h] == r && b.hex_type[h] != HexType::DESERT) want.push_back(h);
        }
        vector<uint8_t> got;
        for (uint8_t h : b.hexes_for_roll[r]) {
            if (h != NONE) got.push_back(h);
        }
        sort(got.begin(), got.end());
        CHECK(got == want);
        // NONE-padded: real entries come first.
        if (b.hexes_for_roll[r][0] == NONE) CHECK(b.hexes_for_roll[r][1] == NONE);
        if (r == 0 || r == 1 || r == 7) {
            CHECK(b.hexes_for_roll[r][0] == NONE);
            CHECK(b.hexes_for_roll[r][1] == NONE);
        } else {
            CHECK(want.size() == (r == 2 || r == 12 ? 1u : 2u));
        }
    }
}

void check_all(const Board& b) {
    check_hex_counts(b);
    check_tokens(b);
    check_ports(b);
    check_lookups(b);
}

}  // namespace

TEST_CASE("board: standard hex counts and desert", "[board]") {
    check_hex_counts(standard_board());
}
TEST_CASE("board: standard tokens", "[board]") { check_tokens(standard_board()); }
TEST_CASE("board: standard ports", "[board]") { check_ports(standard_board()); }
TEST_CASE("board: standard derived lookups", "[board]") { check_lookups(standard_board()); }

TEST_CASE("board: standard hex/token layout matches rulebook beginner map", "[board]") {
    // Rulebook "Starting Map for Beginners" (Illustration A), rows top to bottom.
    constexpr HexType B = HexType::BRICK, L = HexType::LUMBER, W = HexType::WOOL,
                      G = HexType::GRAIN, O = HexType::ORE, D = HexType::DESERT;
    const array<HexType, NUM_HEXES> types = {
        O, W, L,
        G, B, W, B,
        G, L, D, L, O,
        L, O, G, W,
        B, G, W,
    };
    const array<uint8_t, NUM_HEXES> tokens = {
        10, 2, 9,
        12, 6, 4, 10,
        9, 11, 0, 3, 8,
        8, 3, 4, 5,
        5, 6, 11,
    };
    const Board b = standard_board();
    CHECK(b.hex_type == types);
    CHECK(b.token == tokens);
    CHECK(b.desert_hex == 9);
}

TEST_CASE("board: standard has no adjacent red numbers", "[board]") {
    CHECK_FALSE(has_adjacent_reds(standard_board()));
}

TEST_CASE("board: random structural invariants over many seeds", "[board]") {
    for (uint64_t seed = 0; seed < 300; ++seed) {
        CAPTURE(seed);
        check_all(random_board(seed));
        check_all(random_board(seed, false));
    }
}

TEST_CASE("board: random_board is deterministic per seed", "[board]") {
    for (uint64_t seed = 0; seed < 100; ++seed) {
        CAPTURE(seed);
        CHECK(same_board(random_board(seed), random_board(seed)));
        CHECK(same_board(random_board(seed, false), random_board(seed, false)));
    }
}

TEST_CASE("board: different seeds give different boards", "[board]") {
    int differing = 0;
    for (uint64_t seed = 0; seed < 100; ++seed) {
        if (!same_board(random_board(seed), random_board(seed + 1))) ++differing;
    }
    CHECK(differing >= 99);
}

TEST_CASE("board: forbid_adjacent_red keeps 6 and 8 apart", "[board]") {
    for (uint64_t seed = 0; seed < 10000; ++seed) {
        const Board b = random_board(seed, true);
        if (has_adjacent_reds(b)) {
            FAIL("adjacent red numbers at seed " << seed);
        }
    }
}

TEST_CASE("board: without forbid_adjacent_red adjacent reds do occur", "[board]") {
    int with_adjacent = 0;
    for (uint64_t seed = 0; seed < 1000; ++seed) {
        if (has_adjacent_reds(random_board(seed, false))) ++with_adjacent;
    }
    CHECK(with_adjacent > 0);
}

TEST_CASE("board: random layouts vary across seeds", "[board]") {
    set<uint8_t> deserts;
    set<uint8_t> hex0_types, port0_types;
    set<uint8_t> token0;
    for (uint64_t seed = 0; seed < 200; ++seed) {
        const Board b = random_board(seed);
        deserts.insert(b.desert_hex);
        hex0_types.insert(static_cast<uint8_t>(b.hex_type[0]));
        port0_types.insert(static_cast<uint8_t>(b.port_type[0]));
        token0.insert(b.token[0]);
    }
    CHECK(deserts.size() >= 10);
    CHECK(hex0_types.size() >= 4);
    CHECK(port0_types.size() >= 4);
    CHECK(token0.size() >= 5);
}

TEST_CASE("board: initial_state puts the robber on the desert in SETUP", "[board]") {
    auto check = [](const Board& b) {
        const GameState s = initial_state(b);
        CHECK(s.pub.robber_hex == b.desert_hex);
        CHECK(s.pub.phase == Phase::SETUP);
    };
    check(standard_board());
    for (uint64_t seed = 0; seed < 50; ++seed) check(random_board(seed));
}

TEST_CASE("board: standard harbors match rulebook beginner map", "[board]") {
    // Clockwise from the top-left slot; see topology.h for slot positions.
    constexpr PortType A = PortType::GENERIC;
    const array<PortType, NUM_PORTS> expected = {
        A, PortType::GRAIN, PortType::ORE, A, PortType::WOOL, A, A, PortType::BRICK,
        PortType::LUMBER,
    };
    CHECK(standard_board().port_type == expected);
}
