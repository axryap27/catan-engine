#include <catch2/catch_test_macros.hpp>

#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <vector>

#include "catan/action.h"
#include "catan/board.h"
#include "catan/chance.h"
#include "catan/decision.h"
#include "catan/event.h"
#include "catan/mask.h"
#include "catan/rng.h"
#include "catan/rules.h"
#include "catan/state.h"
#include "catan/topology.h"
#include "catan/types.h"

using namespace std;
using namespace catan;

namespace {

constexpr VertexSet ALL54 = (uint64_t{1} << NUM_VERTICES) - 1;

// Plays the 16 setup steps with a uniform random legal choice per step; returns all events.
vector<Event> play_random_setup(const Board& board, GameState& s, uint64_t seed) {
    vector<Event> events;
    for (int i = 0; i < SETUP_STEPS; ++i) {
        const Decision d = next_decision(board, s);
        REQUIRE(d.kind == DecisionKind::PLAYER);
        REQUIRE(d.n_legal >= 1);
        const int n = static_cast<int>(random_below(seed, 99, s.pub.ply, d.n_legal));
        const EventList ev = apply(board, s, nth_legal(d.mask, d.phase, n));
        for (const Event& e : ev) events.push_back(e);
    }
    return events;
}

int count_edges(const EdgeSet& es) { return popcount(es[0]) + popcount(es[1]); }

// All (type, arg) actions is_legal accepts, by brute force, in canonical order.
vector<Action> brute_force_legal(const ActionMask& m, Phase phase) {
    vector<Action> out;
    for (int t = 0; t < NUM_ACTION_TYPES; ++t)
        for (int a = 0; a < 256; ++a) {
            const Action act{static_cast<ActionType>(t), static_cast<uint8_t>(a)};
            if (is_legal(m, phase, act)) out.push_back(act);
        }
    return out;
}

void check_enumeration(const ActionMask& m, Phase phase, int expected_count) {
    const vector<Action> brute = brute_force_legal(m, phase);
    const int count = count_legal(m, phase);
    REQUIRE(count == expected_count);
    REQUIRE(static_cast<int>(brute.size()) == count);
    for (int n = 0; n < count; ++n) {
        const Action a = nth_legal(m, phase, n);
        CHECK(is_legal(m, phase, a));
        CHECK(a.type == brute[n].type);
        CHECK(a.arg == brute[n].arg);
        if (n > 0) {
            const Action prev = nth_legal(m, phase, n - 1);
            const bool ascending =
                idx(prev.type) < idx(a.type) || (prev.type == a.type && prev.arg < a.arg);
            CHECK(ascending);  // strictly ascending => canonical order and no duplicates
        }
    }
}

ActionMask main_like_mask() {
    ActionMask m;
    set_type(m, ActionType::END_TURN);
    set_type(m, ActionType::BUILD_ROAD);
    for (int e : {0, 5, 63, 64, 71}) edge_set_add(m.edge, e);
    set_type(m, ActionType::BUILD_SETTLEMENT);
    m.settlement = vertex_bit(0) | vertex_bit(10) | vertex_bit(53);
    set_type(m, ActionType::BUILD_CITY);
    m.city = vertex_bit(1) | vertex_bit(2);
    set_type(m, ActionType::BANK_EXCHANGE);
    m.resource = (1u << idx(Resource::BRICK)) | (1u << idx(Resource::ORE));
    set_type(m, ActionType::PLAY_DEV);
    m.devcard = (1u << idx(DevCard::KNIGHT)) | (1u << idx(DevCard::MONOPOLY));
    return m;
}

int resources_total(const GameState& s, int r) {
    int t = s.pub.bank[r];
    for (int p = 0; p < NUM_PLAYERS; ++p) t += s.hid.hands[p][r];
    return t;
}

void check_end_of_setup_invariants(const GameState& s) {
    const Topology& topo = topology();
    for (int r = 0; r < NUM_RES; ++r) CHECK(resources_total(s, r) == RESOURCE_PER_TYPE);
    for (int p = 0; p < NUM_PLAYERS; ++p) {
        int sum = 0;
        for (int r = 0; r < NUM_RES; ++r) sum += s.hid.hands[p][r];
        CHECK(s.pub.hand_size[p] == sum);
    }
    int settlements = 0;
    for (int v = 0; v < NUM_VERTICES; ++v) {
        if (!vertex_occupied(s.pub.vertex[v])) continue;
        ++settlements;
        CHECK(!vertex_is_city(s.pub.vertex[v]));
        for (uint8_t n : topo.vertex_vertices[v])
            if (n != NONE) CHECK(!vertex_occupied(s.pub.vertex[n]));
    }
    CHECK(settlements == 8);
    for (int p = 0; p < NUM_PLAYERS; ++p) {
        CHECK(count_edges(s.pub.roads[p]) == 2);
        for (int e = 0; e < NUM_EDGES; ++e) {
            if (!edge_set_has(s.pub.roads[p], e)) continue;
            const auto& ev = topo.edge_vertices[e];
            CHECK((vertex_owner(s.pub.vertex[ev[0]]) == p || vertex_owner(s.pub.vertex[ev[1]]) == p));
        }
    }
}

ChanceDist dice_dist() {
    ChanceDist d;
    d.kind = ChanceKind::DICE;
    d.n_outcomes = 11;
    d.weight = {1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    d.total = 36;
    return d;
}

}  // namespace

// ---------------------------------------------------------------- mask helpers

TEST_CASE("mask: count_legal and nth_legal match brute force on a MAIN-like mask") {
    check_enumeration(main_like_mask(), Phase::MAIN, 1 + 5 + 3 + 2 + 2 + 2);
}

TEST_CASE("mask: YOP_PICK uses the resource head for PLAY_DEV and ignores devcard bits") {
    ActionMask m;
    set_type(m, ActionType::PLAY_DEV);
    m.resource = (1u << idx(Resource::BRICK)) | (1u << idx(Resource::WOOL)) |
                 (1u << idx(Resource::ORE));
    m.devcard = 0b11011;
    check_enumeration(m, Phase::YOP_PICK, 3);
    CHECK(is_legal(m, Phase::YOP_PICK, {ActionType::PLAY_DEV, idx(Resource::WOOL)}));
    CHECK_FALSE(is_legal(m, Phase::YOP_PICK, {ActionType::PLAY_DEV, idx(Resource::LUMBER)}));
    CHECK(nth_legal(m, Phase::YOP_PICK, 2).arg == idx(Resource::ORE));
}

TEST_CASE("mask: empty mask has no legal actions") {
    const ActionMask m{};
    check_enumeration(m, Phase::MAIN, 0);
    check_enumeration(m, Phase::SETUP, 0);
}

TEST_CASE("mask: single-type and full-head masks enumerate correctly") {
    ActionMask roads;
    set_type(roads, ActionType::BUILD_ROAD);
    for (int e = 0; e < NUM_EDGES; ++e) edge_set_add(roads.edge, e);
    check_enumeration(roads, Phase::MAIN, NUM_EDGES);
    CHECK(nth_legal(roads, Phase::MAIN, 71).arg == 71);

    ActionMask setup;
    set_type(setup, ActionType::BUILD_SETTLEMENT);
    setup.settlement = ALL54;
    check_enumeration(setup, Phase::SETUP, NUM_VERTICES);

    ActionMask forced;
    set_type(forced, ActionType::ROLL);
    check_enumeration(forced, Phase::ROLL, 1);
}

TEST_CASE("mask: is_legal rejects out-of-range args, bad NONE args and cleared types") {
    ActionMask m;
    set_type(m, ActionType::BUILD_SETTLEMENT);
    set_type(m, ActionType::BUILD_ROAD);
    set_type(m, ActionType::MOVE_ROBBER);
    set_type(m, ActionType::END_TURN);
    set_type(m, ActionType::ROLL);
    m.settlement = ~uint64_t{0};
    m.edge = {~uint64_t{0}, ~uint64_t{0}};
    m.tile = ~uint32_t{0};

    CHECK(is_legal(m, Phase::MAIN, {ActionType::BUILD_SETTLEMENT, 53}));
    CHECK_FALSE(is_legal(m, Phase::MAIN, {ActionType::BUILD_SETTLEMENT, 54}));
    CHECK(is_legal(m, Phase::MAIN, {ActionType::BUILD_ROAD, 71}));
    CHECK_FALSE(is_legal(m, Phase::MAIN, {ActionType::BUILD_ROAD, 72}));
    CHECK_FALSE(is_legal(m, Phase::MAIN, {ActionType::BUILD_ROAD, 127}));
    CHECK_FALSE(is_legal(m, Phase::MAIN, {ActionType::BUILD_ROAD, 255}));
    CHECK(is_legal(m, Phase::MOVE_ROBBER, {ActionType::MOVE_ROBBER, 18}));
    CHECK_FALSE(is_legal(m, Phase::MOVE_ROBBER, {ActionType::MOVE_ROBBER, 19}));

    CHECK(is_legal(m, Phase::MAIN, {ActionType::END_TURN, 0}));
    CHECK_FALSE(is_legal(m, Phase::MAIN, {ActionType::END_TURN, 1}));
    CHECK_FALSE(is_legal(m, Phase::ROLL, {ActionType::ROLL, 255}));

    // Types whose bit is clear, even with params present.
    m.city = ALL54;
    CHECK_FALSE(is_legal(m, Phase::MAIN, {ActionType::BUILD_CITY, 0}));
    CHECK_FALSE(is_legal(m, Phase::MAIN, {ActionType::BUY_DEV, 0}));
    CHECK_FALSE(is_legal(m, Phase::MAIN, {ActionType::PROPOSE_TRADE, 0}));
    CHECK_FALSE(is_legal(m, Phase::MAIN, {ActionType::RESPOND_TRADE, 0}));

    // Params cleared with type set.
    ActionMask e;
    set_type(e, ActionType::BUILD_SETTLEMENT);
    CHECK_FALSE(is_legal(e, Phase::SETUP, {ActionType::BUILD_SETTLEMENT, 0}));
}

TEST_CASE("mask: seat_from_relative and relative_from_seat round-trip") {
    for (uint8_t d = 0; d < NUM_PLAYERS; ++d) {
        for (uint8_t rel = 0; rel < NUM_PLAYERS - 1; ++rel) {
            const uint8_t seat = seat_from_relative(d, rel);
            CHECK(seat != d);
            CHECK(seat < NUM_PLAYERS);
            CHECK(relative_from_seat(d, seat) == rel);
        }
        for (uint8_t seat = 0; seat < NUM_PLAYERS; ++seat) {
            if (seat == d) continue;
            const uint8_t rel = relative_from_seat(d, seat);
            CHECK(rel < NUM_PLAYERS - 1);
            CHECK(seat_from_relative(d, rel) == seat);
        }
        CHECK(seat_from_relative(d, 0) == (d + 1) % NUM_PLAYERS);
    }
}

// ---------------------------------------------------------------- setup

TEST_CASE("rules: first decision is player 0 placing a settlement anywhere") {
    vector<Board> boards = {standard_board(), random_board(1), random_board(2), random_board(77)};
    for (const Board& b : boards) {
        const GameState s = initial_state(b);
        const Decision d = next_decision(b, s);
        CHECK(d.kind == DecisionKind::PLAYER);
        CHECK(d.player == 0);
        CHECK(d.phase == Phase::SETUP);
        CHECK(d.mask.type == (1u << idx(ActionType::BUILD_SETTLEMENT)));
        CHECK(d.mask.settlement == ALL54);
        CHECK(d.n_legal == 54);
    }
}

TEST_CASE("rules: road mask after a settlement is its incident edges; distance rule applies") {
    const Board b = standard_board();
    const Topology& topo = topology();
    for (int v = 0; v < NUM_VERTICES; ++v) {
        GameState s = initial_state(b);
        apply(b, s, {ActionType::BUILD_SETTLEMENT, static_cast<uint8_t>(v)});
        const Decision d = next_decision(b, s);
        REQUIRE(d.kind == DecisionKind::PLAYER);
        CHECK(d.player == 0);
        CHECK(d.mask.type == (1u << idx(ActionType::BUILD_ROAD)));
        EdgeSet expected{};
        int n = 0;
        for (uint8_t e : topo.vertex_edges[v])
            if (e != NONE) {
                edge_set_add(expected, e);
                ++n;
            }
        CHECK(d.mask.edge == expected);
        CHECK(d.n_legal == n);

        apply(b, s, nth_legal(d.mask, d.phase, 0));
        const Decision d2 = next_decision(b, s);
        CHECK(d2.player == 1);
        VertexSet blocked = vertex_bit(v);
        for (uint8_t u : topo.vertex_vertices[v])
            if (u != NONE) blocked |= vertex_bit(u);
        CHECK(d2.mask.settlement == (ALL54 & ~blocked));
        CHECK(d2.n_legal == popcount(ALL54 & ~blocked));
    }
}

TEST_CASE("rules: setup follows snake order") {
    const array<uint8_t, 16> order = {0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 2, 2, 1, 1, 0, 0};
    for (uint64_t seed : {1ull, 2ull, 3ull}) {
        const Board b = random_board(seed);
        GameState s = initial_state(b);
        for (int i = 0; i < SETUP_STEPS; ++i) {
            const Decision d = next_decision(b, s);
            CHECK(d.player == order[i]);
            CHECK(s.pub.current_player == order[i]);
            CHECK(setup_player(static_cast<uint8_t>(i)) == order[i]);
            const ActionType want =
                i % 2 == 0 ? ActionType::BUILD_SETTLEMENT : ActionType::BUILD_ROAD;
            CHECK(d.mask.type == (1u << idx(want)));
            const int n = static_cast<int>(random_below(seed, 99, s.pub.ply, d.n_legal));
            apply(b, s, nth_legal(d.mask, d.phase, n));
        }
    }
}

TEST_CASE("rules: second-round road mask covers only the new settlement, minus taken edges") {
    const Board b = standard_board();
    const Topology& topo = topology();
    GameState s = initial_state(b);
    // Round 1 via first-legal placements.
    for (int i = 0; i < 8; ++i) {
        const Decision d = next_decision(b, s);
        apply(b, s, nth_legal(d.mask, d.phase, 0));
    }
    REQUIRE(s.pub.setup_step == 8);
    REQUIRE(s.pub.current_player == 3);

    // Pick a free 3-edge vertex w for P3's second settlement.
    const Decision ds = next_decision(b, s);
    uint8_t w = NONE;
    for (int v = NUM_VERTICES - 1; v >= 0; --v)
        if ((ds.mask.settlement >> v) & 1 && topo.vertex_edges[v][2] != NONE) {
            w = static_cast<uint8_t>(v);
            break;
        }
    REQUIRE(w != NONE);

    // An opponent's (P0) road on one edge at w. Setup alone can't produce this (both ends of a
    // setup road are distance-blocked), so inject it.
    const uint8_t taken_edge = topo.vertex_edges[w][0];
    edge_set_add(s.pub.roads[0], taken_edge);
    --s.pub.pieces_left[0][PIECE_ROAD];  // keep piece counts consistent for the checked build

    apply(b, s, {ActionType::BUILD_SETTLEMENT, w});
    const Decision dr = next_decision(b, s);
    REQUIRE(dr.player == 3);
    EdgeSet expected{};
    for (int k = 1; k < 3; ++k) edge_set_add(expected, topo.vertex_edges[w][k]);
    CHECK(dr.mask.edge == expected);
    CHECK(dr.n_legal == 2);
    CHECK_FALSE(edge_set_has(dr.mask.edge, taken_edge));

    // None of P3's round-1 settlement edges are offered.
    for (int v = 0; v < NUM_VERTICES; ++v) {
        if (v == w || s.pub.vertex[v] != make_settlement(3)) continue;
        for (uint8_t e : topo.vertex_edges[v])
            if (e != NONE) CHECK_FALSE(edge_set_has(dr.mask.edge, e));
    }
}

TEST_CASE("rules: state after setup") {
    vector<Board> boards = {standard_board(), random_board(5), random_board(6)};
    uint64_t seed = 11;
    for (const Board& b : boards) {
        GameState s = initial_state(b);
        play_random_setup(b, s, seed++);
        CHECK(s.pub.phase == Phase::ROLL);
        CHECK(s.pub.current_player == 0);
        CHECK(s.pub.turn == 0);
        CHECK(s.pub.setup_step == 16);
        CHECK(s.pub.ply == 16);
        CHECK(s.pub.pending_chance == 0);
        for (int p = 0; p < NUM_PLAYERS; ++p) {
            int mine = 0;
            for (int v = 0; v < NUM_VERTICES; ++v)
                if (s.pub.vertex[v] == make_settlement(static_cast<uint8_t>(p))) ++mine;
            CHECK(mine == 2);
            CHECK(count_edges(s.pub.roads[p]) == 2);
            CHECK(s.pub.pieces_left[p][PIECE_ROAD] == 13);
            CHECK(s.pub.pieces_left[p][PIECE_SETTLEMENT] == 3);
            CHECK(s.pub.pieces_left[p][PIECE_CITY] == 4);
            CHECK(s.pub.public_vp[p] == 2);
        }
        check_end_of_setup_invariants(s);
    }
}

TEST_CASE("rules: second settlement grants adjacent hex resources; events match") {
    const Topology& topo = topology();
    for (const Board& b : {standard_board(), random_board(3), random_board(4)}) {
        GameState s = initial_state(b);
        vector<Event> events;
        array<uint8_t, NUM_PLAYERS> second{};
        second.fill(NONE);
        for (int i = 0; i < SETUP_STEPS; ++i) {
            const Decision d = next_decision(b, s);
            const Action a = nth_legal(d.mask, d.phase, 0);
            const uint8_t p = s.pub.current_player;
            if (i >= 8 && a.type == ActionType::BUILD_SETTLEMENT) second[p] = a.arg;
            const EventList ev = apply(b, s, a);
            for (const Event& e : ev) events.push_back(e);
            if (i == 7) {
                // Round 1 grants nothing.
                for (int q = 0; q < NUM_PLAYERS; ++q) {
                    CHECK(s.pub.hand_size[q] == 0);
                    for (int r = 0; r < NUM_RES; ++r) CHECK(s.hid.hands[q][r] == 0);
                }
                for (int r = 0; r < NUM_RES; ++r) CHECK(s.pub.bank[r] == RESOURCE_PER_TYPE);
            }
        }
        array<array<int, NUM_RES>, NUM_PLAYERS> gained{};
        for (const Event& e : events) {
            if (e.type != EventType::GAIN) continue;
            REQUIRE(e.player < NUM_PLAYERS);
            REQUIRE(e.a < NUM_RES);
            CHECK(e.b > 0);
            gained[e.player][e.a] += e.b;
        }
        for (int p = 0; p < NUM_PLAYERS; ++p) {
            REQUIRE(second[p] != NONE);
            array<int, NUM_RES> expect{};
            for (uint8_t h : topo.vertex_hexes[second[p]])
                if (h != NONE && b.hex_type[h] != HexType::DESERT)
                    ++expect[static_cast<int>(b.hex_type[h])];
            int sum = 0;
            for (int r = 0; r < NUM_RES; ++r) {
                CHECK(s.hid.hands[p][r] == expect[r]);
                CHECK(gained[p][r] == expect[r]);
                sum += expect[r];
            }
            CHECK(s.pub.hand_size[p] == sum);
        }
        for (int r = 0; r < NUM_RES; ++r) CHECK(resources_total(s, r) == RESOURCE_PER_TYPE);
    }
}

TEST_CASE("rules: setup income counts two hexes of the same type twice") {
    const Board b = standard_board();
    const Topology& topo = topology();
    // Find a vertex touching two hexes of the same resource on the standard board.
    uint8_t target = NONE;
    HexType ht = HexType::DESERT;
    for (int v = 0; v < NUM_VERTICES && target == NONE; ++v) {
        const auto& hs = topo.vertex_hexes[v];
        for (int i = 0; i < 3 && target == NONE; ++i)
            for (int j = i + 1; j < 3; ++j)
                if (hs[i] != NONE && hs[j] != NONE && b.hex_type[hs[i]] == b.hex_type[hs[j]] &&
                    b.hex_type[hs[i]] != HexType::DESERT) {
                    target = static_cast<uint8_t>(v);
                    ht = b.hex_type[hs[i]];
                    break;
                }
    }
    REQUIRE(target != NONE);
    GameState s = initial_state(b);
    for (int i = 0; i < 8; ++i) {  // round 1 away from target
        const Decision d = next_decision(b, s);
        int n = 0;
        while (true) {
            const Action a = nth_legal(d.mask, d.phase, n);
            const bool near = a.type == ActionType::BUILD_SETTLEMENT &&
                              (a.arg == target ||
                               ((topo.vertex_neighbor_mask[target] >> a.arg) & 1));
            if (!near) break;
            ++n;
        }
        apply(b, s, nth_legal(d.mask, d.phase, n));
    }
    REQUIRE(((next_decision(b, s).mask.settlement >> target) & 1) == 1);
    const EventList ev = apply(b, s, {ActionType::BUILD_SETTLEMENT, target});
    CHECK(s.hid.hands[3][static_cast<int>(ht)] >= 2);
    bool found = false;
    for (const Event& e : ev)
        if (e.type == EventType::GAIN && e.a == static_cast<int>(ht)) {
            CHECK(e.b >= 2);
            found = true;
        }
    CHECK(found);
}

TEST_CASE("rules: setup events carry player and id") {
    const Board b = standard_board();
    GameState s = initial_state(b);
    for (int i = 0; i < SETUP_STEPS; ++i) {
        const Decision d = next_decision(b, s);
        const Action a = nth_legal(d.mask, d.phase, d.n_legal - 1);
        const uint8_t p = s.pub.current_player;
        const EventList ev = apply(b, s, a);
        REQUIRE(ev.size >= 1);
        const Event& e = ev.items[0];
        CHECK(e.type == (i % 2 == 0 ? EventType::SETTLEMENT_BUILT : EventType::ROAD_BUILT));
        CHECK(e.player == p);
        CHECK(e.a == a.arg);
        for (int k = 1; k < ev.size; ++k) CHECK(ev.items[k].type == EventType::GAIN);
        if (i < 8) CHECK(ev.size == 1);
    }
}

TEST_CASE("rules: random setup playouts keep invariants") {
    for (uint64_t seed = 0; seed < 2000; ++seed) {
        const Board b = random_board(seed);
        GameState s = initial_state(b);
        for (int i = 0; i < SETUP_STEPS; ++i) {
            const Decision d = next_decision(b, s);
            REQUIRE(d.kind == DecisionKind::PLAYER);
            REQUIRE(d.n_legal >= 1);
            REQUIRE(d.n_legal == count_legal(d.mask, d.phase));
            REQUIRE(d.player == s.pub.current_player);
            const int n = static_cast<int>(random_below(seed, 99, s.pub.ply, d.n_legal));
            apply(b, s, nth_legal(d.mask, d.phase, n));
        }
        REQUIRE(s.pub.phase == Phase::ROLL);
        check_end_of_setup_invariants(s);
    }
}

TEST_CASE("rules: setup is deterministic per board and seed") {
    for (uint64_t seed : {0ull, 9ull, 12345ull}) {
        const Board b = random_board(seed);
        GameState s1 = initial_state(b);
        GameState s2 = initial_state(b);
        const vector<Event> e1 = play_random_setup(b, s1, seed);
        const vector<Event> e2 = play_random_setup(b, s2, seed);
        CHECK(memcmp(&s1, &s2, sizeof(GameState)) == 0);
        REQUIRE(e1.size() == e2.size());
        CHECK(memcmp(e1.data(), e2.data(), e1.size() * sizeof(Event)) == 0);
    }
}

TEST_CASE("rules: TERMINAL phase yields a TERMINAL decision") {
    const Board b = standard_board();
    GameState s = initial_state(b);
    s.pub.phase = Phase::TERMINAL;
    const Decision d = next_decision(b, s);
    CHECK(d.kind == DecisionKind::TERMINAL);
    CHECK(d.player == NONE);
    CHECK(d.n_legal == 0);
}

// ---------------------------------------------------------------- chance

TEST_CASE("chance: pick_outcome walks weights") {
    const ChanceDist d = dice_dist();
    array<int, 11> counts{};
    for (uint64_t r = 0; r < 36; ++r) ++counts[pick_outcome(d, r)];
    for (int i = 0; i < 11; ++i) CHECK(counts[i] == d.weight[i]);
    CHECK(pick_outcome(d, 0) == 0);
    CHECK(pick_outcome(d, 35) == 10);
    CHECK(pick_outcome(d, 36) == 0);  // wraps modulo total

    ChanceDist z;
    z.kind = ChanceKind::STEAL;
    z.n_outcomes = 5;
    z.weight = {0, 3, 0, 2, 0};
    z.total = 5;
    for (uint64_t r = 0; r < 1000; ++r) {
        const uint8_t o = pick_outcome(z, r);
        CHECK((o == 1 || o == 3));
    }
    CHECK(pick_outcome(z, ~uint64_t{0}) != 0);
    CHECK(pick_outcome(z, ~uint64_t{0}) != 4);
}

TEST_CASE("chance: sample_chance is deterministic and keyed on turn for dice") {
    const ChanceDist d = dice_dist();
    const GameState base = initial_state(standard_board());
    int turn_changes = 0;
    for (uint64_t seed = 0; seed < 200; ++seed) {
        GameState s = base;
        s.pub.turn = 7;
        s.pub.ply = 30;
        const uint8_t o = sample_chance(d, s, seed);
        CHECK(sample_chance(d, s, seed) == o);
        GameState s2 = s;
        s2.pub.ply = 31;
        CHECK(sample_chance(d, s2, seed) == o);
        GameState s3 = s;
        s3.pub.turn = 8;
        if (sample_chance(d, s3, seed) != o) ++turn_changes;
    }
    CHECK(turn_changes > 100);
}

TEST_CASE("chance: dice frequencies match 1..6..1 / 36") {
    const ChanceDist d = dice_dist();
    GameState s = initial_state(standard_board());
    s.pub.turn = 3;
    array<int, 11> counts{};
    constexpr int N = 36000;
    for (uint64_t seed = 0; seed < N; ++seed) ++counts[sample_chance(d, s, seed)];
    for (int i = 0; i < 11; ++i) {
        const double expected = N / 36.0 * d.weight[i];
        CHECK(counts[i] > expected * 0.9);
        CHECK(counts[i] < expected * 1.1);
    }
}
