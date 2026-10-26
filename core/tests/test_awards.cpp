#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <initializer_list>
#include <vector>

#include "catan/action.h"
#include "catan/board.h"
#include "catan/decision.h"
#include "catan/event.h"
#include "catan/invariants.h"
#include "catan/mask.h"
#include "catan/rng.h"
#include "catan/rules.h"
#include "catan/state.h"
#include "catan/topology.h"
#include "catan/types.h"
#include "support/fixtures.h"
#include "support/reference.h"

using namespace std;
using namespace catan;
using namespace catan::test;

namespace {

const Board& board() {
    static const Board b = standard_board();
    return b;
}

// Player p to act in MAIN on a bare board: no buildings, no hands, no dev cards.
GameState blank(uint8_t p = 0) {
    GameState s = random_setup(board(), 1);
    clear_board(s);
    clear_hands(s);
    set_main(s, p);
    return s;
}

Action act(ActionType t, int arg = 0) { return Action{t, static_cast<uint8_t>(arg)}; }

void check_invariants(const GameState& s) {
    const char* why = invariant_violation(board(), s);
    CHECK(why == nullptr);
}

EventList step(GameState& s, Action a) {
    const Decision d = next_decision(board(), s);
    REQUIRE(d.kind == DecisionKind::PLAYER);
    REQUIRE(is_legal(d.mask, d.phase, a));
    const EventList ev = apply(board(), s, a);
    check_invariants(s);
    return ev;
}

// Pays for and builds a road between adjacent vertices a and b as the current player.
EventList build_road(GameState& s, int a, int b) {
    const uint8_t e = edge_between(a, b);
    REQUIRE(e != NONE);
    give(s, s.pub.current_player, Resource::BRICK, 1);
    give(s, s.pub.current_player, Resource::LUMBER, 1);
    return step(s, act(ActionType::BUILD_ROAD, e));
}

// Places roads along consecutive vertices vs[0..edges] directly (no events, no awards).
void line(GameState& s, uint8_t p, const vector<int>& vs, int edges) {
    for (int i = 0; i < edges; ++i) {
        const uint8_t e = edge_between(vs[i], vs[i + 1]);
        REQUIRE(e != NONE);
        place_road(s, p, e);
    }
}

void give_cities(GameState& s, uint8_t p, initializer_list<int> vs) {
    for (int v : vs) {
        place_settlement(s, p, v);
        place_city(s, p, v);
    }
}

// Three disjoint vertex chains, one per player; the first n edges of each are used.
const vector<int> L0 = {0, 3, 7, 11, 16, 21, 27, 33, 38, 43};
const vector<int> L1 = {1, 5, 9, 14, 19, 25, 31, 37, 42};
const vector<int> L2 = {51, 48, 44, 40, 45, 49, 53, 50};

GameState with_lines(int n0, int n1, int n2, uint8_t mover = 0) {
    GameState s = blank(mover);
    line(s, 0, L0, n0);
    line(s, 1, L1, n1);
    line(s, 2, L2, n2);
    refresh_awards(s);
    return s;
}

// ---- hand-built shapes: everything but the last road is placed directly, the last is built
// through the rules so the engine computes the length.

struct Shape {
    vector<vector<int>> paths;  // vertex chains of pre-placed roads
    int last_a, last_b;         // the road built through the rules
    vector<int> opp;            // settlements of player 1
    vector<int> own;            // settlements of player 0
};

int engine_length(const Shape& sh) {
    GameState s = blank(0);
    for (const auto& path : sh.paths) line(s, 0, path, static_cast<int>(path.size()) - 1);
    for (int v : sh.own) place_settlement(s, 0, v);
    for (int v : sh.opp) place_settlement(s, 1, v);
    refresh_awards(s);
    build_road(s, sh.last_a, sh.last_b);
    CHECK(s.pub.road_len[0] == reference_road_length(s.pub, 0));
    return s.pub.road_len[0];
}

// Hex 9 is ringed by 18-24-30-35-29-23; hex 10 shares its 24-30 edge.
const vector<int> RING = {18, 24, 30, 35, 29, 23};

struct Rnd {
    uint64_t seed;
    uint64_t i = 0;
    int below(int n) {
        return static_cast<int>(random_below(seed, 7, i++, static_cast<uint32_t>(n)));
    }
};

}  // namespace

// ---------------------------------------------------------------- road length shapes

TEST_CASE("awards: a straight road is as long as its edges", "[awards]") {
    const int k = GENERATE(1, 2, 3, 4, 5, 6);
    const vector<int> chain = {0, 3, 7, 12, 17, 23, 29};
    const vector<int> rest(chain.begin() + 1, chain.begin() + k + 1);
    CHECK(engine_length({{rest}, 0, 3, {}, {0}}) == k);
}

TEST_CASE("awards: a Y-shaped network counts the two longest arms", "[awards]") {
    // Arms from 18 of 3 (via 13, 8, 4), 2 (via 23, 17) and 1 (via 24).
    CHECK(engine_length({{{4, 8, 13, 18, 23, 17}}, 18, 24, {}, {}}) == 5);
    // Arms of 3, 3 and 3.
    CHECK(engine_length({{{4, 8, 13, 18, 23, 29, 34}, {18, 24, 19}}, 19, 14, {}, {}}) == 6);
}

TEST_CASE("awards: a closed hex ring is six long", "[awards]") {
    CHECK(engine_length({{{18, 24, 30, 35, 29, 23}}, 23, 18, {}, {}}) == 6);
}

TEST_CASE("awards: a ring with a tail revisits the junction", "[awards]") {
    // Tail 14-19-24 plus the ring through 24: 2 + 6.
    CHECK(engine_length({{{14, 19, 24, 18, 23, 29, 35, 30}}, 30, 24, {}, {}}) == 8);
}

TEST_CASE("awards: two rings sharing an edge use all eleven edges", "[awards]") {
    // Hexes 9 and 10 share 24-30; both junctions have odd degree, so an Euler trail covers
    // all 6 + 5 edges.
    CHECK(engine_length({{{18, 24, 30, 35, 29, 23, 18}, {24, 19, 25, 31, 36}}, 36, 30, {}, {}}) ==
          11);
}

TEST_CASE("awards: separate networks count only the longest", "[awards]") {
    CHECK(engine_length({{{3, 7, 11}, {1, 5, 9, 14, 19}}, 0, 3, {}, {0}}) == 4);
}

TEST_CASE("awards: an opponent's settlement in the middle splits a road", "[awards]") {
    const vector<int> rest = {3, 7, 12, 17, 23, 29};  // plus the last road 0-3
    CHECK(engine_length({{rest}, 0, 3, {12}, {}}) == 3);  // arms 0..12 and 12..29, both 3
    CHECK(engine_length({{rest}, 0, 3, {17}, {}}) == 4);  // arms 0..17 (4) and 17..29 (2)
}

TEST_CASE("awards: an opponent's settlement at an end does not shorten a road", "[awards]") {
    const vector<int> rest = {3, 7, 12, 17, 23, 29};
    CHECK(engine_length({{rest}, 0, 3, {29}, {}}) == 6);
}

TEST_CASE("awards: your own settlement never splits your road", "[awards]") {
    const vector<int> rest = {3, 7, 12, 17, 23, 29};
    CHECK(engine_length({{rest}, 0, 3, {}, {12}}) == 6);
}

TEST_CASE("awards: an opponent at a junction separates the arms", "[awards]") {
    // Opponent on the Y's centre: arms of 3 (4-8-13-18), 2 and 1 can only end there.
    CHECK(engine_length({{{8, 13, 18, 23, 17}, {18, 24}}, 4, 8, {18}, {}}) == 3);
}

TEST_CASE("awards: an opponent on a ring still lets the trail start and end there",
          "[awards]") {
    CHECK(engine_length({{{18, 24, 30, 35, 29, 23}}, 23, 18, {30}, {}}) == 6);
    // Ring with tail 14-19-24 and an opponent on 30: 14-19-24-18-23-29-35-30 is 7.
    CHECK(engine_length({{{14, 19, 24, 18, 23, 29, 35, 30}}, 30, 24, {30}, {}}) == 7);
}

// ---------------------------------------------------------------- Longest Road

TEST_CASE("awards: the first road of five takes Longest Road", "[awards]") {
    GameState s = with_lines(3, 0, 0);
    const EventList e4 = build_road(s, L0[3], L0[4]);
    CHECK(s.pub.road_len[0] == 4);
    CHECK(events_of(e4, EventType::LONGEST_ROAD).empty());
    CHECK(s.pub.longest_road_holder == NONE);
    CHECK(s.pub.public_vp[0] == 0);

    const EventList e5 = build_road(s, L0[4], L0[5]);
    const auto lr = events_of(e5, EventType::LONGEST_ROAD);
    REQUIRE(lr.size() == 1);
    CHECK(lr[0].player == 0);
    CHECK(s.pub.road_len[0] == 5);
    CHECK(s.pub.longest_road_holder == 0);
    CHECK(s.pub.public_vp[0] == 2);

    const EventList e6 = build_road(s, L0[5], L0[6]);  // extending your own record
    CHECK(events_of(e6, EventType::LONGEST_ROAD).empty());
    CHECK(s.pub.public_vp[0] == 2);
}

TEST_CASE("awards: a tie does not take Longest Road, a longer road does", "[awards]") {
    GameState s = with_lines(5, 4, 0, 1);
    REQUIRE(s.pub.longest_road_holder == 0);
    REQUIRE(s.pub.public_vp[0] == 2);

    const EventList tie = build_road(s, L1[4], L1[5]);  // player 1 reaches 5
    CHECK(s.pub.road_len[1] == 5);
    CHECK(events_of(tie, EventType::LONGEST_ROAD).empty());
    CHECK(s.pub.longest_road_holder == 0);
    CHECK(s.pub.public_vp[0] == 2);
    CHECK(s.pub.public_vp[1] == 0);

    const EventList longer = build_road(s, L1[5], L1[6]);  // and 6
    const auto lr = events_of(longer, EventType::LONGEST_ROAD);
    REQUIRE(lr.size() == 1);
    CHECK(lr[0].player == 1);
    CHECK(s.pub.longest_road_holder == 1);
    CHECK(s.pub.public_vp[0] == 0);
    CHECK(s.pub.public_vp[1] == 2);
}

namespace {

// Player 0 holds Longest Road with n0 roads; player 1 (n1 roads, plus a stub 16-22) builds a
// settlement on vertex 16, which sits on player 0's chain.
struct CutResult {
    GameState s;
    EventList ev;
};

CutResult cut_at_16(int n0, int n1, int n2) {
    GameState s = with_lines(n0, n1, n2, 1);
    place_road(s, 1, edge_between(16, 22));
    refresh_awards(s);
    REQUIRE(s.pub.longest_road_holder == 0);
    for (int r : {0, 1, 2, 3}) give(s, 1, static_cast<Resource>(r), 1);
    const EventList ev = step(s, act(ActionType::BUILD_SETTLEMENT, 16));
    return {s, ev};
}

}  // namespace

TEST_CASE("awards: a cut holder who still ties for longest keeps the card", "[awards]") {
    // Player 0: 9 edges, cut at 16 leaves arms of 4 and 5. Player 1 has 5.
    const auto [s, ev] = cut_at_16(9, 5, 0);
    CHECK(s.pub.road_len[0] == 5);
    CHECK(s.pub.road_len[1] == 5);
    CHECK(events_of(ev, EventType::LONGEST_ROAD).empty());
    CHECK(s.pub.longest_road_holder == 0);
    CHECK(s.pub.public_vp[0] == 2);
    CHECK(s.pub.public_vp[1] == 1);
}

TEST_CASE("awards: a cut holder who is still alone on top keeps the card", "[awards]") {
    const auto [s, ev] = cut_at_16(9, 0, 0);
    CHECK(s.pub.road_len[0] == 5);
    CHECK(events_of(ev, EventType::LONGEST_ROAD).empty());
    CHECK(s.pub.longest_road_holder == 0);
}

TEST_CASE("awards: cutting the holder below a single rival passes the card", "[awards]") {
    // Player 0: 7 edges, cut at 16 leaves 4. Player 1 has 5.
    const auto [s, ev] = cut_at_16(7, 5, 0);
    CHECK(s.pub.road_len[0] == 4);
    const auto lr = events_of(ev, EventType::LONGEST_ROAD);
    REQUIRE(lr.size() == 1);
    CHECK(lr[0].player == 1);
    CHECK(s.pub.longest_road_holder == 1);
    CHECK(s.pub.public_vp[0] == 0);
    CHECK(s.pub.public_vp[1] == 3);  // settlement + card
}

TEST_CASE("awards: cutting the holder below two tied rivals sets the card aside", "[awards]") {
    const auto [s, ev] = cut_at_16(7, 5, 5);
    CHECK(s.pub.road_len[0] == 4);
    CHECK(s.pub.road_len[1] == 5);
    CHECK(s.pub.road_len[2] == 5);
    const auto lr = events_of(ev, EventType::LONGEST_ROAD);
    REQUIRE(lr.size() == 1);
    CHECK(lr[0].player == NONE);
    CHECK(s.pub.longest_road_holder == NONE);
    CHECK(s.pub.public_vp[0] == 0);
    CHECK(s.pub.public_vp[1] == 1);
    CHECK(s.pub.public_vp[2] == 0);
}

TEST_CASE("awards: cutting the holder with nobody else at five leaves the card unclaimed",
          "[awards]") {
    const auto [s, ev] = cut_at_16(7, 0, 0);
    CHECK(s.pub.road_len[0] == 4);
    const auto lr = events_of(ev, EventType::LONGEST_ROAD);
    REQUIRE(lr.size() == 1);
    CHECK(lr[0].player == NONE);
    CHECK(s.pub.longest_road_holder == NONE);
    CHECK(s.pub.public_vp[0] == 0);
}

TEST_CASE("awards: a set-aside card goes to the next road that stands alone", "[awards]") {
    auto [s, ev] = cut_at_16(7, 5, 5);
    REQUIRE(s.pub.longest_road_holder == NONE);
    // Player 1 (still the mover) extends to 6 and takes the card.
    const EventList next = build_road(s, L1[5], L1[6]);
    const auto lr = events_of(next, EventType::LONGEST_ROAD);
    REQUIRE(lr.size() == 1);
    CHECK(lr[0].player == 1);
    CHECK(s.pub.public_vp[1] == 3);
}

TEST_CASE("awards: taking Longest Road to ten points wins", "[awards]") {
    GameState s = blank();
    give_cities(s, 0, {2, 10, 20, 32});  // 8 points
    line(s, 0, L0, 4);
    refresh_awards(s);
    const EventList ev = build_road(s, L0[4], L0[5]);
    CHECK(events_of(ev, EventType::LONGEST_ROAD).size() == 1);
    const auto over = events_of(ev, EventType::GAME_OVER);
    REQUIRE(over.size() == 1);
    CHECK(over[0].player == 0);
    CHECK(s.pub.phase == Phase::TERMINAL);
}

TEST_CASE("awards: Longest Road short of ten points does not end the game", "[awards]") {
    GameState s = blank();
    give_cities(s, 0, {2, 10, 20});
    place_settlement(s, 0, 32);  // 7 points
    line(s, 0, L0, 4);
    refresh_awards(s);
    const EventList ev = build_road(s, L0[4], L0[5]);
    CHECK(s.pub.public_vp[0] == 9);
    CHECK(events_of(ev, EventType::GAME_OVER).empty());
    CHECK(s.pub.phase == Phase::MAIN);
}

// ---------------------------------------------------------------- Largest Army

namespace {

// Counts n knights as already played by p (taken from the deck, no award change).
void preplay(GameState& s, uint8_t p, int n) {
    uint8_t& knights_left = s.hid.dev_deck[idx(DevCard::KNIGHT)];
    knights_left = static_cast<uint8_t>(knights_left - n);
    s.pub.dev_deck_remaining = static_cast<uint8_t>(s.pub.dev_deck_remaining - n);
    s.pub.knights_played[p] = static_cast<uint8_t>(s.pub.knights_played[p] + n);
}

// Hands the card to p as of an earlier play.
void set_army(GameState& s, uint8_t p) {
    s.pub.largest_army_holder = p;
    s.pub.public_vp[p] = static_cast<uint8_t>(s.pub.public_vp[p] + 2);
}

// The current player plays a held Knight and moves the robber away. Returns the play events.
EventList play_knight(GameState& s) {
    const EventList ev = step(s, act(ActionType::PLAY_DEV, idx(DevCard::KNIGHT)));
    if (s.pub.phase == Phase::MOVE_ROBBER) {
        const Decision d = next_decision(board(), s);
        step(s, act(ActionType::MOVE_ROBBER, countr_zero(d.mask.tile)));
    }
    return ev;
}

}  // namespace

TEST_CASE("awards: two knights do not make an army", "[awards]") {
    GameState s = blank();
    preplay(s, 0, 1);
    give_dev(s, 0, DevCard::KNIGHT);
    const EventList ev = play_knight(s);
    CHECK(s.pub.knights_played[0] == 2);
    CHECK(events_of(ev, EventType::LARGEST_ARMY).empty());
    CHECK(s.pub.largest_army_holder == NONE);
    CHECK(s.pub.public_vp[0] == 0);
}

TEST_CASE("awards: the third knight takes Largest Army", "[awards]") {
    GameState s = blank();
    preplay(s, 0, 2);
    give_dev(s, 0, DevCard::KNIGHT);
    const EventList ev = play_knight(s);
    const auto la = events_of(ev, EventType::LARGEST_ARMY);
    REQUIRE(la.size() == 1);
    CHECK(la[0].player == 0);
    CHECK(s.pub.knights_played[0] == 3);
    CHECK(s.pub.largest_army_holder == 0);
    CHECK(s.pub.public_vp[0] == 2);
}

TEST_CASE("awards: a tie of knights does not move Largest Army", "[awards]") {
    GameState s = blank();
    preplay(s, 1, 3);
    set_army(s, 1);
    preplay(s, 0, 2);
    give_dev(s, 0, DevCard::KNIGHT);
    const EventList ev = play_knight(s);
    CHECK(s.pub.knights_played[0] == 3);
    CHECK(events_of(ev, EventType::LARGEST_ARMY).empty());
    CHECK(s.pub.largest_army_holder == 1);
    CHECK(s.pub.public_vp[0] == 0);
    CHECK(s.pub.public_vp[1] == 2);
}

TEST_CASE("awards: more knights than the holder takes Largest Army", "[awards]") {
    GameState s = blank();
    preplay(s, 1, 3);
    set_army(s, 1);
    preplay(s, 0, 3);
    give_dev(s, 0, DevCard::KNIGHT);
    const EventList ev = play_knight(s);
    const auto la = events_of(ev, EventType::LARGEST_ARMY);
    REQUIRE(la.size() == 1);
    CHECK(la[0].player == 0);
    CHECK(s.pub.largest_army_holder == 0);
    CHECK(s.pub.public_vp[0] == 2);
    CHECK(s.pub.public_vp[1] == 0);
}

TEST_CASE("awards: the holder's further knights change nothing", "[awards]") {
    GameState s = blank();
    preplay(s, 0, 3);
    set_army(s, 0);
    give_dev(s, 0, DevCard::KNIGHT);
    const EventList ev = play_knight(s);
    CHECK(s.pub.knights_played[0] == 4);
    CHECK(events_of(ev, EventType::LARGEST_ARMY).empty());
    CHECK(s.pub.largest_army_holder == 0);
    CHECK(s.pub.public_vp[0] == 2);
}

TEST_CASE("awards: Largest Army points follow the card across holders", "[awards]") {
    GameState s = blank();
    preplay(s, 0, 2);
    give_dev(s, 0, DevCard::KNIGHT);
    play_knight(s);
    REQUIRE(s.pub.largest_army_holder == 0);

    preplay(s, 1, 3);
    set_main(s, 1);
    give_dev(s, 1, DevCard::KNIGHT);
    play_knight(s);  // 4 > 3
    CHECK(s.pub.largest_army_holder == 1);
    CHECK(s.pub.public_vp[0] == 0);
    CHECK(s.pub.public_vp[1] == 2);

    preplay(s, 2, 4);
    set_main(s, 2);
    give_dev(s, 2, DevCard::KNIGHT);
    const EventList ev = play_knight(s);  // 5 > 4
    CHECK(events_of(ev, EventType::LARGEST_ARMY).size() == 1);
    CHECK(s.pub.largest_army_holder == 2);
    CHECK(s.pub.public_vp[1] == 0);
    CHECK(s.pub.public_vp[2] == 2);
}

TEST_CASE("awards: taking Largest Army to ten points wins", "[awards]") {
    GameState s = blank();
    give_cities(s, 0, {2, 10, 20, 32});
    preplay(s, 0, 2);
    give_dev(s, 0, DevCard::KNIGHT);
    const EventList ev = step(s, act(ActionType::PLAY_DEV, idx(DevCard::KNIGHT)));
    CHECK(events_of(ev, EventType::LARGEST_ARMY).size() == 1);
    const auto over = events_of(ev, EventType::GAME_OVER);
    REQUIRE(over.size() == 1);
    CHECK(over[0].player == 0);
    CHECK(s.pub.phase == Phase::TERMINAL);
}

// ---------------------------------------------------------------- cross-checks

TEST_CASE("awards: engine road_len matches the exhaustive reference on random networks",
          "[awards]") {
    static const GameState base = blank();
    const Topology& topo = topology();
    int checked = 0;
    for (uint64_t seed = 0; seed < 2000; ++seed) {
        GameState s = base;
        Rnd rnd{seed};
        const auto mover = static_cast<uint8_t>(rnd.below(NUM_PLAYERS));

        // Connected random road networks; the mover keeps one piece for the final build.
        for (uint8_t p = 0; p < NUM_PLAYERS; ++p) {
            const int cap = p == mover ? ROADS_PER_PLAYER - 1 : ROADS_PER_PLAYER;
            const int want = 1 + rnd.below(cap);
            for (int i = 0; i < want; ++i) {
                EdgeSet taken{};
                for (const EdgeSet& r : s.pub.roads) {
                    taken[0] |= r[0];
                    taken[1] |= r[1];
                }
                vector<int> cand;
                for (int e = 0; e < NUM_EDGES; ++e) {
                    if (edge_set_has(taken, e)) continue;
                    const EdgeSet& adj = topo.edge_adjacent_mask[e];
                    const bool touches = ((adj[0] & s.pub.roads[p][0]) |
                                          (adj[1] & s.pub.roads[p][1])) != 0;
                    if (i == 0 || touches) cand.push_back(e);
                }
                if (cand.empty()) break;
                const auto pick = static_cast<size_t>(rnd.below(static_cast<int>(cand.size())));
                place_road(s, p, cand[pick]);
            }
        }
        // Opponent (and own) settlements obeying the distance rule.
        const int n_settle = rnd.below(8);
        for (int i = 0; i < n_settle; ++i) {
            const int v = rnd.below(NUM_VERTICES);
            const auto owner = static_cast<uint8_t>(rnd.below(NUM_PLAYERS));
            if (vertex_occupied(s.pub.vertex[v]) ||
                s.pub.pieces_left[owner][PIECE_SETTLEMENT] == 0) {
                continue;
            }
            bool crowded = false;
            for (uint8_t n : topo.vertex_vertices[v]) {
                if (n != NONE && vertex_occupied(s.pub.vertex[n])) crowded = true;
            }
            if (!crowded) place_settlement(s, owner, v);
        }

        refresh_awards(s);
        set_main(s, mover);
        give(s, mover, Resource::BRICK, 1);
        give(s, mover, Resource::LUMBER, 1);
        const Decision d = next_decision(board(), s);
        if (!type_legal(d.mask, ActionType::BUILD_ROAD)) continue;
        vector<int> options;
        for (int e = 0; e < NUM_EDGES; ++e) {
            if (edge_set_has(d.mask.edge, e)) options.push_back(e);
        }
        const int e = options[static_cast<size_t>(rnd.below(static_cast<int>(options.size())))];
        apply(board(), s, act(ActionType::BUILD_ROAD, e));

        for (uint8_t q = 0; q < NUM_PLAYERS; ++q) {
            INFO("seed " << seed << " player " << int{q});
            CHECK(s.pub.road_len[q] == reference_road_length(s.pub, q));
        }
        ++checked;
    }
    CHECK(checked > 1500);
}

TEST_CASE("awards: road_len and award cards stay exact through random games", "[awards]") {
    int compared = 0;
    for (uint64_t seed = 1; seed <= 20; ++seed) {
        const Board b = random_board(seed);
        GameState s = initial_state(b);
        play_random(b, s, seed, 3000,
                    [&](const GameState& before, const Decision&, const GameState& after,
                        const EventList& ev) {
            bool built = false;
            for (const Event& e : ev) {
                if (e.type == EventType::ROAD_BUILT || e.type == EventType::SETTLEMENT_BUILT) {
                    built = true;
                }
            }
            if (built) {
                ++compared;
                for (uint8_t q = 0; q < NUM_PLAYERS; ++q) {
                    INFO("seed " << seed << " ply " << after.pub.ply);
                    CHECK(after.pub.road_len[q] == reference_road_length(after.pub, q));
                }
            }
            // Every change of holder is announced exactly once, naming the new holder.
            const auto lr = events_of(ev, EventType::LONGEST_ROAD);
            if (after.pub.longest_road_holder != before.pub.longest_road_holder) {
                REQUIRE(lr.size() == 1);
                CHECK(lr[0].player == after.pub.longest_road_holder);
            } else {
                CHECK(lr.empty());
            }
            const auto la = events_of(ev, EventType::LARGEST_ARMY);
            if (after.pub.largest_army_holder != before.pub.largest_army_holder) {
                REQUIRE(la.size() == 1);
                CHECK(la[0].player == after.pub.largest_army_holder);
            } else {
                CHECK(la.empty());
            }
        });
    }
    CHECK(compared > 200);
}
