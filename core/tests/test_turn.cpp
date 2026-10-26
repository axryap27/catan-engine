#include <catch2/catch_test_macros.hpp>

#include <array>
#include <bit>
#include <cstdint>
#include <vector>

#include "catan/action.h"
#include "catan/board.h"
#include "catan/decision.h"
#include "catan/event.h"
#include "catan/invariants.h"
#include "catan/mask.h"
#include "catan/rules.h"
#include "catan/state.h"
#include "catan/topology.h"
#include "catan/types.h"
#include "support/fixtures.h"

using namespace std;
using namespace catan;
using namespace catan::test;

namespace {

constexpr uint8_t BRICK = 0, LUMBER = 1, WOOL = 2, GRAIN = 3, ORE = 4;

const Board& board() {
    static const Board b = standard_board();
    return b;
}

// Empty board in MAIN for player p (dice 8), empty hands, full bank, full deck.
GameState blank(uint8_t p = 0) {
    GameState s = random_setup(board(), 11);
    clear_board(s);
    clear_hands(s);
    set_main(s, p);
    return s;
}

// Same, but waiting for the roll.
GameState blank_roll(uint8_t p = 0) {
    GameState s = blank(p);
    s.pub.phase = Phase::ROLL;
    s.pub.dice = 0;
    return s;
}

void ok(const GameState& s) { CHECK(invariant_violation(board(), s) == nullptr); }

uint8_t hv(int hex, int corner) { return topology().hex_vertices[hex][corner]; }

bool can(const Decision& d, ActionType t, int arg = 0) {
    return is_legal(d.mask, d.phase, Action{t, static_cast<uint8_t>(arg)});
}

int count_edges(const EdgeSet& es) { return popcount(es[0]) + popcount(es[1]); }

// Rolls `sum` from a ROLL state: ROLL action, then the dice chance.
EventList roll(GameState& s, int sum) {
    apply(board(), s, Action{ActionType::ROLL, 0});
    return apply_chance(board(), s, static_cast<uint8_t>(sum - 2));
}

// Gains of player p for resource r in `ev` (summed).
int gained(const EventList& ev, uint8_t p, uint8_t r) {
    int n = 0;
    for (const Event& e : events_of(ev, EventType::GAIN)) {
        if (e.player == p && e.a == r) n += e.b;
    }
    return n;
}

// n pairwise non-adjacent vertices.
vector<uint8_t> spread(int n) {
    vector<uint8_t> out;
    for (int v = 0; v < NUM_VERTICES && static_cast<int>(out.size()) < n; ++v) {
        bool clash = false;
        for (uint8_t u : out) clash |= (topology().vertex_neighbor_mask[v] >> u) & 1;
        if (!clash) out.push_back(static_cast<uint8_t>(v));
    }
    return out;
}

uint8_t port_vertex(PortType t) {
    for (int v = 0; v < NUM_VERTICES; ++v) {
        if (board().vertex_port[v] == static_cast<uint8_t>(t)) return static_cast<uint8_t>(v);
    }
    return NONE;
}

// Hex numbering used below (standard board): 0 ore/10, 1 wool/2, 4 brick/6, 5 wool/4,
// 6 brick/10, 9 desert, 11 ore/8, 12 lumber/8, 17 grain/6.

}  // namespace

TEST_CASE("roll decision offers only ROLL", "[turn]") {
    GameState s = blank_roll();
    const Decision d = next_decision(board(), s);
    REQUIRE(d.kind == DecisionKind::PLAYER);
    CHECK(d.player == 0);
    CHECK(d.phase == Phase::ROLL);
    CHECK(d.n_legal == 1);
    CHECK(d.mask.type == (1u << idx(ActionType::ROLL)));
}

TEST_CASE("roll decision also offers a playable dev card", "[turn]") {
    GameState s = blank_roll();
    give_dev(s, 0, DevCard::KNIGHT);
    ok(s);
    const Decision d = next_decision(board(), s);
    CHECK(d.n_legal == 2);
    CHECK(can(d, ActionType::ROLL));
    CHECK(can(d, ActionType::PLAY_DEV, idx(DevCard::KNIGHT)));
    CHECK(d.mask.devcard == (1u << idx(DevCard::KNIGHT)));
}

TEST_CASE("a victory point card is not playable before the roll", "[turn]") {
    GameState s = blank_roll();
    give_dev(s, 0, DevCard::VICTORY_POINT);
    const Decision d = next_decision(board(), s);
    CHECK(d.n_legal == 1);
    CHECK_FALSE(type_legal(d.mask, ActionType::PLAY_DEV));
}

TEST_CASE("ROLL sets a pending dice chance with 1-6-1 weights", "[turn]") {
    GameState s = blank_roll();
    apply(board(), s, Action{ActionType::ROLL, 0});
    CHECK(s.pub.pending_chance == static_cast<uint8_t>(ChanceKind::DICE));
    const Decision d = next_decision(board(), s);
    REQUIRE(d.kind == DecisionKind::CHANCE);
    CHECK(d.chance.kind == ChanceKind::DICE);
    CHECK(d.chance.n_outcomes == 11);
    CHECK(d.chance.total == 36);
    const array<uint8_t, 11> expected = {1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    for (int i = 0; i < 11; ++i) CHECK(d.chance.weight[i] == expected[i]);
}

TEST_CASE("the dice outcome sets dice and emits DICE_ROLLED", "[turn]") {
    GameState s = blank_roll();
    const EventList ev = roll(s, 9);
    CHECK(s.pub.dice == 9);
    CHECK(s.pub.pending_chance == 0);
    const auto rolled = events_of(ev, EventType::DICE_ROLLED);
    REQUIRE(rolled.size() == 1);
    CHECK(rolled[0].player == 0);
    CHECK(rolled[0].a == 9);
}

TEST_CASE("a non-7 roll moves to MAIN", "[turn]") {
    for (int sum : {2, 3, 6, 8, 12}) {
        GameState s = blank_roll();
        roll(s, sum);
        CHECK(s.pub.phase == Phase::MAIN);
        ok(s);
    }
}

TEST_CASE("a settlement gets 1 card and a city 2 on a matching token", "[turn]") {
    // 10 is on hex 0 (ore) and hex 6 (brick).
    GameState s = blank_roll();
    place_settlement(s, 0, hv(0, 0));
    place_settlement(s, 1, hv(6, 3));
    place_city(s, 1, hv(6, 3));
    ok(s);
    const EventList ev = roll(s, 10);
    CHECK(s.hid.hands[0][ORE] == 1);
    CHECK(s.hid.hands[1][BRICK] == 2);
    CHECK(s.pub.hand_size[0] == 1);
    CHECK(s.pub.hand_size[1] == 2);
    CHECK(s.pub.bank[ORE] == 18);
    CHECK(s.pub.bank[BRICK] == 17);
    CHECK(events_of(ev, EventType::GAIN).size() == 2);
    CHECK(gained(ev, 0, ORE) == 1);
    CHECK(gained(ev, 1, BRICK) == 2);
    ok(s);
}

TEST_CASE("two hexes sharing a number both produce for one player", "[turn]") {
    GameState s = blank_roll();
    place_settlement(s, 2, hv(0, 0));  // ore, token 10
    place_settlement(s, 2, hv(6, 3));  // brick, token 10
    const EventList ev = roll(s, 10);
    CHECK(s.hid.hands[2][ORE] == 1);
    CHECK(s.hid.hands[2][BRICK] == 1);
    CHECK(events_of(ev, EventType::GAIN).size() == 2);
}

TEST_CASE("a settlement on several hexes collects from each matching one", "[turn]") {
    // Hex 4 (brick 6) and hex 5 (wool 4) both touch the top corner of the desert.
    const uint8_t v = hv(9, 0);
    const auto& hexes = topology().vertex_hexes[v];
    REQUIRE(((hexes[0] == 4 || hexes[1] == 4 || hexes[2] == 4) &&
             (hexes[0] == 5 || hexes[1] == 5 || hexes[2] == 5)));
    GameState s = blank_roll();
    place_settlement(s, 0, v);
    GameState a = s, b = s, c = s;
    roll(a, 6);
    CHECK(a.hid.hands[0][BRICK] == 1);
    CHECK(a.pub.hand_size[0] == 1);
    roll(b, 4);
    CHECK(b.hid.hands[0][WOOL] == 1);
    CHECK(b.pub.hand_size[0] == 1);
    roll(c, 5);  // hexes 15 and 16 only; neither touches this vertex
    CHECK(c.pub.hand_size[0] == 0);
}

TEST_CASE("the robber hex produces nothing", "[turn]") {
    GameState s = blank_roll();
    place_settlement(s, 0, hv(0, 0));  // ore 10
    place_settlement(s, 1, hv(6, 3));  // brick 10
    s.pub.robber_hex = 0;
    const EventList ev = roll(s, 10);
    CHECK(s.hid.hands[0][ORE] == 0);
    CHECK(s.hid.hands[1][BRICK] == 1);
    CHECK(events_of(ev, EventType::GAIN).size() == 1);
}

TEST_CASE("the desert never produces", "[turn]") {
    // A vertex whose three hexes include the desert pays only for the other two.
    const uint8_t v = hv(9, 0);
    GameState s = blank_roll();
    place_settlement(s, 0, v);
    for (int sum = 2; sum <= 12; ++sum) {
        if (sum == 7) continue;
        GameState t = s;
        roll(t, sum);
        const int expected = (sum == 6 || sum == 4) ? 1 : 0;
        CHECK(t.pub.hand_size[0] == expected);
    }
}

TEST_CASE("a short bank pays nobody when two players are owed", "[turn]") {
    GameState s = blank_roll();
    give(s, 3, Resource::ORE, 18);  // bank ore = 1
    place_settlement(s, 0, hv(0, 0));  // ore 10
    place_settlement(s, 1, hv(0, 2));  // ore 10
    place_settlement(s, 2, hv(6, 3));  // brick 10, unaffected
    ok(s);
    const EventList ev = roll(s, 10);
    CHECK(s.hid.hands[0][ORE] == 0);
    CHECK(s.hid.hands[1][ORE] == 0);
    CHECK(s.pub.bank[ORE] == 1);
    CHECK(s.hid.hands[2][BRICK] == 1);
    CHECK(gained(ev, 0, ORE) + gained(ev, 1, ORE) == 0);
    ok(s);
}

TEST_CASE("a short bank gives a sole claimant what is left", "[turn]") {
    GameState s = blank_roll();
    give(s, 3, Resource::ORE, 18);  // bank ore = 1
    place_settlement(s, 0, hv(0, 0));
    place_city(s, 0, hv(0, 0));  // owed 2, bank has 1
    const EventList ev = roll(s, 10);
    CHECK(s.hid.hands[0][ORE] == 1);
    CHECK(s.pub.bank[ORE] == 0);
    CHECK(gained(ev, 0, ORE) == 1);
    ok(s);
}

TEST_CASE("a bank that exactly covers demand pays everyone", "[turn]") {
    GameState s = blank_roll();
    give(s, 3, Resource::ORE, 17);  // bank ore = 2
    place_settlement(s, 0, hv(0, 0));
    place_settlement(s, 1, hv(0, 2));
    roll(s, 10);
    CHECK(s.hid.hands[0][ORE] == 1);
    CHECK(s.hid.hands[1][ORE] == 1);
    CHECK(s.pub.bank[ORE] == 0);
}

TEST_CASE("MAIN with nothing affordable offers only END_TURN", "[turn]") {
    GameState s = blank();
    place_settlement(s, 0, hv(9, 0));
    const Decision d = next_decision(board(), s);
    REQUIRE(d.kind == DecisionKind::PLAYER);
    CHECK(d.phase == Phase::MAIN);
    CHECK(d.n_legal == 1);
    CHECK(d.mask.type == (1u << idx(ActionType::END_TURN)));
}

TEST_CASE("road legality depends on affordability", "[turn]") {
    GameState s = blank();
    place_settlement(s, 0, hv(9, 0));
    give(s, 0, Resource::BRICK, 1);
    CHECK_FALSE(type_legal(next_decision(board(), s).mask, ActionType::BUILD_ROAD));
    give(s, 0, Resource::LUMBER, 1);
    CHECK(type_legal(next_decision(board(), s).mask, ActionType::BUILD_ROAD));
}

TEST_CASE("a road needs an unused road piece", "[turn]") {
    GameState s = blank();
    place_settlement(s, 0, hv(9, 0));
    give(s, 0, Resource::BRICK, 1);
    give(s, 0, Resource::LUMBER, 1);
    s.pub.pieces_left[0][PIECE_ROAD] = 0;  // direct edit; deliberately not invariant-checked
    CHECK_FALSE(type_legal(next_decision(board(), s).mask, ActionType::BUILD_ROAD));
}

TEST_CASE("roads connect to own buildings", "[turn]") {
    GameState s = blank();
    const uint8_t v = hv(9, 0);
    place_settlement(s, 0, v);
    give(s, 0, Resource::BRICK, 1);
    give(s, 0, Resource::LUMBER, 1);
    const Decision d = next_decision(board(), s);
    // Exactly the edges at the settlement.
    CHECK(d.mask.edge == topology().vertex_edge_mask[v]);
    CHECK(count_edges(d.mask.edge) == 3);
}

TEST_CASE("roads extend an own road through an empty vertex", "[turn]") {
    GameState s = blank();
    const uint8_t v0 = hv(9, 5), v1 = hv(9, 0);
    place_settlement(s, 0, v0);
    place_road(s, 0, edge_between(v0, v1));
    give(s, 0, Resource::BRICK, 1);
    give(s, 0, Resource::LUMBER, 1);
    const Decision d = next_decision(board(), s);
    const uint8_t built = edge_between(v0, v1);
    EdgeSet expected = topology().vertex_edge_mask[v0];
    expected[0] |= topology().vertex_edge_mask[v1][0];
    expected[1] |= topology().vertex_edge_mask[v1][1];
    expected[built / 64] &= ~(uint64_t{1} << (built % 64));
    CHECK(d.mask.edge == expected);
    CHECK_FALSE(can(d, ActionType::BUILD_ROAD, built));
}

TEST_CASE("an opponent building blocks road extension through its vertex", "[turn]") {
    GameState s = blank();
    const uint8_t a = hv(9, 5), b = hv(9, 0), c = hv(9, 1);
    place_settlement(s, 0, a);
    place_road(s, 0, edge_between(a, b));
    place_road(s, 0, edge_between(b, c));
    give(s, 0, Resource::BRICK, 1);
    give(s, 0, Resource::LUMBER, 1);

    // Free vertex c: the road may continue past it.
    Decision d = next_decision(board(), s);
    for (uint8_t e : topology().vertex_edges[c]) {
        if (e == NONE || e == edge_between(b, c)) continue;
        CHECK(can(d, ActionType::BUILD_ROAD, e));
    }

    // Opponent settlement on c: those edges are cut off.
    place_settlement(s, 1, c);
    refresh_awards(s);
    ok(s);
    d = next_decision(board(), s);
    for (uint8_t e : topology().vertex_edges[c]) {
        if (e == NONE || e == edge_between(b, c)) continue;
        CHECK_FALSE(can(d, ActionType::BUILD_ROAD, e));
    }
    // The rest of the network is still buildable.
    for (uint8_t e : topology().vertex_edges[a]) {
        if (e != edge_between(a, b)) CHECK(can(d, ActionType::BUILD_ROAD, e));
    }
}

TEST_CASE("roads cannot be built on taken edges", "[turn]") {
    GameState s = blank();
    const uint8_t v = hv(9, 0);
    place_settlement(s, 0, v);
    const uint8_t mine = edge_between(v, hv(9, 1));
    const uint8_t theirs = edge_between(v, hv(9, 5));
    place_road(s, 0, mine);
    place_road(s, 1, theirs);
    give(s, 0, Resource::BRICK, 1);
    give(s, 0, Resource::LUMBER, 1);
    const Decision d = next_decision(board(), s);
    CHECK_FALSE(can(d, ActionType::BUILD_ROAD, mine));
    CHECK_FALSE(can(d, ActionType::BUILD_ROAD, theirs));
}

TEST_CASE("settlements need resources, a connecting road and a free vertex", "[turn]") {
    GameState s = blank();
    const uint8_t v0 = hv(9, 0), v1 = hv(9, 1), v2 = hv(9, 2);
    place_settlement(s, 0, v0);
    place_road(s, 0, edge_between(v0, v1));
    place_road(s, 0, edge_between(v1, v2));
    for (uint8_t r : {BRICK, LUMBER, WOOL, GRAIN}) give(s, 0, static_cast<Resource>(r), 1);
    refresh_awards(s);
    ok(s);
    const Decision d = next_decision(board(), s);
    // v0 is occupied, v1 is next to it, only v2 satisfies the distance rule.
    CHECK(d.mask.settlement == vertex_bit(v2));
    CHECK(can(d, ActionType::BUILD_SETTLEMENT, v2));
}

TEST_CASE("the distance rule blocks a settlement next to an opponent", "[turn]") {
    GameState s = blank();
    const uint8_t v0 = hv(9, 0), v1 = hv(9, 1), v2 = hv(9, 2);
    place_settlement(s, 0, v0);
    place_road(s, 0, edge_between(v0, v1));
    place_road(s, 0, edge_between(v1, v2));
    // v2's third neighbor, away from the hex.
    uint8_t outside = NONE;
    for (uint8_t n : topology().vertex_vertices[v2]) {
        if (n != NONE && n != v1 && n != hv(9, 3)) outside = n;
    }
    REQUIRE(outside != NONE);
    place_settlement(s, 1, outside);
    for (uint8_t r : {BRICK, LUMBER, WOOL, GRAIN}) give(s, 0, static_cast<Resource>(r), 1);
    refresh_awards(s);
    ok(s);
    const Decision d = next_decision(board(), s);
    CHECK(d.mask.settlement == 0);
    CHECK_FALSE(type_legal(d.mask, ActionType::BUILD_SETTLEMENT));
}

TEST_CASE("a settlement needs one of the player's own roads at the vertex", "[turn]") {
    GameState s = blank();
    place_settlement(s, 0, hv(9, 0));
    for (uint8_t r : {BRICK, LUMBER, WOOL, GRAIN}) give(s, 0, static_cast<Resource>(r), 1);
    // No road yet: nothing is connected.
    CHECK_FALSE(type_legal(next_decision(board(), s).mask, ActionType::BUILD_SETTLEMENT));
    // An opponent's road does not count.
    const uint8_t v1 = hv(9, 1), v2 = hv(9, 2);
    place_road(s, 1, edge_between(v1, v2));
    CHECK_FALSE(type_legal(next_decision(board(), s).mask, ActionType::BUILD_SETTLEMENT));
}

TEST_CASE("a settlement needs the full cost and a piece", "[turn]") {
    GameState s = blank();
    const uint8_t v0 = hv(9, 0), v1 = hv(9, 1), v2 = hv(9, 2);
    place_settlement(s, 0, v0);
    place_road(s, 0, edge_between(v0, v1));
    place_road(s, 0, edge_between(v1, v2));
    for (uint8_t r : {BRICK, LUMBER, WOOL}) give(s, 0, static_cast<Resource>(r), 1);
    CHECK_FALSE(type_legal(next_decision(board(), s).mask, ActionType::BUILD_SETTLEMENT));
    give(s, 0, Resource::GRAIN, 1);
    CHECK(type_legal(next_decision(board(), s).mask, ActionType::BUILD_SETTLEMENT));
    s.pub.pieces_left[0][PIECE_SETTLEMENT] = 0;  // direct edit; not invariant-checked
    CHECK_FALSE(type_legal(next_decision(board(), s).mask, ActionType::BUILD_SETTLEMENT));
}

TEST_CASE("cities go only on the player's own settlements", "[turn]") {
    GameState s = blank();
    const auto v = spread(4);
    place_settlement(s, 0, v[0]);
    place_settlement(s, 0, v[1]);
    place_settlement(s, 1, v[2]);  // opponent's
    place_settlement(s, 0, v[3]);
    place_city(s, 0, v[3]);        // already a city
    give(s, 0, Resource::GRAIN, 2);
    give(s, 0, Resource::ORE, 3);
    ok(s);
    const Decision d = next_decision(board(), s);
    CHECK(d.mask.city == (vertex_bit(v[0]) | vertex_bit(v[1])));
    CHECK(type_legal(d.mask, ActionType::BUILD_CITY));
}

TEST_CASE("a city needs 2 grain and 3 ore", "[turn]") {
    GameState s = blank();
    place_settlement(s, 0, hv(9, 0));
    give(s, 0, Resource::GRAIN, 2);
    give(s, 0, Resource::ORE, 2);
    CHECK_FALSE(type_legal(next_decision(board(), s).mask, ActionType::BUILD_CITY));
    give(s, 0, Resource::ORE, 1);
    CHECK(type_legal(next_decision(board(), s).mask, ActionType::BUILD_CITY));
    s.pub.pieces_left[0][PIECE_CITY] = 0;  // direct edit; not invariant-checked
    CHECK_FALSE(type_legal(next_decision(board(), s).mask, ActionType::BUILD_CITY));
}

TEST_CASE("buying a development card needs wool, grain, ore and a non-empty deck", "[turn]") {
    GameState s = blank();
    give(s, 0, Resource::WOOL, 1);
    give(s, 0, Resource::GRAIN, 1);
    CHECK_FALSE(type_legal(next_decision(board(), s).mask, ActionType::BUY_DEV));
    give(s, 0, Resource::ORE, 1);
    CHECK(type_legal(next_decision(board(), s).mask, ActionType::BUY_DEV));

    // Empty the deck into player 3's hand.
    for (int c = 0; c < NUM_DEV; ++c) give_dev(s, 3, static_cast<DevCard>(c), s.hid.dev_deck[c]);
    REQUIRE(s.pub.dev_deck_remaining == 0);
    ok(s);
    CHECK_FALSE(type_legal(next_decision(board(), s).mask, ActionType::BUY_DEV));
}

TEST_CASE("END_TURN is always legal in MAIN", "[turn]") {
    for (uint64_t seed = 1; seed <= 5; ++seed) {
        GameState s = random_setup(board(), seed);
        set_main(s, static_cast<uint8_t>(seed % NUM_PLAYERS));
        CHECK(type_legal(next_decision(board(), s).mask, ActionType::END_TURN));
    }
}

TEST_CASE("building a road pays the bank and uses a piece", "[turn]") {
    GameState s = blank();
    const uint8_t v = hv(9, 0);
    place_settlement(s, 0, v);
    give(s, 0, Resource::BRICK, 2);
    give(s, 0, Resource::LUMBER, 1);
    const uint8_t e = edge_between(v, hv(9, 1));
    const EventList ev = apply(board(), s, Action{ActionType::BUILD_ROAD, e});
    CHECK(edge_set_has(s.pub.roads[0], e));
    CHECK(s.pub.pieces_left[0][PIECE_ROAD] == 14);
    CHECK(s.hid.hands[0][BRICK] == 1);
    CHECK(s.hid.hands[0][LUMBER] == 0);
    CHECK(s.pub.bank[BRICK] == 18);
    CHECK(s.pub.bank[LUMBER] == 19);
    CHECK(s.pub.hand_size[0] == 1);
    CHECK(s.pub.public_vp[0] == 1);
    const auto spent = events_of(ev, EventType::SPEND);
    REQUIRE(spent.size() == 2);
    CHECK(spent[0].a == BRICK);
    CHECK(spent[0].b == 1);
    CHECK(spent[1].a == LUMBER);
    CHECK(spent[1].b == 1);
    const auto built = events_of(ev, EventType::ROAD_BUILT);
    REQUIRE(built.size() == 1);
    CHECK(built[0].player == 0);
    CHECK(built[0].a == e);
    ok(s);
}

TEST_CASE("building a settlement scores a point and costs four cards", "[turn]") {
    GameState s = blank();
    const uint8_t v0 = hv(9, 0), v1 = hv(9, 1), v2 = hv(9, 2);
    place_settlement(s, 0, v0);
    place_road(s, 0, edge_between(v0, v1));
    place_road(s, 0, edge_between(v1, v2));
    for (uint8_t r : {BRICK, LUMBER, WOOL, GRAIN}) give(s, 0, static_cast<Resource>(r), 1);
    const EventList ev = apply(board(), s, Action{ActionType::BUILD_SETTLEMENT, v2});
    CHECK(vertex_owner(s.pub.vertex[v2]) == 0);
    CHECK_FALSE(vertex_is_city(s.pub.vertex[v2]));
    CHECK(s.pub.public_vp[0] == 2);
    CHECK(s.pub.pieces_left[0][PIECE_SETTLEMENT] == 3);
    CHECK(s.pub.hand_size[0] == 0);
    for (int r = 0; r < 4; ++r) CHECK(s.pub.bank[r] == 19);
    CHECK(events_of(ev, EventType::SPEND).size() == 4);
    CHECK(events_of(ev, EventType::SETTLEMENT_BUILT).size() == 1);
    ok(s);
}

TEST_CASE("building a city adds a point and returns the settlement piece", "[turn]") {
    GameState s = blank();
    const uint8_t v = hv(9, 0);
    place_settlement(s, 0, v);
    give(s, 0, Resource::GRAIN, 3);
    give(s, 0, Resource::ORE, 3);
    const EventList ev = apply(board(), s, Action{ActionType::BUILD_CITY, v});
    CHECK(vertex_is_city(s.pub.vertex[v]));
    CHECK(vertex_owner(s.pub.vertex[v]) == 0);
    CHECK(s.pub.public_vp[0] == 2);
    CHECK(s.pub.pieces_left[0][PIECE_SETTLEMENT] == 5);
    CHECK(s.pub.pieces_left[0][PIECE_CITY] == 3);
    CHECK(s.hid.hands[0][GRAIN] == 1);
    CHECK(s.hid.hands[0][ORE] == 0);
    CHECK(s.pub.bank[GRAIN] == 18);
    CHECK(s.pub.bank[ORE] == 19);
    const auto spent = events_of(ev, EventType::SPEND);
    REQUIRE(spent.size() == 2);
    CHECK(spent[0].a == GRAIN);
    CHECK(spent[0].b == 2);
    CHECK(spent[1].a == ORE);
    CHECK(spent[1].b == 3);
    CHECK(events_of(ev, EventType::CITY_BUILT).size() == 1);
    ok(s);
}

TEST_CASE("bank exchange is 4 to 1 without a harbor", "[turn]") {
    GameState s = blank();
    give(s, 0, Resource::LUMBER, 3);
    CHECK_FALSE(type_legal(next_decision(board(), s).mask, ActionType::BANK_EXCHANGE));
    give(s, 0, Resource::LUMBER, 1);
    const Decision d = next_decision(board(), s);
    CHECK(d.mask.resource == (1u << LUMBER));
    CHECK(can(d, ActionType::BANK_EXCHANGE, LUMBER));
    CHECK_FALSE(can(d, ActionType::BANK_EXCHANGE, BRICK));
}

TEST_CASE("a generic harbor makes every exchange 3 to 1", "[turn]") {
    const uint8_t port = port_vertex(PortType::GENERIC);
    REQUIRE(port != NONE);
    GameState s = blank();
    place_settlement(s, 0, port);
    give(s, 0, Resource::LUMBER, 3);
    give(s, 0, Resource::ORE, 2);
    ok(s);
    const Decision d = next_decision(board(), s);
    CHECK(d.mask.resource == (1u << LUMBER));
}

TEST_CASE("a matching harbor makes that resource 2 to 1 only", "[turn]") {
    const uint8_t port = port_vertex(PortType::GRAIN);
    REQUIRE(port != NONE);
    GameState s = blank();
    place_settlement(s, 0, port);
    give(s, 0, Resource::GRAIN, 2);
    give(s, 0, Resource::LUMBER, 3);   // not enough: other resources stay 4:1
    give(s, 0, Resource::BRICK, 4);
    ok(s);
    const Decision d = next_decision(board(), s);
    CHECK(d.mask.resource == ((1u << GRAIN) | (1u << BRICK)));
}

TEST_CASE("a harbor on an opponent's settlement gives nothing", "[turn]") {
    GameState s = blank();
    place_settlement(s, 1, port_vertex(PortType::GENERIC));
    give(s, 0, Resource::LUMBER, 3);
    CHECK_FALSE(type_legal(next_decision(board(), s).mask, ActionType::BANK_EXCHANGE));
}

TEST_CASE("a 2 to 1 harbor beats a generic one for its resource", "[turn]") {
    GameState s = blank();
    place_settlement(s, 0, port_vertex(PortType::GENERIC));
    place_settlement(s, 0, port_vertex(PortType::GRAIN));
    give(s, 0, Resource::GRAIN, 2);
    give(s, 0, Resource::ORE, 3);
    ok(s);
    const Decision d = next_decision(board(), s);
    CHECK(d.mask.resource == ((1u << GRAIN) | (1u << ORE)));
}

TEST_CASE("bank exchange is two steps and moves cards on the second", "[turn]") {
    GameState s = blank();
    give(s, 0, Resource::LUMBER, 5);
    apply(board(), s, Action{ActionType::BANK_EXCHANGE, LUMBER});
    CHECK(s.pub.phase == Phase::EXCHANGE_RECV);
    CHECK(s.pub.exchange_give == LUMBER);
    CHECK(s.hid.hands[0][LUMBER] == 5);  // nothing has moved yet

    const Decision d = next_decision(board(), s);
    REQUIRE(d.kind == DecisionKind::PLAYER);
    CHECK(d.player == 0);
    CHECK(d.phase == Phase::EXCHANGE_RECV);
    CHECK(d.mask.resource == (0x1Fu & ~(1u << LUMBER)));
    CHECK(d.n_legal == 4);

    const EventList ev = apply(board(), s, Action{ActionType::BANK_EXCHANGE, WOOL});
    CHECK(s.pub.phase == Phase::MAIN);
    CHECK(s.hid.hands[0][LUMBER] == 1);
    CHECK(s.hid.hands[0][WOOL] == 1);
    CHECK(s.pub.hand_size[0] == 2);
    CHECK(s.pub.bank[LUMBER] == 19 - 5 + 4);
    CHECK(s.pub.bank[WOOL] == 18);
    const auto spent = events_of(ev, EventType::SPEND);
    REQUIRE(spent.size() == 1);
    CHECK(spent[0].a == LUMBER);
    CHECK(spent[0].b == 4);
    const auto gain = events_of(ev, EventType::GAIN);
    REQUIRE(gain.size() == 1);
    CHECK(gain[0].a == WOOL);
    CHECK(gain[0].b == 1);
    ok(s);
}

TEST_CASE("exchange at a 2 to 1 harbor moves two cards", "[turn]") {
    GameState s = blank();
    place_settlement(s, 0, port_vertex(PortType::GRAIN));
    give(s, 0, Resource::GRAIN, 3);
    apply(board(), s, Action{ActionType::BANK_EXCHANGE, GRAIN});
    const EventList ev = apply(board(), s, Action{ActionType::BANK_EXCHANGE, ORE});
    CHECK(s.hid.hands[0][GRAIN] == 1);
    CHECK(s.hid.hands[0][ORE] == 1);
    CHECK(s.pub.bank[GRAIN] == 18);
    CHECK(events_of(ev, EventType::SPEND)[0].b == 2);
    ok(s);
}

TEST_CASE("exchange at a generic harbor moves three cards", "[turn]") {
    GameState s = blank();
    place_settlement(s, 0, port_vertex(PortType::GENERIC));
    give(s, 0, Resource::BRICK, 3);
    apply(board(), s, Action{ActionType::BANK_EXCHANGE, BRICK});
    apply(board(), s, Action{ActionType::BANK_EXCHANGE, LUMBER});
    CHECK(s.hid.hands[0][BRICK] == 0);
    CHECK(s.hid.hands[0][LUMBER] == 1);
    CHECK(s.pub.bank[BRICK] == 19);
    ok(s);
}

TEST_CASE("an exchange cannot give when the bank has nothing else", "[turn]") {
    GameState s = blank();
    for (uint8_t r : {LUMBER, WOOL, GRAIN, ORE}) give(s, 3, static_cast<Resource>(r), 19);
    give(s, 0, Resource::BRICK, 4);
    ok(s);
    CHECK_FALSE(type_legal(next_decision(board(), s).mask, ActionType::BANK_EXCHANGE));
    // Once the bank holds a single other resource the exchange is back.
    give(s, 3, Resource::ORE, -1);
    const Decision d = next_decision(board(), s);
    CHECK(d.mask.resource == (1u << BRICK));
    apply(board(), s, Action{ActionType::BANK_EXCHANGE, BRICK});
    CHECK(next_decision(board(), s).mask.resource == (1u << ORE));
}

TEST_CASE("an exchange cannot receive a resource the bank lacks", "[turn]") {
    GameState s = blank();
    give(s, 3, Resource::WOOL, 19);
    give(s, 0, Resource::LUMBER, 4);
    apply(board(), s, Action{ActionType::BANK_EXCHANGE, LUMBER});
    const Decision d = next_decision(board(), s);
    CHECK(d.mask.resource == ((1u << BRICK) | (1u << GRAIN) | (1u << ORE)));
    CHECK_FALSE(can(d, ActionType::BANK_EXCHANGE, WOOL));
    CHECK_FALSE(can(d, ActionType::BANK_EXCHANGE, LUMBER));
}

TEST_CASE("END_TURN passes the turn on", "[turn]") {
    GameState s = blank(1);
    s.pub.turn = 5;
    s.pub.played_dev_this_turn = 1;
    const EventList ev = apply(board(), s, Action{ActionType::END_TURN, 0});
    CHECK(s.pub.current_player == 2);
    CHECK(s.pub.phase == Phase::ROLL);
    CHECK(s.pub.dice == 0);
    CHECK(s.pub.turn == 6);
    CHECK(s.pub.played_dev_this_turn == 0);
    const auto ended = events_of(ev, EventType::TURN_ENDED);
    REQUIRE(ended.size() == 1);
    CHECK(ended[0].player == 1);
    CHECK(events_of(ev, EventType::GAME_OVER).empty());
    ok(s);
}

TEST_CASE("END_TURN wraps from the last seat to the first", "[turn]") {
    GameState s = blank(3);
    apply(board(), s, Action{ActionType::END_TURN, 0});
    CHECK(s.pub.current_player == 0);
    const Decision d = next_decision(board(), s);
    CHECK(d.player == 0);
    CHECK(d.phase == Phase::ROLL);
}

TEST_CASE("END_TURN clears this turn's dev purchases", "[turn]") {
    GameState s = blank(0);
    give_dev(s, 0, DevCard::KNIGHT);
    s.hid.dev_bought_this_turn[idx(DevCard::KNIGHT)] = 1;
    apply(board(), s, Action{ActionType::END_TURN, 0});
    CHECK(s.hid.dev_bought_this_turn[idx(DevCard::KNIGHT)] == 0);
}

TEST_CASE("reaching the turn cap ends the game with no winner", "[turn]") {
    GameState s = blank(0);
    s.pub.turn = TURN_CAP - 1;
    const EventList ev = apply(board(), s, Action{ActionType::END_TURN, 0});
    CHECK(s.pub.phase == Phase::TERMINAL);
    const auto over = events_of(ev, EventType::GAME_OVER);
    REQUIRE(over.size() == 1);
    CHECK(over[0].player == NONE);
    CHECK(next_decision(board(), s).kind == DecisionKind::TERMINAL);
}

TEST_CASE("one turn short of the cap the game goes on", "[turn]") {
    GameState s = blank(0);
    s.pub.turn = TURN_CAP - 2;
    apply(board(), s, Action{ActionType::END_TURN, 0});
    CHECK(s.pub.phase == Phase::ROLL);
}

namespace {

// Player 0 with 3 cities and `settlements` settlements on spread vertices; returns them.
vector<uint8_t> near_win(GameState& s, int settlements) {
    const auto v = spread(3 + settlements);
    for (int i = 0; i < 3; ++i) {
        place_settlement(s, 0, v[i]);
        place_city(s, 0, v[i]);
    }
    for (int i = 3; i < 3 + settlements; ++i) place_settlement(s, 0, v[i]);
    return v;
}

}  // namespace

TEST_CASE("reaching 10 points on your own turn wins at once", "[turn]") {
    GameState s = blank();
    const auto v = near_win(s, 3);  // 6 + 3 = 9 points
    REQUIRE(s.pub.public_vp[0] == 9);
    give(s, 0, Resource::GRAIN, 2);
    give(s, 0, Resource::ORE, 3);
    ok(s);
    const EventList ev = apply(board(), s, Action{ActionType::BUILD_CITY, v[3]});
    CHECK(s.pub.public_vp[0] == 10);
    CHECK(s.pub.phase == Phase::TERMINAL);
    const auto over = events_of(ev, EventType::GAME_OVER);
    REQUIRE(over.size() == 1);
    CHECK(over[0].player == 0);
    CHECK(next_decision(board(), s).kind == DecisionKind::TERMINAL);
}

TEST_CASE("9 points is not a win", "[turn]") {
    GameState s = blank();
    const auto v = near_win(s, 2);  // 6 + 2 = 8 points
    give(s, 0, Resource::GRAIN, 2);
    give(s, 0, Resource::ORE, 3);
    const EventList ev = apply(board(), s, Action{ActionType::BUILD_CITY, v[3]});
    CHECK(s.pub.public_vp[0] == 9);
    CHECK(s.pub.phase == Phase::MAIN);
    CHECK(events_of(ev, EventType::GAME_OVER).empty());
}

TEST_CASE("victory point cards count toward the win", "[turn]") {
    GameState s = blank();
    const auto v = near_win(s, 2);  // 8 public points
    give_dev(s, 0, DevCard::VICTORY_POINT);
    give(s, 0, Resource::GRAIN, 2);
    give(s, 0, Resource::ORE, 3);
    ok(s);
    const EventList ev = apply(board(), s, Action{ActionType::BUILD_CITY, v[3]});
    CHECK(s.pub.public_vp[0] == 9);  // public count excludes the hidden card
    CHECK(s.pub.phase == Phase::TERMINAL);
    const auto over = events_of(ev, EventType::GAME_OVER);
    REQUIRE(over.size() == 1);
    CHECK(over[0].player == 0);
}

TEST_CASE("two victory point cards on 8 points win on the next action", "[turn]") {
    GameState s = blank();
    const auto v = near_win(s, 2);  // 8 public points
    give_dev(s, 0, DevCard::VICTORY_POINT, 2);
    give(s, 0, Resource::BRICK, 1);
    give(s, 0, Resource::LUMBER, 1);
    const uint8_t e = topology().vertex_edges[v[0]][0];
    apply(board(), s, Action{ActionType::BUILD_ROAD, e});  // any action runs the check
    CHECK(s.pub.phase == Phase::TERMINAL);
}

TEST_CASE("a player at 10 points wins when their turn begins", "[turn]") {
    GameState s = blank(0);
    // Player 1 sits on 10 points: 4 cities + 2 settlements.
    const auto v = spread(6);
    for (int i = 0; i < 4; ++i) {
        place_settlement(s, 1, v[i]);
        place_city(s, 1, v[i]);
    }
    place_settlement(s, 1, v[4]);
    place_settlement(s, 1, v[5]);
    REQUIRE(s.pub.public_vp[1] == 10);
    ok(s);

    // Player 0's non-ending moves do not end the game.
    give(s, 0, Resource::LUMBER, 4);
    apply(board(), s, Action{ActionType::BANK_EXCHANGE, LUMBER});
    apply(board(), s, Action{ActionType::BANK_EXCHANGE, WOOL});
    CHECK(s.pub.phase == Phase::MAIN);

    const EventList ev = apply(board(), s, Action{ActionType::END_TURN, 0});
    CHECK(s.pub.phase == Phase::TERMINAL);
    const auto over = events_of(ev, EventType::GAME_OVER);
    REQUIRE(over.size() == 1);
    CHECK(over[0].player == 1);
    CHECK(events_of(ev, EventType::TURN_ENDED).size() == 1);
}

TEST_CASE("every legal MAIN action applies cleanly", "[turn]") {
    for (uint64_t seed = 1; seed <= 6; ++seed) {
        GameState base = random_setup(board(), seed);
        const uint8_t p = static_cast<uint8_t>(seed % NUM_PLAYERS);
        for (uint8_t r = 0; r < NUM_RES; ++r) give(base, p, static_cast<Resource>(r), 4);
        give_dev(base, p, DevCard::KNIGHT);
        set_main(base, p);
        REQUIRE(invariant_violation(board(), base) == nullptr);

        const Decision d = next_decision(board(), base);
        REQUIRE(d.kind == DecisionKind::PLAYER);
        REQUIRE(d.n_legal > 1);
        for (int n = 0; n < d.n_legal; ++n) {
            const Action a = nth_legal(d.mask, d.phase, n);
            REQUIRE(is_legal(d.mask, d.phase, a));
            GameState s = base;
            apply(board(), s, a);
            INFO("seed " << seed << " action type " << idx(a.type) << " arg " << int(a.arg));
            CHECK(invariant_violation(board(), s) == nullptr);
        }
    }
}
