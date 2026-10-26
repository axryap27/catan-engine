#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <initializer_list>

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

// The same position before the roll (dev cards may be played in ROLL).
void to_roll(GameState& s) {
    s.pub.phase = Phase::ROLL;
    s.pub.dice = 0;
}

Action act(ActionType t, int arg = 0) { return Action{t, static_cast<uint8_t>(arg)}; }
Action play(DevCard c) { return act(ActionType::PLAY_DEV, idx(c)); }
Action pick(Resource r) { return act(ActionType::PLAY_DEV, idx(r)); }  // YoP / Monopoly pick

constexpr uint16_t tbit(ActionType t) { return static_cast<uint16_t>(1u << idx(t)); }
constexpr uint8_t dbit(DevCard c) { return static_cast<uint8_t>(1u << idx(c)); }
constexpr uint8_t rbit(Resource r) { return static_cast<uint8_t>(1u << idx(r)); }

Decision decide(const GameState& s) { return next_decision(board(), s); }

void check_invariants(const GameState& s) {
    const char* why = invariant_violation(board(), s);
    CHECK(why == nullptr);
}

// Applies a legal PLAYER action, asserting it is in the mask.
EventList step(GameState& s, Action a) {
    const Decision d = decide(s);
    REQUIRE(d.kind == DecisionKind::PLAYER);
    REQUIRE(is_legal(d.mask, d.phase, a));
    const EventList ev = apply(board(), s, a);
    check_invariants(s);
    return ev;
}

EventList resolve(GameState& s, ChanceKind kind, uint8_t outcome) {
    const Decision d = decide(s);
    REQUIRE(d.kind == DecisionKind::CHANCE);
    REQUIRE(d.chance.kind == kind);
    REQUIRE(d.chance.weight[outcome] > 0);
    const EventList ev = apply_chance(board(), s, outcome);
    check_invariants(s);
    return ev;
}

EventList draw(GameState& s, DevCard c) {
    return resolve(s, ChanceKind::DEV_DRAW, static_cast<uint8_t>(idx(c)));
}

bool legal(const GameState& s, Action a) {
    const Decision d = decide(s);
    return d.kind == DecisionKind::PLAYER && is_legal(d.mask, d.phase, a);
}

void give_buy_cost(GameState& s, uint8_t p) {
    give(s, p, Resource::WOOL, 1);
    give(s, p, Resource::GRAIN, 1);
    give(s, p, Resource::ORE, 1);
}

// Buys one card of type c (resources are added first).
void buy(GameState& s, DevCard c) {
    give_buy_cost(s, s.pub.current_player);
    step(s, act(ActionType::BUY_DEV));
    draw(s, c);
}

// Moves the robber to the lowest-numbered legal hex.
void move_robber_away(GameState& s) {
    const Decision d = decide(s);
    REQUIRE(d.phase == Phase::MOVE_ROBBER);
    step(s, act(ActionType::MOVE_ROBBER, countr_zero(d.mask.tile)));
}

// From MAIN, ends the current player's turn and plays empty turns (roll 8, end) for the others
// until the same player is back in ROLL.
void pass_turns(GameState& s) {
    for (int i = 0; i < NUM_PLAYERS; ++i) {
        step(s, act(ActionType::END_TURN));
        if (i == NUM_PLAYERS - 1) break;
        step(s, act(ActionType::ROLL));
        resolve(s, ChanceKind::DICE, 6);  // outcome i is a roll of i + 2
    }
}

EdgeSet edge_set(initializer_list<int> es) {
    EdgeSet out{};
    for (int e : es) edge_set_add(out, e);
    return out;
}

int edge_count(const EdgeSet& es) { return popcount(es[0]) + popcount(es[1]); }

// n roads for p on the lowest edges that avoid the edges at vertex 18.
void fill_roads(GameState& s, uint8_t p, int n) {
    const auto& at18 = topology().vertex_edges[18];
    for (int e = 0, placed = 0; placed < n && e < NUM_EDGES; ++e) {
        if (find(at18.begin(), at18.end(), e) != at18.end()) continue;
        place_road(s, p, e);
        ++placed;
    }
    refresh_awards(s);
}

void give_cities(GameState& s, uint8_t p, initializer_list<int> vs) {
    for (int v : vs) {
        place_settlement(s, p, v);
        place_city(s, p, v);
    }
}

// Player 0 at vertex 18 with a Road Building card.
GameState rb_state(bool before_roll) {
    GameState s = blank();
    if (before_roll) to_roll(s);
    place_settlement(s, 0, 18);
    give_dev(s, 0, DevCard::ROAD_BUILDING);
    return s;
}

}  // namespace

// ---------------------------------------------------------------- buying

TEST_CASE("devcards: BUY_DEV pays wool, grain and ore and opens a deck-weighted chance",
          "[devcards]") {
    GameState s = blank();
    give_buy_cost(s, 0);
    const auto bank_before = s.pub.bank;
    CHECK(type_legal(decide(s).mask, ActionType::BUY_DEV));

    const EventList ev = step(s, act(ActionType::BUY_DEV));
    const auto spends = events_of(ev, EventType::SPEND);
    REQUIRE(spends.size() == 3);
    for (const Event& e : spends) {
        CHECK(e.player == 0);
        CHECK(e.b == 1);
    }
    CHECK(spends[0].a == idx(Resource::WOOL));
    CHECK(spends[1].a == idx(Resource::GRAIN));
    CHECK(spends[2].a == idx(Resource::ORE));
    CHECK(s.pub.hand_size[0] == 0);
    CHECK(s.pub.bank[idx(Resource::WOOL)] == bank_before[idx(Resource::WOOL)] + 1);
    CHECK(s.pub.bank[idx(Resource::GRAIN)] == bank_before[idx(Resource::GRAIN)] + 1);
    CHECK(s.pub.bank[idx(Resource::ORE)] == bank_before[idx(Resource::ORE)] + 1);

    CHECK(s.pub.pending_chance == static_cast<uint8_t>(ChanceKind::DEV_DRAW));
    CHECK(s.pub.dev_count[0] == 0);  // the card arrives with the chance outcome
    const Decision d = decide(s);
    REQUIRE(d.kind == DecisionKind::CHANCE);
    CHECK(d.chance.kind == ChanceKind::DEV_DRAW);
    CHECK(d.chance.n_outcomes == NUM_DEV);
    CHECK(d.chance.total == s.pub.dev_deck_remaining);
    CHECK(d.chance.total == DEV_DECK_SIZE);
    for (int c = 0; c < NUM_DEV; ++c) CHECK(d.chance.weight[c] == DEV_DECK_COUNTS[c]);
}

TEST_CASE("devcards: a drawn card goes to the buyer's hand", "[devcards]") {
    const int c = GENERATE(0, 1, 2, 3, 4);
    GameState s = blank();
    give_buy_cost(s, 0);
    step(s, act(ActionType::BUY_DEV));
    const EventList ev = draw(s, static_cast<DevCard>(c));

    const auto bought = events_of(ev, EventType::DEV_BOUGHT);
    REQUIRE(bought.size() == 1);
    CHECK(bought[0].player == 0);
    CHECK(bought[0].a == c);
    CHECK(s.hid.dev_held[0][c] == 1);
    CHECK(s.hid.dev_deck[c] == DEV_DECK_COUNTS[c] - 1);
    CHECK(s.pub.dev_deck_remaining == DEV_DECK_SIZE - 1);
    CHECK(s.pub.dev_count[0] == 1);
    CHECK(s.pub.pending_chance == 0);
    CHECK(s.pub.phase == Phase::MAIN);
    CHECK(s.hid.dev_bought_this_turn[c] == 1);
}

TEST_CASE("devcards: chance weights follow the remaining deck", "[devcards]") {
    GameState s = blank();
    give_dev(s, 1, DevCard::KNIGHT, 10);
    give_dev(s, 2, DevCard::MONOPOLY, 2);
    give_buy_cost(s, 0);
    step(s, act(ActionType::BUY_DEV));

    const Decision d = decide(s);
    REQUIRE(d.kind == DecisionKind::CHANCE);
    const array<int, NUM_DEV> expected = {4, 5, 2, 2, 0};
    for (int c = 0; c < NUM_DEV; ++c) CHECK(d.chance.weight[c] == expected[c]);
    CHECK(d.chance.total == 13);
    CHECK(d.chance.total == s.pub.dev_deck_remaining);
}

TEST_CASE("devcards: BUY_DEV needs all three resources", "[devcards]") {
    for (int missing : {idx(Resource::WOOL), idx(Resource::GRAIN), idx(Resource::ORE)}) {
        GameState s = blank();
        for (int r : {idx(Resource::WOOL), idx(Resource::GRAIN), idx(Resource::ORE)}) {
            if (r != missing) give(s, 0, static_cast<Resource>(r), 1);
        }
        CHECK_FALSE(type_legal(decide(s).mask, ActionType::BUY_DEV));
    }
}

TEST_CASE("devcards: BUY_DEV is masked once the deck is empty", "[devcards]") {
    GameState s = blank();
    give_dev(s, 1, DevCard::KNIGHT, 14);
    give_dev(s, 1, DevCard::VICTORY_POINT, 5);
    give_dev(s, 1, DevCard::ROAD_BUILDING, 2);
    give_dev(s, 1, DevCard::YEAR_OF_PLENTY, 2);
    give_dev(s, 1, DevCard::MONOPOLY, 1);
    give_buy_cost(s, 0);
    REQUIRE(s.pub.dev_deck_remaining == 1);
    CHECK(type_legal(decide(s).mask, ActionType::BUY_DEV));

    step(s, act(ActionType::BUY_DEV));
    const Decision d = decide(s);
    REQUIRE(d.kind == DecisionKind::CHANCE);
    CHECK(d.chance.total == 1);
    CHECK(d.chance.weight[idx(DevCard::MONOPOLY)] == 1);
    draw(s, DevCard::MONOPOLY);

    REQUIRE(s.pub.dev_deck_remaining == 0);
    give_buy_cost(s, 0);
    CHECK_FALSE(type_legal(decide(s).mask, ActionType::BUY_DEV));
}

// ---------------------------------------------------------------- playability

TEST_CASE("devcards: a card bought this turn is locked", "[devcards]") {
    GameState s = blank();
    buy(s, DevCard::KNIGHT);
    const Decision d = decide(s);
    CHECK(d.mask.devcard == 0);
    CHECK_FALSE(type_legal(d.mask, ActionType::PLAY_DEV));
    CHECK_FALSE(legal(s, play(DevCard::KNIGHT)));
}

TEST_CASE("devcards: an older copy of the same type stays playable", "[devcards]") {
    GameState s = blank();
    give_dev(s, 0, DevCard::KNIGHT);
    buy(s, DevCard::KNIGHT);
    REQUIRE(s.hid.dev_held[0][idx(DevCard::KNIGHT)] == 2);
    const Decision d = decide(s);
    CHECK(d.mask.devcard == dbit(DevCard::KNIGHT));
    CHECK(type_legal(d.mask, ActionType::PLAY_DEV));
}

TEST_CASE("devcards: an older card of another type does not unlock the new one", "[devcards]") {
    GameState s = blank();
    give_dev(s, 0, DevCard::KNIGHT);
    buy(s, DevCard::MONOPOLY);
    CHECK(decide(s).mask.devcard == dbit(DevCard::KNIGHT));
}

TEST_CASE("devcards: a bought card becomes playable on the buyer's next turn", "[devcards]") {
    GameState s = blank();
    buy(s, DevCard::MONOPOLY);
    CHECK(decide(s).mask.devcard == 0);

    pass_turns(s);
    REQUIRE(s.pub.current_player == 0);
    REQUIRE(s.pub.phase == Phase::ROLL);
    for (int c = 0; c < NUM_DEV; ++c) CHECK(s.hid.dev_bought_this_turn[c] == 0);
    const Decision d = decide(s);
    CHECK(d.mask.devcard == dbit(DevCard::MONOPOLY));
    CHECK(type_legal(d.mask, ActionType::ROLL));
    CHECK(type_legal(d.mask, ActionType::PLAY_DEV));

    step(s, act(ActionType::ROLL));
    resolve(s, ChanceKind::DICE, 6);
    CHECK(decide(s).mask.devcard == dbit(DevCard::MONOPOLY));
}

TEST_CASE("devcards: only one development card may be played per turn", "[devcards]") {
    GameState s = blank();
    give_dev(s, 0, DevCard::KNIGHT, 2);
    give_dev(s, 0, DevCard::MONOPOLY);
    CHECK(decide(s).mask.devcard == (dbit(DevCard::KNIGHT) | dbit(DevCard::MONOPOLY)));

    step(s, play(DevCard::KNIGHT));
    move_robber_away(s);
    REQUIRE(s.pub.phase == Phase::MAIN);
    CHECK(s.pub.played_dev_this_turn == 1);
    CHECK(s.pub.dev_count[0] == 2);  // still holding cards
    const Decision d = decide(s);
    CHECK(d.mask.devcard == 0);
    CHECK_FALSE(type_legal(d.mask, ActionType::PLAY_DEV));
    CHECK(type_legal(d.mask, ActionType::END_TURN));

    pass_turns(s);  // the allowance resets with the turn
    CHECK(decide(s).mask.devcard == (dbit(DevCard::KNIGHT) | dbit(DevCard::MONOPOLY)));
}

TEST_CASE("devcards: victory point cards are never playable", "[devcards]") {
    GameState s = blank();
    give_dev(s, 0, DevCard::VICTORY_POINT, 2);
    CHECK(decide(s).mask.devcard == 0);
    CHECK_FALSE(legal(s, play(DevCard::VICTORY_POINT)));
    to_roll(s);
    CHECK(decide(s).mask.devcard == 0);
    CHECK_FALSE(legal(s, play(DevCard::VICTORY_POINT)));
}

TEST_CASE("devcards: cards can be played before rolling", "[devcards]") {
    GameState s = blank();
    to_roll(s);
    give_dev(s, 0, DevCard::KNIGHT);
    give_dev(s, 0, DevCard::MONOPOLY);
    const Decision d = decide(s);
    CHECK(d.mask.type == (tbit(ActionType::ROLL) | tbit(ActionType::PLAY_DEV)));
    CHECK(d.mask.devcard == (dbit(DevCard::KNIGHT) | dbit(DevCard::MONOPOLY)));
}

// ---------------------------------------------------------------- Knight

TEST_CASE("devcards: Knight raises the army count and emits DEV_PLAYED", "[devcards]") {
    GameState s = blank();
    give_dev(s, 0, DevCard::KNIGHT);
    const EventList ev = step(s, play(DevCard::KNIGHT));
    const auto played = events_of(ev, EventType::DEV_PLAYED);
    REQUIRE(played.size() == 1);
    CHECK(played[0].player == 0);
    CHECK(played[0].a == idx(DevCard::KNIGHT));
    CHECK(s.pub.knights_played[0] == 1);
    CHECK(s.pub.dev_count[0] == 0);
    CHECK(s.hid.dev_held[0][idx(DevCard::KNIGHT)] == 0);
    CHECK(s.pub.played_dev_this_turn == 1);
    CHECK(s.pub.phase == Phase::MOVE_ROBBER);
}

TEST_CASE("devcards: Knight moves the robber, steals, and resumes the turn", "[devcards]") {
    const bool before_roll = GENERATE(true, false);
    GameState s = blank();
    if (before_roll) to_roll(s);
    const uint8_t hex = s.pub.robber_hex == 9 ? 4 : 9;
    place_settlement(s, 1, topology().hex_vertices[hex][0]);
    give(s, 1, Resource::ORE, 2);
    give(s, 1, Resource::GRAIN, 1);
    give_dev(s, 0, DevCard::KNIGHT);

    step(s, play(DevCard::KNIGHT));
    Decision d = decide(s);
    REQUIRE(d.phase == Phase::MOVE_ROBBER);
    CHECK(d.mask.type == tbit(ActionType::MOVE_ROBBER));
    CHECK(popcount(d.mask.tile) == NUM_HEXES - 1);
    CHECK(((d.mask.tile >> s.pub.robber_hex) & 1) == 0);

    const EventList moved = step(s, act(ActionType::MOVE_ROBBER, hex));
    const auto rm = events_of(moved, EventType::ROBBER_MOVED);
    REQUIRE(rm.size() == 1);
    CHECK(rm[0].a == hex);
    d = decide(s);
    REQUIRE(d.phase == Phase::STEAL);
    CHECK(d.mask.player == 1);  // seat 1 is relative seat 0 from seat 0

    step(s, act(ActionType::STEAL, 0));
    d = decide(s);
    REQUIRE(d.kind == DecisionKind::CHANCE);
    CHECK(d.chance.kind == ChanceKind::STEAL);
    CHECK(d.chance.total == 3);
    CHECK(d.chance.weight[idx(Resource::GRAIN)] == 1);
    CHECK(d.chance.weight[idx(Resource::ORE)] == 2);

    const EventList stolen = resolve(s, ChanceKind::STEAL, idx(Resource::ORE));
    const auto st = events_of(stolen, EventType::STEAL);
    REQUIRE(st.size() == 1);
    CHECK(st[0].player == 0);
    CHECK(st[0].target == 1);
    CHECK(st[0].a == idx(Resource::ORE));
    CHECK(s.pub.hand_size[0] == 1);
    CHECK(s.pub.hand_size[1] == 2);
    CHECK(s.hid.hands[0][idx(Resource::ORE)] == 1);

    CHECK(s.pub.phase == (before_roll ? Phase::ROLL : Phase::MAIN));
    d = decide(s);
    CHECK_FALSE(type_legal(d.mask, ActionType::PLAY_DEV));
    CHECK(type_legal(d.mask, before_roll ? ActionType::ROLL : ActionType::END_TURN));
}

TEST_CASE("devcards: Knight without a victim resumes the turn directly", "[devcards]") {
    const bool before_roll = GENERATE(true, false);
    GameState s = blank();
    if (before_roll) to_roll(s);
    give_dev(s, 0, DevCard::KNIGHT);
    step(s, play(DevCard::KNIGHT));
    move_robber_away(s);
    CHECK(s.pub.phase == (before_roll ? Phase::ROLL : Phase::MAIN));
    CHECK(s.pub.dice == (before_roll ? 0 : 8));
}

// ---------------------------------------------------------------- Road Building

TEST_CASE("devcards: Road Building places two free roads", "[devcards]") {
    GameState s = rb_state(false);
    const auto bank_before = s.pub.bank;
    const auto hand_before = s.pub.hand_size;

    const EventList played = step(s, play(DevCard::ROAD_BUILDING));
    CHECK(events_of(played, EventType::DEV_PLAYED).size() == 1);
    CHECK(s.pub.phase == Phase::ROAD_BUILDING);
    CHECK(s.pub.free_roads == 2);
    Decision d = decide(s);
    CHECK(d.mask.type == tbit(ActionType::BUILD_ROAD));
    CHECK(d.mask.edge == edge_set({edge_between(18, 13), edge_between(18, 23),
                                   edge_between(18, 24)}));

    const uint8_t first = edge_between(18, 13);
    const EventList e1 = step(s, act(ActionType::BUILD_ROAD, first));
    CHECK(events_of(e1, EventType::ROAD_BUILT).size() == 1);
    CHECK(events_of(e1, EventType::SPEND).empty());
    CHECK(s.pub.free_roads == 1);
    CHECK(s.pub.pieces_left[0][PIECE_ROAD] == ROADS_PER_PLAYER - 1);
    d = decide(s);
    REQUIRE(d.phase == Phase::ROAD_BUILDING);
    CHECK(d.mask.edge == edge_set({edge_between(18, 23), edge_between(18, 24),
                                   edge_between(13, 8), edge_between(13, 9)}));

    const EventList e2 = step(s, act(ActionType::BUILD_ROAD, edge_between(13, 8)));
    CHECK(events_of(e2, EventType::SPEND).empty());
    CHECK(s.pub.phase == Phase::MAIN);
    CHECK(s.pub.free_roads == 0);
    CHECK(s.pub.pieces_left[0][PIECE_ROAD] == ROADS_PER_PLAYER - 2);
    CHECK(edge_count(s.pub.roads[0]) == 2);
    CHECK(s.pub.bank == bank_before);
    CHECK(s.pub.hand_size == hand_before);
    CHECK(s.pub.dev_count[0] == 0);
    CHECK(decide(s).mask.devcard == 0);
}

TEST_CASE("devcards: Road Building before the roll returns to ROLL", "[devcards]") {
    GameState s = rb_state(true);
    step(s, play(DevCard::ROAD_BUILDING));
    step(s, act(ActionType::BUILD_ROAD, edge_between(18, 13)));
    step(s, act(ActionType::BUILD_ROAD, edge_between(18, 23)));
    CHECK(s.pub.phase == Phase::ROLL);
    CHECK(s.pub.dice == 0);
    CHECK(type_legal(decide(s).mask, ActionType::ROLL));
}

TEST_CASE("devcards: Road Building ends after one road with one piece left", "[devcards]") {
    GameState s = rb_state(false);
    fill_roads(s, 0, ROADS_PER_PLAYER - 1);
    REQUIRE(s.pub.pieces_left[0][PIECE_ROAD] == 1);
    CHECK(decide(s).mask.devcard == dbit(DevCard::ROAD_BUILDING));

    step(s, play(DevCard::ROAD_BUILDING));
    CHECK(s.pub.free_roads == 1);
    step(s, act(ActionType::BUILD_ROAD, edge_between(18, 13)));
    CHECK(s.pub.phase == Phase::MAIN);
    CHECK(s.pub.free_roads == 0);
    CHECK(s.pub.pieces_left[0][PIECE_ROAD] == 0);
}

TEST_CASE("devcards: Road Building ends early when no edge remains", "[devcards]") {
    GameState s = blank();
    place_settlement(s, 0, 0);
    // Vertex 0 touches edges to 3 and 4; vertex 3 touches only (0,3) and (3,7).
    place_road(s, 1, edge_between(0, 4));
    place_road(s, 1, edge_between(3, 7));
    refresh_awards(s);
    give_dev(s, 0, DevCard::ROAD_BUILDING);

    step(s, play(DevCard::ROAD_BUILDING));
    CHECK(s.pub.free_roads == 2);
    const Decision d = decide(s);
    CHECK(d.mask.edge == edge_set({edge_between(0, 3)}));
    CHECK(d.n_legal == 1);

    step(s, act(ActionType::BUILD_ROAD, edge_between(0, 3)));
    CHECK(s.pub.phase == Phase::MAIN);
    CHECK(s.pub.free_roads == 0);
    CHECK(s.pub.pieces_left[0][PIECE_ROAD] == ROADS_PER_PLAYER - 1);
}

TEST_CASE("devcards: Road Building is masked when it cannot place a road", "[devcards]") {
    SECTION("no buildings or roads") {
        GameState s = blank();
        give_dev(s, 0, DevCard::ROAD_BUILDING);
        CHECK(decide(s).mask.devcard == 0);
        CHECK_FALSE(type_legal(decide(s).mask, ActionType::PLAY_DEV));
    }
    SECTION("every adjacent edge is taken") {
        GameState s = blank();
        place_settlement(s, 0, 0);
        place_road(s, 1, edge_between(0, 3));
        place_road(s, 1, edge_between(0, 4));
        refresh_awards(s);
        give_dev(s, 0, DevCard::ROAD_BUILDING);
        give_dev(s, 0, DevCard::KNIGHT);
        CHECK(decide(s).mask.devcard == dbit(DevCard::KNIGHT));  // only the Knight remains
    }
    SECTION("no road pieces left") {
        GameState s = rb_state(false);
        fill_roads(s, 0, ROADS_PER_PLAYER);
        REQUIRE(s.pub.pieces_left[0][PIECE_ROAD] == 0);
        CHECK(decide(s).mask.devcard == 0);
    }
}

TEST_CASE("devcards: free roads can win Longest Road", "[devcards]") {
    GameState s = blank();
    const int line[] = {0, 3, 7, 11, 16, 21, 27};
    for (int i = 0; i < 4; ++i) place_road(s, 0, edge_between(line[i], line[i + 1]));
    refresh_awards(s);
    REQUIRE(s.pub.longest_road_holder == NONE);
    give_dev(s, 0, DevCard::ROAD_BUILDING);

    step(s, play(DevCard::ROAD_BUILDING));
    const EventList e1 = step(s, act(ActionType::BUILD_ROAD, edge_between(16, 21)));
    const auto lr = events_of(e1, EventType::LONGEST_ROAD);
    REQUIRE(lr.size() == 1);
    CHECK(lr[0].player == 0);
    CHECK(s.pub.longest_road_holder == 0);
    CHECK(s.pub.public_vp[0] == 2);

    const EventList e2 = step(s, act(ActionType::BUILD_ROAD, edge_between(21, 27)));
    CHECK(events_of(e2, EventType::LONGEST_ROAD).empty());
    CHECK(s.pub.road_len[0] == 6);
    CHECK(s.pub.phase == Phase::MAIN);
}

// ---------------------------------------------------------------- Year of Plenty

TEST_CASE("devcards: Year of Plenty takes two cards from the bank", "[devcards]") {
    GameState s = blank();
    give_dev(s, 0, DevCard::YEAR_OF_PLENTY);
    step(s, play(DevCard::YEAR_OF_PLENTY));
    CHECK(s.pub.phase == Phase::YOP_PICK);
    CHECK(s.pub.yop_remaining == 2);
    Decision d = decide(s);
    CHECK(d.mask.type == tbit(ActionType::PLAY_DEV));
    CHECK(d.mask.resource == 0x1F);

    const EventList e1 = step(s, pick(Resource::ORE));
    const auto g1 = events_of(e1, EventType::GAIN);
    REQUIRE(g1.size() == 1);
    CHECK(g1[0].player == 0);
    CHECK(g1[0].a == idx(Resource::ORE));
    CHECK(g1[0].b == 1);
    CHECK(s.pub.phase == Phase::YOP_PICK);
    CHECK(s.pub.yop_remaining == 1);

    step(s, pick(Resource::BRICK));
    CHECK(s.pub.phase == Phase::MAIN);
    CHECK(s.pub.yop_remaining == 0);
    CHECK(s.hid.hands[0][idx(Resource::ORE)] == 1);
    CHECK(s.hid.hands[0][idx(Resource::BRICK)] == 1);
    CHECK(s.pub.hand_size[0] == 2);
    CHECK(s.pub.bank[idx(Resource::ORE)] == RESOURCE_PER_TYPE - 1);
    CHECK(s.pub.bank[idx(Resource::BRICK)] == RESOURCE_PER_TYPE - 1);
}

TEST_CASE("devcards: Year of Plenty may take the same resource twice", "[devcards]") {
    GameState s = blank();
    give_dev(s, 0, DevCard::YEAR_OF_PLENTY);
    step(s, play(DevCard::YEAR_OF_PLENTY));
    step(s, pick(Resource::GRAIN));
    CHECK(legal(s, pick(Resource::GRAIN)));
    step(s, pick(Resource::GRAIN));
    CHECK(s.hid.hands[0][idx(Resource::GRAIN)] == 2);
    CHECK(s.pub.bank[idx(Resource::GRAIN)] == RESOURCE_PER_TYPE - 2);
    CHECK(s.pub.phase == Phase::MAIN);
}

TEST_CASE("devcards: Year of Plenty can only pick resources the bank holds", "[devcards]") {
    GameState s = blank();
    give(s, 1, Resource::BRICK, 19);  // bank brick = 0
    give(s, 1, Resource::LUMBER, 18);  // bank lumber = 1
    give_dev(s, 0, DevCard::YEAR_OF_PLENTY);
    step(s, play(DevCard::YEAR_OF_PLENTY));
    Decision d = decide(s);
    CHECK(d.mask.resource == (rbit(Resource::LUMBER) | rbit(Resource::WOOL) |
                              rbit(Resource::GRAIN) | rbit(Resource::ORE)));
    CHECK_FALSE(legal(s, pick(Resource::BRICK)));

    step(s, pick(Resource::LUMBER));  // drains lumber
    d = decide(s);
    REQUIRE(d.phase == Phase::YOP_PICK);
    CHECK(d.mask.resource == (rbit(Resource::WOOL) | rbit(Resource::GRAIN) | rbit(Resource::ORE)));
    CHECK_FALSE(legal(s, pick(Resource::LUMBER)));
    step(s, pick(Resource::WOOL));
    CHECK(s.pub.phase == Phase::MAIN);
}

TEST_CASE("devcards: Year of Plenty ends after one pick when the bank runs dry", "[devcards]") {
    GameState s = blank();
    for (int r = 0; r < NUM_RES; ++r) {
        give(s, 1, static_cast<Resource>(r), r == idx(Resource::ORE) ? 18 : 19);
    }
    give_dev(s, 0, DevCard::YEAR_OF_PLENTY);
    REQUIRE(decide(s).mask.devcard == dbit(DevCard::YEAR_OF_PLENTY));

    step(s, play(DevCard::YEAR_OF_PLENTY));
    CHECK(decide(s).mask.resource == rbit(Resource::ORE));
    step(s, pick(Resource::ORE));
    CHECK(s.pub.phase == Phase::MAIN);
    CHECK(s.pub.yop_remaining == 0);
    CHECK(s.pub.hand_size[0] == 1);
    for (int r = 0; r < NUM_RES; ++r) CHECK(s.pub.bank[r] == 0);
}

TEST_CASE("devcards: Year of Plenty is masked when the bank is empty", "[devcards]") {
    GameState s = blank();
    for (int r = 0; r < NUM_RES; ++r) give(s, 1, static_cast<Resource>(r), 19);
    give_dev(s, 0, DevCard::YEAR_OF_PLENTY);
    give_dev(s, 0, DevCard::MONOPOLY);
    CHECK(decide(s).mask.devcard == dbit(DevCard::MONOPOLY));
}

TEST_CASE("devcards: Year of Plenty before the roll returns to ROLL", "[devcards]") {
    GameState s = blank();
    to_roll(s);
    give_dev(s, 0, DevCard::YEAR_OF_PLENTY);
    step(s, play(DevCard::YEAR_OF_PLENTY));
    step(s, pick(Resource::WOOL));
    step(s, pick(Resource::WOOL));
    CHECK(s.pub.phase == Phase::ROLL);
    CHECK(s.pub.dice == 0);
}

// ---------------------------------------------------------------- Monopoly

TEST_CASE("devcards: Monopoly takes the resource from every opponent", "[devcards]") {
    GameState s = blank();
    give(s, 0, Resource::BRICK, 1);
    give(s, 1, Resource::BRICK, 3);
    give(s, 1, Resource::ORE, 2);
    give(s, 3, Resource::BRICK, 2);
    give_dev(s, 0, DevCard::MONOPOLY);
    const auto bank_before = s.pub.bank;

    step(s, play(DevCard::MONOPOLY));
    CHECK(s.pub.phase == Phase::MONOPOLY_PICK);
    const Decision d = decide(s);
    CHECK(d.mask.type == tbit(ActionType::PLAY_DEV));
    CHECK(d.mask.resource == 0x1F);

    const EventList ev = step(s, pick(Resource::BRICK));
    const auto mono = events_of(ev, EventType::MONOPOLY);
    REQUIRE(mono.size() == 3);
    const array<int, 3> counts = {3, 0, 2};  // opponents 1, 2, 3 in seat order
    for (int i = 0; i < 3; ++i) {
        CHECK(mono[i].player == 0);
        CHECK(mono[i].target == i + 1);
        CHECK(mono[i].a == idx(Resource::BRICK));
        CHECK(mono[i].b == counts[i]);
    }
    CHECK(s.hid.hands[0][idx(Resource::BRICK)] == 6);
    CHECK(s.pub.hand_size[0] == 6);
    CHECK(s.pub.hand_size[1] == 2);
    CHECK(s.hid.hands[1][idx(Resource::ORE)] == 2);
    CHECK(s.pub.hand_size[2] == 0);
    CHECK(s.pub.hand_size[3] == 0);
    CHECK(s.pub.bank == bank_before);
    CHECK(s.pub.phase == Phase::MAIN);
    CHECK(s.pub.dev_count[0] == 0);
}

TEST_CASE("devcards: Monopoly reports zero counts when nobody holds the resource",
          "[devcards]") {
    const bool before_roll = GENERATE(true, false);
    GameState s = blank();
    if (before_roll) to_roll(s);
    give(s, 2, Resource::WOOL, 4);
    give_dev(s, 0, DevCard::MONOPOLY);
    step(s, play(DevCard::MONOPOLY));
    const EventList ev = step(s, pick(Resource::GRAIN));
    const auto mono = events_of(ev, EventType::MONOPOLY);
    REQUIRE(mono.size() == 3);
    for (const Event& e : mono) CHECK(e.b == 0);
    CHECK(s.pub.hand_size[0] == 0);
    CHECK(s.pub.hand_size[2] == 4);
    CHECK(s.pub.phase == (before_roll ? Phase::ROLL : Phase::MAIN));
}

// ---------------------------------------------------------------- victory points

TEST_CASE("devcards: drawing the tenth point on your own turn wins at once", "[devcards]") {
    GameState s = blank();
    give_cities(s, 0, {2, 10, 20, 32});
    place_settlement(s, 0, 42);
    REQUIRE(s.pub.public_vp[0] == 9);
    give_buy_cost(s, 0);
    step(s, act(ActionType::BUY_DEV));

    const EventList ev = draw(s, DevCard::VICTORY_POINT);
    CHECK(events_of(ev, EventType::DEV_BOUGHT).size() == 1);
    const auto over = events_of(ev, EventType::GAME_OVER);
    REQUIRE(over.size() == 1);
    CHECK(over[0].player == 0);
    CHECK(s.pub.phase == Phase::TERMINAL);
    CHECK(decide(s).kind == DecisionKind::TERMINAL);
}

TEST_CASE("devcards: held victory points are hidden and can fall short", "[devcards]") {
    GameState s = blank();
    give_cities(s, 0, {2, 10, 20, 32});
    REQUIRE(s.pub.public_vp[0] == 8);
    give_buy_cost(s, 0);
    step(s, act(ActionType::BUY_DEV));
    const EventList ev = draw(s, DevCard::VICTORY_POINT);

    CHECK(events_of(ev, EventType::GAME_OVER).empty());  // 8 + 1 = 9
    CHECK(s.pub.public_vp[0] == 8);                      // the card is not public
    CHECK(s.hid.dev_held[0][idx(DevCard::VICTORY_POINT)] == 1);
    CHECK(s.pub.dev_count[0] == 1);
    CHECK(s.pub.phase == Phase::MAIN);
}

TEST_CASE("devcards: held victory point cards count toward ten", "[devcards]") {
    GameState s = blank();
    give_cities(s, 0, {2, 10, 20, 32});
    give_dev(s, 0, DevCard::VICTORY_POINT, 2);  // 8 public + 2 hidden
    give_buy_cost(s, 0);
    const EventList ev = apply(board(), s, act(ActionType::BUY_DEV));
    const auto over = events_of(ev, EventType::GAME_OVER);
    REQUIRE(over.size() == 1);
    CHECK(over[0].player == 0);
    CHECK(s.pub.phase == Phase::TERMINAL);
}
