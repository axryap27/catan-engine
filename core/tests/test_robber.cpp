#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdint>
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

using namespace std;
using namespace catan;
using namespace catan::test;

namespace {

constexpr uint8_t BRICK = 0, LUMBER = 1, WOOL = 2, GRAIN = 3, ORE = 4;
constexpr uint32_t ALL_HEXES = (uint32_t{1} << NUM_HEXES) - 1;

const Board& board() {
    static const Board b = standard_board();
    return b;
}

// Empty board in MAIN for player p (dice 8), empty hands, robber on the desert (hex 9).
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

// Rolls a 7 from a ROLL state.
EventList roll_seven(GameState& s) {
    apply(board(), s, Action{ActionType::ROLL, 0});
    return apply_chance(board(), s, 5);
}

// Hand with the given counts per resource (brick, lumber, wool, grain, ore).
void hand(GameState& s, uint8_t p, array<int, NUM_RES> counts) {
    for (int r = 0; r < NUM_RES; ++r) give(s, p, static_cast<Resource>(r), counts[r]);
}

// Moves the robber from MOVE_ROBBER to `hex`.
EventList move_to(GameState& s, int hex) {
    return apply(board(), s, Action{ActionType::MOVE_ROBBER, static_cast<uint8_t>(hex)});
}

// Phase MOVE_ROBBER for player p, dice 8 (a roll of 7 already resolved is not needed).
GameState robber_state(uint8_t p = 0) {
    GameState s = blank(p);
    s.pub.phase = Phase::MOVE_ROBBER;
    return s;
}

}  // namespace

TEST_CASE("rolling a 7 produces nothing", "[robber]") {
    GameState s = blank_roll();
    place_settlement(s, 0, hv(0, 0));
    place_settlement(s, 1, hv(11, 3));
    const EventList ev = roll_seven(s);
    CHECK(s.pub.dice == 7);
    CHECK(events_of(ev, EventType::GAIN).empty());
    CHECK(s.pub.hand_size[0] == 0);
    CHECK(s.pub.hand_size[1] == 0);
    CHECK(events_of(ev, EventType::DICE_ROLLED)[0].a == 7);
}

TEST_CASE("a 7 with nobody over the limit goes straight to MOVE_ROBBER", "[robber]") {
    GameState s = blank_roll(1);
    hand(s, 0, {2, 2, 1, 1, 1});  // exactly 7: safe
    roll_seven(s);
    CHECK(s.pub.phase == Phase::MOVE_ROBBER);
    CHECK(s.pub.pending_discard == array<uint8_t, NUM_PLAYERS>{0, 0, 0, 0});
    const Decision d = next_decision(board(), s);
    CHECK(d.player == 1);
    CHECK(d.phase == Phase::MOVE_ROBBER);
}

TEST_CASE("players over 7 cards owe half rounded down", "[robber]") {
    GameState s = blank_roll(0);
    hand(s, 0, {2, 2, 2, 1, 1});    // 8 -> 4
    hand(s, 1, {2, 2, 1, 1, 1});    // 7 -> 0
    hand(s, 2, {3, 3, 1, 1, 1});    // 9 -> 4
    hand(s, 3, {3, 3, 3, 1, 1});    // 11 -> 5
    ok(s);
    roll_seven(s);
    CHECK(s.pub.phase == Phase::DISCARD);
    CHECK(s.pub.pending_discard == array<uint8_t, NUM_PLAYERS>{4, 0, 4, 5});
    ok(s);
}

TEST_CASE("fifteen cards owe seven", "[robber]") {
    GameState s = blank_roll(0);
    hand(s, 2, {5, 5, 5, 0, 0});
    roll_seven(s);
    CHECK(s.pub.pending_discard[2] == 7);
}

TEST_CASE("no cards are taken from players under the limit", "[robber]") {
    GameState s = blank_roll(0);
    hand(s, 1, {1, 1, 1, 1, 1});
    hand(s, 2, {4, 4, 0, 0, 0});
    roll_seven(s);
    CHECK(s.pub.hand_size[1] == 5);
    CHECK(s.pub.hand_size[2] == 8);  // only owes; nothing is discarded yet
    CHECK(s.pub.pending_discard[1] == 0);
}

TEST_CASE("discarders go clockwise starting from the current player", "[robber]") {
    GameState s = blank_roll(2);
    hand(s, 0, {8, 0, 0, 0, 0});
    hand(s, 1, {1, 1, 1, 1, 3});  // 7: exempt
    hand(s, 2, {0, 8, 0, 0, 0});
    hand(s, 3, {0, 0, 8, 0, 0});
    roll_seven(s);
    REQUIRE(s.pub.phase == Phase::DISCARD);

    vector<uint8_t> order;
    while (s.pub.phase == Phase::DISCARD) {
        const Decision d = next_decision(board(), s);
        REQUIRE(d.kind == DecisionKind::PLAYER);
        CHECK(d.phase == Phase::DISCARD);
        order.push_back(d.player);
        apply(board(), s, nth_legal(d.mask, d.phase, 0));
    }
    const vector<uint8_t> expected = {2, 2, 2, 2, 3, 3, 3, 3, 0, 0, 0, 0};
    CHECK(order == expected);
    CHECK(s.pub.phase == Phase::MOVE_ROBBER);
    CHECK(next_decision(board(), s).player == 2);
    ok(s);
}

TEST_CASE("a discard moves one card to the bank and emits DISCARD", "[robber]") {
    GameState s = blank_roll(0);
    hand(s, 0, {0, 0, 0, 4, 4});  // 8 -> 4
    roll_seven(s);
    const Decision d = next_decision(board(), s);
    REQUIRE(d.kind == DecisionKind::PLAYER);
    CHECK(d.player == 0);
    CHECK(d.mask.resource == ((1u << GRAIN) | (1u << ORE)));
    CHECK(d.n_legal == 2);

    const EventList ev = apply(board(), s, Action{ActionType::DISCARD, GRAIN});
    CHECK(s.hid.hands[0][GRAIN] == 3);
    CHECK(s.pub.hand_size[0] == 7);
    CHECK(s.pub.bank[GRAIN] == 19 - 3);
    CHECK(s.pub.pending_discard[0] == 3);
    CHECK(s.pub.phase == Phase::DISCARD);
    const auto dis = events_of(ev, EventType::DISCARD);
    REQUIRE(dis.size() == 1);
    CHECK(dis[0].player == 0);
    CHECK(dis[0].a == GRAIN);
    CHECK(dis[0].b == 1);
    ok(s);
}

TEST_CASE("the last discard moves on to MOVE_ROBBER", "[robber]") {
    GameState s = blank_roll(0);
    hand(s, 0, {8, 0, 0, 0, 0});
    roll_seven(s);
    for (int i = 0; i < 3; ++i) {
        apply(board(), s, Action{ActionType::DISCARD, BRICK});
        CHECK(s.pub.phase == Phase::DISCARD);
    }
    apply(board(), s, Action{ActionType::DISCARD, BRICK});
    CHECK(s.pub.phase == Phase::MOVE_ROBBER);
    CHECK(s.hid.hands[0][BRICK] == 4);
    CHECK(s.pub.bank[BRICK] == 15);
    ok(s);
}

TEST_CASE("discards are non-decreasing and always completable", "[robber]") {
    // 2 brick, 2 lumber, 2 grain, 2 ore = 8 cards, owes 4.
    GameState s = blank_roll(0);
    hand(s, 0, {2, 2, 0, 2, 2});
    roll_seven(s);

    // Ore alone cannot cover 4 cards (only 2 at or above it), so it is excluded for now.
    Decision d = next_decision(board(), s);
    CHECK(d.mask.resource == ((1u << BRICK) | (1u << LUMBER) | (1u << GRAIN)));

    apply(board(), s, Action{ActionType::DISCARD, GRAIN});  // owes 3, floor grain
    CHECK(s.pub.discard_floor == GRAIN);
    d = next_decision(board(), s);
    // Brick and lumber are below the floor; ore has 2 < 3 owed. Only grain (1 + 2 = 3) works.
    CHECK(d.mask.resource == (1u << GRAIN));

    apply(board(), s, Action{ActionType::DISCARD, GRAIN});  // owes 2, grain exhausted
    d = next_decision(board(), s);
    CHECK(d.mask.resource == (1u << ORE));

    apply(board(), s, Action{ActionType::DISCARD, ORE});
    apply(board(), s, Action{ActionType::DISCARD, ORE});
    CHECK(s.pub.phase == Phase::MOVE_ROBBER);
    CHECK(s.hid.hands[0] == array<uint8_t, NUM_RES>{2, 2, 0, 0, 0});
    ok(s);
}

TEST_CASE("a low resource is excluded when higher ones cannot cover the rest", "[robber]") {
    // 1 brick + 7 ore, owes 4: brick is fine first (7 ore remain), then only ore.
    GameState s = blank_roll(0);
    hand(s, 0, {1, 0, 0, 0, 7});
    roll_seven(s);
    CHECK(next_decision(board(), s).mask.resource == ((1u << BRICK) | (1u << ORE)));
    apply(board(), s, Action{ActionType::DISCARD, ORE});
    // Floor is now ore: brick is below it.
    CHECK(next_decision(board(), s).mask.resource == (1u << ORE));
}

TEST_CASE("the discard floor resets for the next player", "[robber]") {
    GameState s = blank_roll(0);
    hand(s, 0, {0, 0, 0, 0, 8});
    hand(s, 1, {4, 0, 0, 0, 4});
    roll_seven(s);
    for (int i = 0; i < 4; ++i) apply(board(), s, Action{ActionType::DISCARD, ORE});
    REQUIRE(s.pub.phase == Phase::DISCARD);
    const Decision d = next_decision(board(), s);
    CHECK(d.player == 1);
    CHECK(d.mask.resource == ((1u << BRICK) | (1u << ORE)));  // brick allowed again
}

TEST_CASE("random discards always finish with exactly half gone", "[robber]") {
    for (uint64_t seed = 1; seed <= 40; ++seed) {
        GameState s = blank_roll(static_cast<uint8_t>(seed % NUM_PLAYERS));
        uint64_t i = 0;
        for (uint8_t p = 0; p < NUM_PLAYERS; ++p) {
            const auto n = random_below(seed, 5, i++, 16);  // 0..15 cards
            for (uint32_t c = 0; c < n; ++c) {
                const auto r = random_below(seed, 5, i++, NUM_RES);
                if (s.pub.bank[r] > 0) give(s, p, static_cast<Resource>(r), 1);
            }
        }
        const auto before = s.pub.hand_size;
        roll_seven(s);

        array<int, NUM_PLAYERS> discarded{};
        array<int, NUM_PLAYERS> last{};
        last.fill(-1);
        while (s.pub.phase == Phase::DISCARD) {
            const Decision d = next_decision(board(), s);
            REQUIRE(d.n_legal >= 1);
            const auto pick = random_below(seed, 6, i++, d.n_legal);
            const Action a = nth_legal(d.mask, d.phase, static_cast<int>(pick));
            CHECK(a.arg >= last[d.player]);  // non-decreasing within one player's discards
            last[d.player] = a.arg;
            ++discarded[d.player];
            apply(board(), s, a);
        }
        CAPTURE(seed);
        for (int p = 0; p < NUM_PLAYERS; ++p) {
            const int owed = before[p] > 7 ? before[p] / 2 : 0;
            CHECK(discarded[p] == owed);
            CHECK(s.pub.hand_size[p] == before[p] - owed);
        }
        CHECK(s.pub.phase == Phase::MOVE_ROBBER);
        ok(s);
    }
}

TEST_CASE("the robber can move to any hex except its current one", "[robber]") {
    GameState s = robber_state();
    REQUIRE(s.pub.robber_hex == 9);
    const Decision d = next_decision(board(), s);
    REQUIRE(d.kind == DecisionKind::PLAYER);
    CHECK(d.player == 0);
    CHECK(d.phase == Phase::MOVE_ROBBER);
    CHECK(d.n_legal == 18);
    CHECK(d.mask.tile == (ALL_HEXES & ~(uint32_t{1} << 9)));
    CHECK_FALSE(can(d, ActionType::MOVE_ROBBER, 9));
    CHECK(can(d, ActionType::MOVE_ROBBER, 0));
    CHECK(can(d, ActionType::MOVE_ROBBER, 18));
    CHECK(d.mask.type == (1u << idx(ActionType::MOVE_ROBBER)));
}

TEST_CASE("the robber can return to the desert from elsewhere", "[robber]") {
    GameState s = robber_state();
    s.pub.robber_hex = 4;
    const Decision d = next_decision(board(), s);
    CHECK(d.n_legal == 18);
    CHECK(can(d, ActionType::MOVE_ROBBER, 9));
    CHECK_FALSE(can(d, ActionType::MOVE_ROBBER, 4));
}

TEST_CASE("moving the robber updates its hex and emits ROBBER_MOVED", "[robber]") {
    GameState s = robber_state(2);
    const EventList ev = move_to(s, 13);
    CHECK(s.pub.robber_hex == 13);
    const auto moved = events_of(ev, EventType::ROBBER_MOVED);
    REQUIRE(moved.size() == 1);
    CHECK(moved[0].player == 2);
    CHECK(moved[0].a == 13);
    ok(s);
}

TEST_CASE("without victims play resumes in MAIN after a roll", "[robber]") {
    GameState s = robber_state();
    place_settlement(s, 1, hv(0, 0));  // elsewhere
    hand(s, 1, {2, 0, 0, 0, 0});
    move_to(s, 18);
    CHECK(s.pub.phase == Phase::MAIN);
    CHECK(s.pub.pending_chance == 0);
    const Decision d = next_decision(board(), s);
    CHECK(d.kind == DecisionKind::PLAYER);
    CHECK(d.phase == Phase::MAIN);
}

TEST_CASE("a victim needs a building on the hex and cards in hand", "[robber]") {
    GameState s = robber_state(0);
    place_settlement(s, 0, hv(4, 0));  // the thief's own building
    place_settlement(s, 1, hv(4, 2));  // victim
    place_settlement(s, 2, hv(4, 4));  // building but empty hand
    hand(s, 0, {0, 0, 1, 0, 0});
    hand(s, 1, {1, 0, 0, 0, 0});
    ok(s);
    move_to(s, 4);
    REQUIRE(s.pub.phase == Phase::STEAL);
    const Decision d = next_decision(board(), s);
    REQUIRE(d.kind == DecisionKind::PLAYER);
    CHECK(d.phase == Phase::STEAL);
    CHECK(d.n_legal == 1);
    // Seat 1 is relative 0 from seat 0.
    CHECK(d.mask.player == 0b001);
    CHECK(can(d, ActionType::STEAL, 0));
    CHECK_FALSE(can(d, ActionType::STEAL, 1));
}

TEST_CASE("the thief never robs themself", "[robber]") {
    GameState s = robber_state(0);
    place_settlement(s, 0, hv(4, 0));
    hand(s, 0, {3, 0, 0, 0, 0});
    move_to(s, 4);
    CHECK(s.pub.phase == Phase::MAIN);  // only the thief is there
}

TEST_CASE("only empty-handed opponents means no steal", "[robber]") {
    GameState s = robber_state(0);
    place_settlement(s, 1, hv(4, 2));
    place_city(s, 1, hv(4, 2));
    move_to(s, 4);
    CHECK(s.pub.phase == Phase::MAIN);
}

TEST_CASE("steal targets are relative seats clockwise from the thief", "[robber]") {
    GameState s = robber_state(1);
    place_settlement(s, 0, hv(4, 0));  // relative 2 from seat 1
    place_settlement(s, 2, hv(4, 2));  // relative 0
    place_settlement(s, 3, hv(4, 4));  // relative 1
    hand(s, 0, {1, 0, 0, 0, 0});
    hand(s, 2, {1, 0, 0, 0, 0});
    hand(s, 3, {1, 0, 0, 0, 0});
    ok(s);
    move_to(s, 4);
    const Decision d = next_decision(board(), s);
    CHECK(d.mask.player == 0b111);
    CHECK(d.n_legal == 3);
    CHECK(seat_from_relative(1, 0) == 2);
    CHECK(seat_from_relative(1, 1) == 3);
    CHECK(seat_from_relative(1, 2) == 0);

    GameState t = s;
    apply(board(), t, Action{ActionType::STEAL, 1});
    CHECK(t.pub.steal_victim == 3);
}

TEST_CASE("a victim with several buildings on the hex is listed once", "[robber]") {
    GameState s = robber_state(0);
    place_settlement(s, 2, hv(4, 0));
    place_settlement(s, 2, hv(4, 2));
    hand(s, 2, {1, 0, 0, 0, 0});
    move_to(s, 4);
    const Decision d = next_decision(board(), s);
    CHECK(d.mask.player == 0b001 << 1);  // seat 2 is relative 1 from seat 0
    CHECK(d.n_legal == 1);
}

TEST_CASE("choosing a victim opens a steal chance weighted by their hand", "[robber]") {
    GameState s = robber_state(0);
    place_settlement(s, 1, hv(4, 2));
    hand(s, 1, {2, 0, 0, 0, 1});
    hand(s, 0, {0, 0, 1, 0, 0});
    move_to(s, 4);
    apply(board(), s, Action{ActionType::STEAL, 0});
    CHECK(s.pub.steal_victim == 1);
    CHECK(s.pub.pending_chance == static_cast<uint8_t>(ChanceKind::STEAL));
    const Decision d = next_decision(board(), s);
    REQUIRE(d.kind == DecisionKind::CHANCE);
    CHECK(d.chance.kind == ChanceKind::STEAL);
    CHECK(d.chance.total == 3);
    CHECK(d.chance.n_outcomes == NUM_RES);
    const array<uint8_t, NUM_RES> expected = {2, 0, 0, 0, 1};
    for (int r = 0; r < NUM_RES; ++r) CHECK(d.chance.weight[r] == expected[r]);
    ok(s);
}

TEST_CASE("a steal outcome moves exactly that card from victim to thief", "[robber]") {
    GameState s = robber_state(0);
    place_settlement(s, 1, hv(4, 2));
    hand(s, 1, {2, 0, 0, 0, 1});
    hand(s, 0, {0, 0, 1, 0, 0});
    move_to(s, 4);
    apply(board(), s, Action{ActionType::STEAL, 0});
    const auto bank = s.pub.bank;
    const EventList ev = apply_chance(board(), s, ORE);

    CHECK(s.hid.hands[1] == array<uint8_t, NUM_RES>{2, 0, 0, 0, 0});
    CHECK(s.hid.hands[0] == array<uint8_t, NUM_RES>{0, 0, 1, 0, 1});
    CHECK(s.pub.hand_size[1] == 2);
    CHECK(s.pub.hand_size[0] == 2);
    CHECK(s.pub.bank == bank);
    const auto st = events_of(ev, EventType::STEAL);
    REQUIRE(st.size() == 1);
    CHECK(st[0].player == 0);
    CHECK(st[0].target == 1);
    CHECK(st[0].a == ORE);
    CHECK(st[0].b == 1);
    CHECK(s.pub.pending_chance == 0);
    CHECK(s.pub.steal_victim == NONE);
    CHECK(s.pub.phase == Phase::MAIN);
    ok(s);
}

TEST_CASE("stealing the only card empties the victim's hand", "[robber]") {
    GameState s = robber_state(3);
    place_settlement(s, 0, hv(4, 2));
    hand(s, 0, {0, 1, 0, 0, 0});
    move_to(s, 4);
    const Decision d = next_decision(board(), s);
    CHECK(d.mask.player == 0b001);  // seat 0 is the next seat after 3
    apply(board(), s, Action{ActionType::STEAL, 0});
    CHECK(next_decision(board(), s).chance.total == 1);
    apply_chance(board(), s, LUMBER);
    CHECK(s.pub.hand_size[0] == 0);
    CHECK(s.hid.hands[3][LUMBER] == 1);
    ok(s);
}

TEST_CASE("a full roll of 7 flows from dice to discard to robber to steal", "[robber]") {
    GameState s = blank_roll(0);
    place_settlement(s, 1, hv(4, 2));
    hand(s, 1, {0, 0, 0, 0, 8});  // victim, also owes 4
    roll_seven(s);
    CHECK(s.pub.phase == Phase::DISCARD);
    for (int i = 0; i < 4; ++i) {
        const Decision d = next_decision(board(), s);
        CHECK(d.player == 1);
        apply(board(), s, Action{ActionType::DISCARD, ORE});
    }
    CHECK(s.pub.phase == Phase::MOVE_ROBBER);
    move_to(s, 4);
    CHECK(s.pub.phase == Phase::STEAL);
    apply(board(), s, Action{ActionType::STEAL, 0});
    const Decision d = next_decision(board(), s);
    CHECK(d.chance.total == 4);
    CHECK(d.chance.weight[ORE] == 4);
    apply_chance(board(), s, ORE);
    CHECK(s.pub.phase == Phase::MAIN);
    CHECK(s.hid.hands[0][ORE] == 1);
    CHECK(s.hid.hands[1][ORE] == 3);
    ok(s);
}

TEST_CASE("a knight before the roll leads to the robber", "[robber]") {
    GameState s = blank_roll(0);
    give_dev(s, 0, DevCard::KNIGHT);
    apply(board(), s, Action{ActionType::PLAY_DEV, static_cast<uint8_t>(idx(DevCard::KNIGHT))});
    CHECK(s.pub.phase == Phase::MOVE_ROBBER);
    CHECK(s.pub.dice == 0);
    CHECK(next_decision(board(), s).n_legal == 18);
}

TEST_CASE("a knight before the roll with no victim returns to ROLL", "[robber]") {
    GameState s = blank_roll(0);
    give_dev(s, 0, DevCard::KNIGHT);
    apply(board(), s, Action{ActionType::PLAY_DEV, static_cast<uint8_t>(idx(DevCard::KNIGHT))});
    move_to(s, 18);
    CHECK(s.pub.phase == Phase::ROLL);
    CHECK(s.pub.dice == 0);
    const Decision d = next_decision(board(), s);
    CHECK(d.phase == Phase::ROLL);
    CHECK(can(d, ActionType::ROLL));
}

TEST_CASE("a knight before the roll then a steal returns to ROLL", "[robber]") {
    GameState s = blank_roll(0);
    place_settlement(s, 2, hv(4, 2));
    hand(s, 2, {0, 0, 3, 0, 0});
    give_dev(s, 0, DevCard::KNIGHT);
    apply(board(), s, Action{ActionType::PLAY_DEV, static_cast<uint8_t>(idx(DevCard::KNIGHT))});
    move_to(s, 4);
    REQUIRE(s.pub.phase == Phase::STEAL);
    const Decision d = next_decision(board(), s);
    CHECK(d.mask.player == 0b010);  // seat 2 is relative 1 from seat 0
    apply(board(), s, Action{ActionType::STEAL, 1});
    apply_chance(board(), s, WOOL);
    CHECK(s.pub.phase == Phase::ROLL);
    CHECK(s.pub.dice == 0);
    CHECK(s.hid.hands[0][WOOL] == 1);
    CHECK(s.hid.hands[2][WOOL] == 2);
    ok(s);
    // The player can now roll as usual.
    CHECK(next_decision(board(), s).n_legal >= 1);
}

TEST_CASE("a knight played in MAIN resumes in MAIN", "[robber]") {
    GameState s = blank(0);
    place_settlement(s, 1, hv(4, 2));
    hand(s, 1, {1, 0, 0, 0, 0});
    give_dev(s, 0, DevCard::KNIGHT);
    apply(board(), s, Action{ActionType::PLAY_DEV, static_cast<uint8_t>(idx(DevCard::KNIGHT))});
    move_to(s, 4);
    apply(board(), s, Action{ActionType::STEAL, 0});
    apply_chance(board(), s, BRICK);
    CHECK(s.pub.phase == Phase::MAIN);
    CHECK(s.pub.dice == 8);
}
