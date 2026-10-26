#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "catan/board.h"
#include "catan/check.h"
#include "catan/decision.h"
#include "catan/event.h"
#include "catan/rules.h"
#include "catan/state.h"
#include "catan/topology.h"

// Private to the rules implementation: shared helpers and one decision/apply pair per phase.
// rules.cpp dispatches on phase and pending chance; each phase group lives in its own file.
//   setup.cpp   SETUP
//   turn.cpp    ROLL, dice, production, MAIN, EXCHANGE_RECV, END_TURN, victory
//   robber.cpp  DISCARD, MOVE_ROBBER, STEAL and the steal chance
//   devcards.cpp  BUY_DEV, PLAY_DEV, ROAD_BUILDING, YOP_PICK, MONOPOLY_PICK, dev draw chance
//   awards.cpp  road length, Longest Road, Largest Army

namespace catan::detail {

inline constexpr VertexSet ALL_VERTICES = (uint64_t{1} << NUM_VERTICES) - 1;

// Positional rather than designated init: -Wmissing-field-initializers flags omitted fields.
inline Event event(EventType type, uint8_t player, uint8_t a, uint8_t b = 0,
                   uint8_t target = NONE) {
    return Event{type, player, target, a, b};
}

inline Decision player_decision(uint8_t player, Phase phase) {
    Decision d{};
    d.kind = DecisionKind::PLAYER;
    d.player = player;
    d.phase = phase;
    return d;
}

// Sets n_legal from the mask; call last when building a PLAYER decision.
inline Decision& finish(Decision& d) {
    d.n_legal = static_cast<uint16_t>(count_legal(d.mask, d.phase));
    return d;
}

[[noreturn]] inline void unimplemented(const char* what, int value) {
    std::fprintf(stderr, "catan: %s %d not implemented\n", what, value);
    std::abort();
}

inline EdgeSet all_roads(const PublicState& pub) {
    EdgeSet all{};
    for (const EdgeSet& r : pub.roads) {
        all[0] |= r[0];
        all[1] |= r[1];
    }
    return all;
}

// Vertices free under the distance rule: unoccupied with no occupied neighbour.
inline VertexSet distance_legal(const PublicState& pub) {
    const Topology& topo = topology();
    VertexSet occupied = 0, blocked = 0;
    for (int v = 0; v < NUM_VERTICES; ++v) {
        if (!vertex_occupied(pub.vertex[v])) continue;
        occupied |= vertex_bit(v);
        blocked |= topo.vertex_neighbor_mask[v];
    }
    return ALL_VERTICES & ~occupied & ~blocked;
}

inline bool can_afford(const GameState& s, uint8_t p, const std::array<uint8_t, NUM_RES>& cost) {
    for (int r = 0; r < NUM_RES; ++r) {
        if (s.hid.hands[p][r] < cost[r]) return false;
    }
    return true;
}

// Moves `n` of resource r between a hand and the bank, keeping hand_size in step.
inline void bank_to_hand(GameState& s, uint8_t p, int r, uint8_t n) {
    CATAN_CHECK(s.pub.bank[r] >= n);
    s.pub.bank[r] = static_cast<uint8_t>(s.pub.bank[r] - n);
    s.hid.hands[p][r] = static_cast<uint8_t>(s.hid.hands[p][r] + n);
    s.pub.hand_size[p] = static_cast<uint8_t>(s.pub.hand_size[p] + n);
}
inline void hand_to_bank(GameState& s, uint8_t p, int r, uint8_t n) {
    CATAN_CHECK(s.hid.hands[p][r] >= n);
    s.hid.hands[p][r] = static_cast<uint8_t>(s.hid.hands[p][r] - n);
    s.pub.hand_size[p] = static_cast<uint8_t>(s.pub.hand_size[p] - n);
    s.pub.bank[r] = static_cast<uint8_t>(s.pub.bank[r] + n);
}

// Pays a build cost to the bank with one SPEND event per resource paid.
inline void pay(GameState& s, uint8_t p, const std::array<uint8_t, NUM_RES>& cost,
                EventList& events) {
    for (int r = 0; r < NUM_RES; ++r) {
        if (cost[r] == 0) continue;
        hand_to_bank(s, p, r, cost[r]);
        events.push(event(EventType::SPEND, p, static_cast<uint8_t>(r), cost[r]));
    }
}

// Phase where play resumes after the robber or a development card's sub-phase: ROLL if the
// card was played before rolling (dice is 0 until the roll), otherwise MAIN.
inline Phase resume_phase(const PublicState& pub) {
    return pub.dice == 0 ? Phase::ROLL : Phase::MAIN;
}

// setup.cpp
Decision setup_decision(const PublicState& pub);
void apply_setup(const Board& board, GameState& s, Action action, EventList& events);

// turn.cpp
// Edges p may build a road on (cost aside): free edges touching p's building, or touching an
// unoccupied vertex that one of p's roads reaches.
EdgeSet buildable_roads(const PublicState& pub, uint8_t p);
// Bank trade rate per resource for p: 4:1, 3:1 with a generic harbor, 2:1 with that resource's.
std::array<uint8_t, NUM_RES> exchange_rates(const Board& board, const PublicState& pub,
                                            uint8_t p);
Decision roll_decision(const GameState& s);
void apply_roll(GameState& s, Action action, EventList& events);     // sets pending DICE
ChanceDist dice_dist();
void apply_dice(const Board& board, GameState& s, uint8_t outcome, EventList& events);
Decision main_decision(const Board& board, const GameState& s);
void apply_main(const Board& board, GameState& s, Action action, EventList& events);
Decision exchange_recv_decision(const Board& board, const GameState& s);
void apply_exchange_recv(const Board& board, GameState& s, Action action, EventList& events);
// Ends the game if the current player has reached VICTORY_POINTS_TO_WIN (public VP plus held
// VP cards). Called by the dispatcher after every apply outside setup.
void check_victory(GameState& s, EventList& events);

// robber.cpp
void start_robber(GameState& s);  // after a 7: discards owed, else MOVE_ROBBER
Decision discard_decision(const GameState& s);
void apply_discard(GameState& s, Action action, EventList& events);
Decision move_robber_decision(const GameState& s);
void apply_move_robber(const Board& board, GameState& s, Action action, EventList& events);
Decision steal_decision(const Board& board, const GameState& s);
void apply_steal(GameState& s, Action action, EventList& events);  // sets pending STEAL
ChanceDist steal_dist(const GameState& s);
void apply_steal_outcome(GameState& s, uint8_t outcome, EventList& events);

// devcards.cpp
// Cards the current player may play now, as a bitmask over DevCard: at most one card per turn,
// never VICTORY_POINT, never a card bought this turn, and only if the card can take effect.
uint8_t playable_dev_cards(const GameState& s);
void apply_buy_dev(GameState& s, EventList& events);  // pays, sets pending DEV_DRAW
ChanceDist dev_draw_dist(const GameState& s);
void apply_dev_draw(GameState& s, uint8_t outcome, EventList& events);
void apply_play_dev(GameState& s, uint8_t card, EventList& events);
Decision road_building_decision(const GameState& s);
void apply_road_building(GameState& s, Action action, EventList& events);
Decision yop_decision(const GameState& s);
void apply_yop(GameState& s, Action action, EventList& events);
Decision monopoly_decision(const GameState& s);
void apply_monopoly(GameState& s, Action action, EventList& events);

// awards.cpp
// Longest simple road (each edge used once) of player p. A path may end at, but not pass
// through, a vertex holding another player's building.
uint8_t road_length(const PublicState& pub, uint8_t p);
// Recomputes every road_len and moves the Longest Road card (and its 2 VP) per the rules.
// Call after any road or settlement is placed.
void update_longest_road(GameState& s, EventList& events);
// Moves the Largest Army card (and its 2 VP) after a Knight is played.
void update_largest_army(GameState& s, EventList& events);

}  // namespace catan::detail
