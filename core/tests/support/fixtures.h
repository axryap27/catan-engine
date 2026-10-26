#pragma once
#include <cstdint>
#include <functional>
#include <vector>

#include "catan/action.h"
#include "catan/board.h"
#include "catan/decision.h"
#include "catan/event.h"
#include "catan/state.h"

// Shared helpers for rules tests. Positions are built two ways:
//   - by playing: random_setup / play_random drive the real rules;
//   - by editing: give / place_* write state directly, keeping bank, pieces and VP consistent,
//     so invariant_violation() still passes. Edits never update road_len or award holders;
//     call refresh_awards() after editing roads or buildings.

namespace catan::test {

// Every syntactically possible action: 13 types x args 0..71 (the largest head).
const std::vector<Action>& all_actions();

// Plays the 16 setup steps with uniform random legal moves; leaves the state at ROLL, turn 0.
GameState random_setup(const Board& board, uint64_t seed);

// Called after every transition of play_random: the decision just answered and its events.
using StepHook = std::function<void(const GameState& before, const Decision& d,
                                    const GameState& after, const EventList& events)>;

// Plays from `s` with uniform random legal moves and seeded chance until TERMINAL or until
// `max_decisions` more decisions. Returns the number of decisions taken.
int play_random(const Board& board, GameState& s, uint64_t seed, int max_decisions = 1 << 30,
                const StepHook& hook = {});

// Puts the game in MAIN for player p after a roll of `dice` (default 8), with no pending
// chance or sub-phase state.
void set_main(GameState& s, uint8_t p, uint8_t dice = 8);

// Moves n cards of resource r from the bank into p's hand (n may be negative: hand -> bank).
void give(GameState& s, uint8_t p, Resource r, int n);
// Empties every hand back into the bank.
void clear_hands(GameState& s);
// Gives p a development card straight from the deck (not bought this turn).
void give_dev(GameState& s, uint8_t p, DevCard c, int n = 1);

// Direct placements; each takes a piece from p's supply and adjusts public_vp.
void place_settlement(GameState& s, uint8_t p, int v);
void place_city(GameState& s, uint8_t p, int v);  // upgrades p's settlement at v
void place_road(GameState& s, uint8_t p, int e);
// Removes every building and road from the board, returning pieces and VP (keeps hands).
void clear_board(GameState& s);
// Recomputes road_len and award holders as the rules would after a build, without events.
void refresh_awards(GameState& s);

// Edge joining vertices a and b; NONE if not adjacent.
uint8_t edge_between(int a, int b);

// All events of type t in `events`.
std::vector<Event> events_of(const EventList& events, EventType t);

}  // namespace catan::test
