#pragma once
#include <array>
#include <cstdint>

#include "catan/action.h"
#include "catan/board.h"
#include "catan/decision.h"
#include "catan/event.h"
#include "catan/state.h"

// The engine's whole interface to drivers (self-play, search, eval, humans):
//
//   for (Decision d = next_decision(b, s); d.kind != DecisionKind::TERMINAL;
//        d = next_decision(b, s)) {
//       if (d.kind == DecisionKind::CHANCE) apply_chance(b, s, sample_chance(d.chance, s, seed));
//       else                                apply(b, s, policy(s, d));
//   }

namespace catan {

inline constexpr int VICTORY_POINTS_TO_WIN = 10;
inline constexpr int DISCARD_THRESHOLD = 7;  // a 7 makes hands larger than this discard half
// The game ends with no winner once `turn` reaches this. Placeholder until scoring of capped
// games is decided (STATUS.md, open decisions).
inline constexpr uint16_t TURN_CAP = 1000;

// Build costs, indexed by Resource (brick, lumber, wool, grain, ore).
inline constexpr std::array<uint8_t, NUM_RES> ROAD_COST = {1, 1, 0, 0, 0};
inline constexpr std::array<uint8_t, NUM_RES> SETTLEMENT_COST = {1, 1, 1, 1, 0};
inline constexpr std::array<uint8_t, NUM_RES> CITY_COST = {0, 0, 0, 2, 3};
inline constexpr std::array<uint8_t, NUM_RES> DEV_CARD_COST = {0, 0, 1, 1, 1};
inline constexpr int SETUP_STEPS = 16;  // 8 placements (snake order), settlement then road each

// Seat placing at setup step 0..15: 0 1 2 3 3 2 1 0, two steps per placement.
constexpr uint8_t setup_player(uint8_t step) {
    const int placement = step / 2;
    const int seat = placement < NUM_PLAYERS ? placement : 2 * NUM_PLAYERS - 1 - placement;
    return static_cast<uint8_t>(seat);
}

// Who must decide what next, with legal moves or chance weights. Pure.
Decision next_decision(const Board& board, const GameState& state);

// Applies a legal PLAYER action. Precondition: is_legal(next_decision(...).mask, phase, action);
// checked only in the checked build.
EventList apply(const Board& board, GameState& state, Action action);

// Applies the outcome of a pending CHANCE decision. Precondition: weight[outcome] > 0.
EventList apply_chance(const Board& board, GameState& state, uint8_t outcome);

}  // namespace catan
