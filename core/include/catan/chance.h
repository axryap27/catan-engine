#pragma once
#include <cstdint>

#include "catan/decision.h"
#include "catan/state.h"

// Drivers resolve CHANCE decisions from the game seed, never from state. Each kind has its own
// stream and a counter that does not depend on the players' choices where possible:
//   DICE:     index = turn, so two policies on the same seed see the same dice
//   DEV_DRAW: index = cards drawn so far (DEV_DECK_SIZE - dev_deck_remaining)
//   STEAL:    index = ply

namespace catan {

// Weighted pick from `dist` using a uniform 64-bit value.
uint8_t pick_outcome(const ChanceDist& dist, uint64_t r);

// The outcome for `dist` in `state`, fully determined by `seed`.
uint8_t sample_chance(const ChanceDist& dist, const GameState& state, uint64_t seed);

}  // namespace catan
