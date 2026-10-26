#pragma once

#include "catan/action.h"
#include "catan/board.h"
#include "catan/state.h"

// An independent, deliberately slow statement of the rules' legality: for a state awaiting a
// PLAYER decision, is `action` legal? Written from the rulebook against raw state, sharing no
// code with core/src (no masks, no rules_internal.h helpers), so the fuzzer can cross-check
// next_decision's masks against it on every decision of random games.
//
// Precondition: state.pub.pending_chance == 0 and phase != TERMINAL.

namespace catan::test {

bool reference_is_legal(const Board& board, const GameState& state, Action action);

// Longest road of player p by exhaustive search over every simple edge-trail (no edge reused),
// honoring the rule that an opponent's building blocks passing through a vertex. Exponential;
// for cross-checking road_length on small and random networks.
int reference_road_length(const PublicState& pub, uint8_t p);

}  // namespace catan::test
