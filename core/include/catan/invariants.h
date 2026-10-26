#pragma once

#include "catan/board.h"
#include "catan/state.h"

namespace catan {

// Checks every rules invariant that must hold between decisions. Returns nullptr if `state` is
// consistent, else a short static description of the first violation. Pure; does not abort.
//
// The checked build calls it after every apply/apply_chance and aborts on a violation. Tests
// and fuzzers call it directly. Covered:
//   - resources: bank + all hands == 19 per type; hand_size == sum of hand
//   - dev cards: deck counts sum to dev_deck_remaining; dev_count == sum of dev_held;
//     per type, deck + held <= deck composition; knights: deck + held + played == 14
//   - pieces: pieces_left + pieces on the board == 15 / 5 / 4 for every player
//   - board: distance rule (no two adjacent occupied vertices); roads disjoint between players
//   - VP: public_vp == settlements + 2 * cities + 2 per award held
//   - awards: road_len matches a fresh road-length computation; the Longest Road holder has
//     length >= 5 and no other player is strictly longer; the Largest Army holder has >= 3
//     knights and no other player has strictly more
//   - phase bookkeeping: pending_chance in range, steal_victim set iff a STEAL chance is
//     pending, pending_discard nonzero only in DISCARD, free_roads only in ROAD_BUILDING,
//     yop_remaining only in YOP_PICK, exchange_give set iff EXCHANGE_RECV, dice == 0 in SETUP
//     and before the roll, dev_bought_this_turn <= current player's dev_held
//   - liveness: if the next decision is a PLAYER decision, it has at least one legal action
const char* invariant_violation(const Board& board, const GameState& state);

}  // namespace catan
