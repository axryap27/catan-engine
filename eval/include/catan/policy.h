#pragma once
#include <cstdint>
#include <memory>

#include "catan/action.h"
#include "catan/board.h"
#include "catan/decision.h"
#include "catan/state.h"

// A player strategy for evaluation matches and baselines.
//
// Policies receive the full GameState for convenience but may read only what their seat can
// know: the public state, and their own hidden data (hid.hands[seat], hid.dev_held[seat], and
// hid.dev_bought_this_turn while they are the current player). Never another hand, the deck
// or the steal outcome. test_eval checks this by resampling opponents' hidden cards and
// requiring the same choice.

namespace catan {

class Policy {
public:
    virtual ~Policy() = default;
    virtual const char* name() const = 0;
    // Called for every PLAYER decision with d.player == the seat this policy plays. Must return
    // an action legal in d.mask. `rng` is a fresh uniform value the policy may use for ties
    // and randomization, so choices are deterministic given the game seed.
    virtual Action choose(const Board& board, const GameState& state, const Decision& d,
                          uint64_t rng) = 0;
};

// Uniform over legal actions (nth_legal with rng % n_legal).
std::unique_ptr<Policy> make_random_policy();

// Hand-written heuristic baseline; see greedy.cpp for its rules.
std::unique_ptr<Policy> make_greedy_policy();

}  // namespace catan
