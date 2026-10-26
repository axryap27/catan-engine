#pragma once
#include <array>
#include <cstdint>
#include <vector>

#include "catan/policy.h"

// Matches with paired seeds and full seat rotation: every lineup is played on the same boards
// and dice (dice are keyed by turn, boards by seed) with each policy in every seat.

namespace catan {

struct GameResult {
    uint64_t seed = 0;
    uint8_t winner = NONE;            // seat, NONE if the turn cap ended the game
    std::array<uint8_t, NUM_PLAYERS> vp{};  // final VP incl. hidden VP cards
    uint16_t turns = 0;
    uint16_t decisions = 0;
};

// Plays one game on random_board(seed) (or standard_board() if `standard`). Chance outcomes
// come from sample_chance(..., seed); seat s's policy gets rng = random_u64(seed, 100 + s, ply).
GameResult play_game(uint64_t seed, const std::array<Policy*, NUM_PLAYERS>& seats,
                     bool standard = false);

// Per-entrant totals over a match. An entrant is one slot of the lineup (a policy may be
// entered in several slots; stats are per slot).
struct EntrantStats {
    const char* name = "";
    int games = 0;
    int wins = 0;
    double mean_vp = 0;
    double win_rate() const { return games ? static_cast<double>(wins) / games : 0; }
    // 95% Wilson score interval for the win rate.
    double ci_low() const;
    double ci_high() const;
};

struct MatchResult {
    std::array<EntrantStats, NUM_PLAYERS> entrant;
    int games = 0;
    int capped = 0;  // games ended by the turn cap
    std::vector<GameResult> results;  // in play order (seed-major, then rotation)
};

// For each seed in [first_seed, first_seed + n_seeds), plays the lineup in all 4 cyclic
// rotations (entrant i sits in seat (i + k) % 4 in rotation k), so 4 * n_seeds games.
MatchResult run_match(const std::array<Policy*, NUM_PLAYERS>& lineup, int n_seeds,
                      uint64_t first_seed = 1);

}  // namespace catan
