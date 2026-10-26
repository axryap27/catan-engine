// Times random-play games end to end. The checksum (xor of final state hashes) changes only
// when play changes, so runs are comparable.
//   catan_bench [games] [first_seed]     defaults 2000 and 1
#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "catan/board.h"
#include "catan/chance.h"
#include "catan/rng.h"
#include "catan/rules.h"
#include "catan/state.h"

using namespace std;
using namespace catan;

int main(int argc, char** argv) {
    int games = 2000;
    uint64_t first_seed = 1;

    if (argc > 1) games = atoi(argv[1]);
    if (argc > 2) first_seed = strtoull(argv[2], nullptr, 10);

    // First pass: time board generation only.
    auto board_start = chrono::steady_clock::now();
    uint64_t board_sum = 0;  // keeps the boards observable
    for (int i = 0; i < games; ++i) board_sum += random_board(first_seed + i).token[0];
    auto board_end = chrono::steady_clock::now();
    auto board_micros =
        chrono::duration_cast<chrono::microseconds>(board_end - board_start).count();
    double board_per = static_cast<double>(board_micros) / games;

    // Main pass: play all games.
    uint64_t checksum = 0;
    uint64_t total_decisions = 0;
    uint64_t total_turns = 0;
    int games_capped = 0;

    auto play_start = chrono::steady_clock::now();
    for (int g = 0; g < games; ++g) {
        const uint64_t seed = first_seed + g;
        const Board board = random_board(seed);
        GameState s = initial_state(board);
        uint64_t moves = 0;

        for (Decision d = next_decision(board, s); d.kind != DecisionKind::TERMINAL;
             d = next_decision(board, s)) {
            if (d.kind == DecisionKind::CHANCE) {
                apply_chance(board, s, sample_chance(d.chance, s, seed));
            } else {
                const auto n = random_below(seed, 1000, moves++, d.n_legal);
                apply(board, s, nth_legal(d.mask, d.phase, static_cast<int>(n)));
            }
        }

        // Accumulate state hash and check for turn cap.
        checksum ^= state_hash(s);
        total_decisions += s.pub.ply;  // player and chance decisions
        total_turns += s.pub.turn;
        if (s.pub.turn >= TURN_CAP) {
            games_capped++;
        }
    }
    auto play_end = chrono::steady_clock::now();

    auto total_micros = chrono::duration_cast<chrono::microseconds>(play_end - play_start).count();
    auto total_nanos = chrono::duration_cast<chrono::nanoseconds>(play_end - play_start).count();

    double total_secs = static_cast<double>(total_micros) / 1e6;
    double us_per_game = static_cast<double>(total_micros) / games;
    double ns_per_decision = static_cast<double>(total_nanos) / total_decisions;

    printf("games %d, decisions %llu, turns %llu\n",
           games,
           static_cast<unsigned long long>(total_decisions),
           static_cast<unsigned long long>(total_turns));
    printf("total %.3f s, %.0f us/game, %.0f ns/decision\n",
           total_secs, us_per_game, ns_per_decision);
    printf("board generation %.1f us/board (sum %llu)\n", board_per,
           static_cast<unsigned long long>(board_sum));
    printf("checksum 0x%016llx\n", static_cast<unsigned long long>(checksum));
    printf("games capped by turn limit: %d\n", games_capped);

    return 0;
}
