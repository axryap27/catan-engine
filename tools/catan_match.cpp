// Plays policy lineups on paired seeds with full seat rotation and prints win rates.
//   catan_match <a> <b> [seeds=250] [first_seed=1]     a, b in {random, greedy}
// Runs "1 a vs 3 b" and the fair 2v2 lineup a,b,a,b. Each seed is played in all 4 rotations.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

#include "catan/match.h"

using namespace std;
using namespace catan;

namespace {

unique_ptr<Policy> make(const char* name) {
    if (strcmp(name, "random") == 0) return make_random_policy();
    if (strcmp(name, "greedy") == 0) return make_greedy_policy();
    return nullptr;
}

void run(const char* title, const array<Policy*, NUM_PLAYERS>& lineup, int seeds,
         uint64_t first_seed) {
    const auto t0 = chrono::steady_clock::now();
    const MatchResult m = run_match(lineup, seeds, first_seed);
    const double secs = chrono::duration<double>(chrono::steady_clock::now() - t0).count();

    printf("%s, %d seeds x 4 rotations = %d games (%d capped), %.1f s\n", title, seeds, m.games,
           m.capped, secs);
    printf("  %-9s  %-7s  %5s  %-14s  %s\n", "seat-slot", "policy", "win%", "95% CI", "mean VP");
    for (int i = 0; i < NUM_PLAYERS; ++i) {
        const EntrantStats& e = m.entrant[i];
        char ci[32];
        snprintf(ci, sizeof ci, "[%.1f, %.1f]", 100 * e.ci_low(), 100 * e.ci_high());
        printf("  %-9d  %-7s  %5.1f  %-14s  %.1f\n", i, e.name, 100 * e.win_rate(), ci,
               e.mean_vp);
    }
    printf("  break-even 25.0\n\n");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr,
                "usage: catan_match <random|greedy> <random|greedy> [seeds] [first_seed]\n");
        return 2;
    }
    auto a = make(argv[1]);
    auto b = make(argv[2]);
    if (!a || !b) {
        fprintf(stderr, "unknown policy (use random or greedy)\n");
        return 2;
    }
    const int seeds = argc > 3 ? atoi(argv[3]) : 250;
    const uint64_t first_seed = argc > 4 ? strtoull(argv[4], nullptr, 10) : 1;
    if (seeds <= 0) {
        fprintf(stderr, "seeds must be positive\n");
        return 2;
    }

    char title[64];
    snprintf(title, sizeof title, "1 %s vs 3 %s", a->name(), b->name());
    run(title, {a.get(), b.get(), b.get(), b.get()}, seeds, first_seed);
    snprintf(title, sizeof title, "2 %s vs 2 %s (a,b,a,b)", a->name(), b->name());
    run(title, {a.get(), b.get(), a.get(), b.get()}, seeds, first_seed);
    return 0;
}
