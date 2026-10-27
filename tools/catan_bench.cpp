// Benchmarks the engine by playing whole games.
//   catan_bench [games] [first_seed] [--policy random|greedy] [--threads N] [--sweep] [--latency]
//
// Defaults: 2000 games from seed 1, random play, one thread. The checksum (xor of final state
// hashes) changes only when play changes, and does not depend on the thread count.
//   --policy greedy  all four seats play the greedy baseline: realistic ~70-turn games
//   --threads N      play on N threads (0 = all hardware threads); games share nothing
//   --sweep          repeat the run at 1, 2, 4, ... threads and the hardware count, and print
//                    throughput and scaling
//   --latency        time every decision (next_decision + choice + apply) and print
//                    percentiles. Timer cost is subtracted from samples but not from totals.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

#include "catan/board.h"
#include "catan/chance.h"
#include "catan/policy.h"
#include "catan/rng.h"
#include "catan/rules.h"
#include "catan/state.h"

using namespace std;
using namespace catan;

namespace {

using Clock = chrono::steady_clock;

int64_t nanos_between(Clock::time_point a, Clock::time_point b) {
    return chrono::duration_cast<chrono::nanoseconds>(b - a).count();
}

// Exact latency distribution: 1 ns bins below kBins, raw values above.
class Histogram {
public:
    static constexpr int64_t kBins = 1 << 14;

    Histogram() : bins_(kBins, 0) {}

    void add(int64_t ns) {
        ns = std::max<int64_t>(ns, 0);
        ++count_;
        sum_ += ns;
        max_ = std::max(max_, ns);
        if (ns < kBins) ++bins_[ns];
        else overflow_.push_back(ns);
    }

    void merge(const Histogram& o) {
        count_ += o.count_;
        sum_ += o.sum_;
        max_ = std::max(max_, o.max_);
        for (int64_t i = 0; i < kBins; ++i) bins_[i] += o.bins_[i];
        overflow_.insert(overflow_.end(), o.overflow_.begin(), o.overflow_.end());
    }

    uint64_t count() const { return count_; }
    double mean() const { return count_ ? static_cast<double>(sum_) / count_ : 0.0; }
    int64_t max() const { return max_; }

    // Smallest sample v with at least a fraction q of all samples <= v.
    int64_t percentile(double q) {
        if (count_ == 0) return 0;
        const uint64_t rank = std::max<uint64_t>(1, static_cast<uint64_t>(ceil(q * count_)));
        uint64_t seen = 0;
        for (int64_t v = 0; v < kBins; ++v) {
            seen += bins_[v];
            if (seen >= rank) return v;
        }
        sort(overflow_.begin(), overflow_.end());
        return overflow_[rank - seen - 1];
    }

private:
    vector<uint64_t> bins_;
    vector<int64_t> overflow_;
    uint64_t count_ = 0;
    int64_t sum_ = 0;
    int64_t max_ = 0;
};

struct ClockInfo {
    int64_t overhead;    // median cost of one back-to-back pair of clock reads
    int64_t resolution;  // smallest observable tick
};

ClockInfo calibrate_clock() {
    vector<int64_t> pairs(1'000'000);
    for (auto& p : pairs) {
        const auto a = Clock::now();
        const auto b = Clock::now();
        p = nanos_between(a, b);
    }
    nth_element(pairs.begin(), pairs.begin() + pairs.size() / 2, pairs.end());

    int64_t resolution = INT64_MAX;
    for (int i = 0; i < 1000; ++i) {
        const auto a = Clock::now();
        auto b = Clock::now();
        while (b == a) b = Clock::now();
        resolution = std::min(resolution, nanos_between(a, b));
    }
    return {pairs[pairs.size() / 2], resolution};
}

struct Options {
    int games = 2000;
    uint64_t first_seed = 1;
    bool greedy = false;
    int threads = 1;
    bool sweep = false;
    bool latency = false;
};

struct Run {
    uint64_t checksum = 0;
    uint64_t decisions = 0;  // player and chance
    uint64_t player_decisions = 0;
    uint64_t turns = 0;
    int capped = 0;
    vector<int64_t> game_ns;
    Histogram player, chance;  // filled only with --latency

    void merge(const Run& o) {
        checksum ^= o.checksum;
        decisions += o.decisions;
        player_decisions += o.player_decisions;
        turns += o.turns;
        capped += o.capped;
        game_ns.insert(game_ns.end(), o.game_ns.begin(), o.game_ns.end());
        player.merge(o.player);
        chance.merge(o.chance);
    }
};

// seats == nullptr plays the bench's own random policy, kept so checksums stay comparable
// with earlier versions of this tool.
template <bool kTimed>
void play_one(uint64_t seed, Policy* const* seats, int64_t overhead, Run& r) {
    const Board board = random_board(seed);
    GameState s = initial_state(board);
    uint64_t moves = 0;

    const auto game_start = Clock::now();
    for (;;) {
        Clock::time_point t0;
        if constexpr (kTimed) t0 = Clock::now();

        const Decision d = next_decision(board, s);
        if (d.kind == DecisionKind::TERMINAL) break;
        const bool chance = d.kind == DecisionKind::CHANCE;
        if (chance) {
            apply_chance(board, s, sample_chance(d.chance, s, seed));
        } else if (seats) {
            apply(board, s,
                  seats[d.player]->choose(board, s, d,
                                          random_u64(seed, 100 + d.player, s.pub.ply)));
        } else {
            const auto n = random_below(seed, 1000, moves++, d.n_legal);
            apply(board, s, nth_legal(d.mask, d.phase, static_cast<int>(n)));
        }

        if constexpr (kTimed) (chance ? r.chance : r.player).add(
            nanos_between(t0, Clock::now()) - overhead);
        if (!chance) ++r.player_decisions;
    }
    r.game_ns.push_back(nanos_between(game_start, Clock::now()));

    r.checksum ^= state_hash(s);
    r.decisions += s.pub.ply;
    r.turns += s.pub.turn;
    if (s.pub.turn >= TURN_CAP) ++r.capped;
}

// Plays all o.games on `threads` threads, which take games from a shared counter. Returns the
// merged result and the wall time in seconds.
pair<Run, double> run_games(const Options& o, int threads, int64_t overhead) {
    vector<Run> runs(threads);
    atomic<int> next{0};

    auto worker = [&](int t) {
        vector<unique_ptr<Policy>> owned;
        Policy* seats[NUM_PLAYERS] = {};
        if (o.greedy) {
            for (int p = 0; p < NUM_PLAYERS; ++p) {
                owned.push_back(make_greedy_policy());
                seats[p] = owned.back().get();
            }
        }
        Run& r = runs[t];
        r.game_ns.reserve(o.games / threads + 1);
        for (int g; (g = next.fetch_add(1, memory_order_relaxed)) < o.games;) {
            const uint64_t seed = o.first_seed + static_cast<uint64_t>(g);
            if (o.latency) play_one<true>(seed, o.greedy ? seats : nullptr, overhead, r);
            else play_one<false>(seed, o.greedy ? seats : nullptr, overhead, r);
        }
    };

    const auto start = Clock::now();
    if (threads == 1) {
        worker(0);
    } else {
        vector<thread> pool;
        for (int t = 0; t < threads; ++t) pool.emplace_back(worker, t);
        for (auto& th : pool) th.join();
    }
    const double wall = static_cast<double>(nanos_between(start, Clock::now())) / 1e9;

    Run total;
    for (const auto& r : runs) total.merge(r);
    return {std::move(total), wall};
}

int64_t game_percentile(vector<int64_t>& v, double q) {
    sort(v.begin(), v.end());
    const size_t rank = std::max<size_t>(1, static_cast<size_t>(ceil(q * v.size())));
    return v[rank - 1];
}

void print_latency_row(const char* label, Histogram& h) {
    printf("  %-7s %12llu %7.0f %6lld %6lld %6lld %7lld %9lld\n", label,
           static_cast<unsigned long long>(h.count()), h.mean(),
           static_cast<long long>(h.percentile(0.50)), static_cast<long long>(h.percentile(0.90)),
           static_cast<long long>(h.percentile(0.99)), static_cast<long long>(h.percentile(0.999)),
           static_cast<long long>(h.max()));
}

int usage() {
    fprintf(stderr,
            "usage: catan_bench [games] [first_seed] [--policy random|greedy] [--threads N] "
            "[--sweep] [--latency]\n");
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    Options o;
    int positional = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--policy") && i + 1 < argc) {
            const char* p = argv[++i];
            if (!strcmp(p, "greedy")) o.greedy = true;
            else if (strcmp(p, "random")) return usage();
        } else if (!strcmp(argv[i], "--threads") && i + 1 < argc) {
            o.threads = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--sweep")) {
            o.sweep = true;
        } else if (!strcmp(argv[i], "--latency")) {
            o.latency = true;
        } else if (argv[i][0] != '-' && positional == 0) {
            o.games = atoi(argv[i]);
            ++positional;
        } else if (argv[i][0] != '-' && positional == 1) {
            o.first_seed = strtoull(argv[i], nullptr, 10);
            ++positional;
        } else {
            return usage();
        }
    }
    const int hw = std::max(1, static_cast<int>(thread::hardware_concurrency()));
    if (o.threads <= 0) o.threads = hw;
    if (o.games <= 0 || (o.sweep && o.latency)) return usage();

    const char* policy = o.greedy ? "greedy" : "random";

    if (o.sweep) {
        vector<int> counts;
        for (int t = 1; t < hw; t *= 2) counts.push_back(t);
        counts.push_back(hw);

        printf("policy %s, %d games from seed %llu, %d hardware threads\n", policy, o.games,
               static_cast<unsigned long long>(o.first_seed), hw);
        printf("  threads   wall s   M decisions/s   games/s   speedup   efficiency\n");
        double base = 0;
        uint64_t checksum = 0;
        bool same = true;
        for (const int t : counts) {
            auto [r, wall] = run_games(o, t, 0);
            const double rate = static_cast<double>(r.decisions) / wall;
            if (t == 1) base = rate, checksum = r.checksum;
            same = same && r.checksum == checksum;
            printf("  %7d %8.3f %15.2f %9.0f %8.2fx %11.0f%%\n", t, wall, rate / 1e6,
                   o.games / wall, rate / base, 100.0 * rate / base / t);
        }
        printf("checksum 0x%016llx %s\n", static_cast<unsigned long long>(checksum),
               same ? "(identical at every thread count)" : "MISMATCH across thread counts");
        return same ? 0 : 1;
    }

    // Board generation alone, single thread.
    const auto board_start = Clock::now();
    uint64_t board_sum = 0;  // keeps the boards observable
    for (int i = 0; i < o.games; ++i) board_sum += random_board(o.first_seed + i).token[0];
    const double board_us = static_cast<double>(nanos_between(board_start, Clock::now())) /
                            1e3 / o.games;

    ClockInfo clock{0, 0};
    if (o.latency) clock = calibrate_clock();

    auto [r, wall] = run_games(o, o.threads, clock.overhead);

    printf("policy %s, %d thread%s, %d games from seed %llu\n", policy, o.threads,
           o.threads == 1 ? "" : "s", o.games, static_cast<unsigned long long>(o.first_seed));
    printf("decisions %llu (player %llu), turns %llu, %.1f turns/game, capped %d\n",
           static_cast<unsigned long long>(r.decisions),
           static_cast<unsigned long long>(r.player_decisions),
           static_cast<unsigned long long>(r.turns), static_cast<double>(r.turns) / o.games,
           r.capped);
    printf("wall %.3f s, %.2f M decisions/s, %.0f games/s, %.0f ns/decision per thread\n", wall,
           static_cast<double>(r.decisions) / wall / 1e6, o.games / wall,
           wall * o.threads * 1e9 / static_cast<double>(r.decisions));
    printf("game time us: p50 %.0f, p99 %.0f, max %.0f\n",
           static_cast<double>(game_percentile(r.game_ns, 0.50)) / 1e3,
           static_cast<double>(game_percentile(r.game_ns, 0.99)) / 1e3,
           static_cast<double>(r.game_ns.back()) / 1e3);
    printf("board generation %.1f us/board (sum %llu)\n", board_us,
           static_cast<unsigned long long>(board_sum));

    if (o.latency) {
        printf("latency per decision, ns (timer overhead %lld ns subtracted, clock "
               "resolution %lld ns; totals above include timer cost)\n",
               static_cast<long long>(clock.overhead), static_cast<long long>(clock.resolution));
        printf("  %-7s %12s %7s %6s %6s %6s %7s %9s\n", "", "count", "mean", "p50", "p90",
               "p99", "p99.9", "max");
        Histogram all;
        all.merge(r.player);
        all.merge(r.chance);
        print_latency_row("player", r.player);
        print_latency_row("chance", r.chance);
        print_latency_row("all", all);
    }

    printf("checksum 0x%016llx\n", static_cast<unsigned long long>(r.checksum));
    return 0;
}
