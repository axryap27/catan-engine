// Game loop and paired-seed matches with seat rotation.
#include "catan/match.h"

#include <cmath>

#include "catan/board.h"
#include "catan/chance.h"
#include "catan/check.h"
#include "catan/rng.h"
#include "catan/rules.h"

using namespace std;

namespace catan {

namespace {

// Wilson score interval bound; sign = -1 for the lower end, +1 for the upper.
double wilson(int wins, int games, int sign) {
    if (games == 0) return 0;
    const double z = 1.96;
    const double n = games;
    const double p = wins / n;
    const double z2 = z * z;
    const double center = p + z2 / (2 * n);
    const double margin = z * sqrt(p * (1 - p) / n + z2 / (4 * n * n));
    return (center + sign * margin) / (1 + z2 / n);
}

}  // namespace

double EntrantStats::ci_low() const { return wilson(wins, games, -1); }
double EntrantStats::ci_high() const { return wilson(wins, games, +1); }

GameResult play_game(uint64_t seed, const array<Policy*, NUM_PLAYERS>& seats, bool standard) {
    const Board board = standard ? standard_board() : random_board(seed);
    GameState s = initial_state(board);
    GameResult r;
    r.seed = seed;
    int decisions = 0;

    for (Decision d = next_decision(board, s); d.kind != DecisionKind::TERMINAL;
         d = next_decision(board, s)) {
        EventList events;
        if (d.kind == DecisionKind::CHANCE) {
            events = apply_chance(board, s, sample_chance(d.chance, s, seed));
        } else {
            const Action a = seats[d.player]->choose(board, s, d,
                                                     random_u64(seed, 100 + d.player, s.pub.ply));
            CATAN_CHECK(is_legal(d.mask, d.phase, a));
            events = apply(board, s, a);
        }
        ++decisions;
        for (const Event& e : events) {
            if (e.type == EventType::GAME_OVER) r.winner = e.player;
        }
    }

    for (int p = 0; p < NUM_PLAYERS; ++p) {
        r.vp[p] = static_cast<uint8_t>(s.pub.public_vp[p] +
                                       s.hid.dev_held[p][static_cast<int>(DevCard::VICTORY_POINT)]);
    }
    r.turns = s.pub.turn;
    r.decisions = static_cast<uint16_t>(decisions);
    return r;
}

MatchResult run_match(const array<Policy*, NUM_PLAYERS>& lineup, int n_seeds,
                      uint64_t first_seed) {
    MatchResult m;
    array<long, NUM_PLAYERS> vp_sum{};
    for (int i = 0; i < NUM_PLAYERS; ++i) m.entrant[i].name = lineup[i]->name();
    m.results.reserve(static_cast<size_t>(n_seeds) * NUM_PLAYERS);

    for (int n = 0; n < n_seeds; ++n) {
        const uint64_t seed = first_seed + n;
        for (int k = 0; k < NUM_PLAYERS; ++k) {
            array<Policy*, NUM_PLAYERS> seats{};
            for (int i = 0; i < NUM_PLAYERS; ++i) seats[(i + k) % NUM_PLAYERS] = lineup[i];
            const GameResult r = play_game(seed, seats);
            ++m.games;
            if (r.winner == NONE) ++m.capped;
            for (int i = 0; i < NUM_PLAYERS; ++i) {
                const int seat = (i + k) % NUM_PLAYERS;
                ++m.entrant[i].games;
                vp_sum[i] += r.vp[seat];
                if (r.winner == seat) ++m.entrant[i].wins;
            }
            m.results.push_back(r);
        }
    }
    for (int i = 0; i < NUM_PLAYERS; ++i) {
        EntrantStats& e = m.entrant[i];
        if (e.games) e.mean_vp = static_cast<double>(vp_sum[i]) / e.games;
    }
    return m;
}

}  // namespace catan
