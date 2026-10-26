#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "catan/action.h"
#include "catan/belief.h"
#include "catan/board.h"
#include "catan/chance.h"
#include "catan/decision.h"
#include "catan/event.h"
#include "catan/mask.h"
#include "catan/match.h"
#include "catan/policy.h"
#include "catan/rng.h"
#include "catan/rules.h"
#include "catan/state.h"

using namespace std;
using namespace catan;

namespace {

using Lineup = array<Policy*, NUM_PLAYERS>;

bool same_mask(const ActionMask& a, const ActionMask& b) {
    return a.type == b.type && a.edge == b.edge && a.settlement == b.settlement &&
           a.city == b.city && a.tile == b.tile && a.devcard == b.devcard &&
           a.player == b.player && a.resource == b.resource;
}

bool same_result(const GameResult& a, const GameResult& b) {
    return a.seed == b.seed && a.winner == b.winner && a.vp == b.vp && a.turns == b.turns &&
           a.decisions == b.decisions;
}

uint64_t policy_rng(uint64_t seed, const Decision& d, const GameState& s) {
    return random_u64(seed, 100 + d.player, s.pub.ply);
}

struct Counters {
    int games = 0;
    int won = 0;
    int illegal = 0;
};

// Plays one game with our own loop, checking every choice against the mask.
void play_checked(uint64_t seed, const Lineup& seats, Counters& c) {
    const Board board = random_board(seed);
    GameState s = initial_state(board);
    bool winner = false;
    for (Decision d = next_decision(board, s); d.kind != DecisionKind::TERMINAL;
         d = next_decision(board, s)) {
        if (d.kind == DecisionKind::CHANCE) {
            for (const Event& e : apply_chance(board, s, sample_chance(d.chance, s, seed)))
                if (e.type == EventType::GAME_OVER) winner = true;
            continue;
        }
        const Action a = seats[d.player]->choose(board, s, d, policy_rng(seed, d, s));
        if (!is_legal(d.mask, d.phase, a)) {
            ++c.illegal;
            FAIL("illegal action from " << seats[d.player]->name() << " seed " << seed
                                        << " phase " << static_cast<int>(d.phase) << " type "
                                        << static_cast<int>(a.type) << " arg "
                                        << static_cast<int>(a.arg));
        }
        for (const Event& e : apply(board, s, a))
            if (e.type == EventType::GAME_OVER) winner = true;
    }
    ++c.games;
    if (winner) ++c.won;
}

}  // namespace

TEST_CASE("policies only return legal actions", "[eval]") {
    auto rnd = make_random_policy();
    auto gr = make_greedy_policy();
    REQUIRE(string(rnd->name()) == "random");
    REQUIRE(string(gr->name()) == "greedy");
    Counters c;
    SECTION("random x4") {
        for (uint64_t seed = 1; seed <= 100; ++seed)
            play_checked(seed, {rnd.get(), rnd.get(), rnd.get(), rnd.get()}, c);
        CHECK(c.games == 100);
        CHECK(c.won >= 70);  // random games are slow; some may hit the cap
    }
    SECTION("greedy x4") {
        for (uint64_t seed = 1; seed <= 100; ++seed)
            play_checked(seed, {gr.get(), gr.get(), gr.get(), gr.get()}, c);
        CHECK(c.games == 100);
        CHECK(c.won >= 95);
    }
    SECTION("mixed lineups") {
        for (uint64_t seed = 1; seed <= 100; ++seed) {
            const int k = static_cast<int>(seed % 4);
            Lineup l{rnd.get(), rnd.get(), rnd.get(), rnd.get()};
            l[k] = gr.get();
            if (seed % 3 == 0) l[(k + 2) % 4] = gr.get();
            play_checked(seed, l, c);
        }
        CHECK(c.games == 100);
        CHECK(c.won >= 90);
    }
}

TEST_CASE("play_game terminates, mostly with a winner", "[eval]") {
    auto rnd = make_random_policy();
    auto gr = make_greedy_policy();
    int won = 0;
    for (uint64_t seed = 1; seed <= 100; ++seed) {
        const Lineup l = seed % 2 ? Lineup{gr.get(), gr.get(), gr.get(), gr.get()}
                                  : Lineup{gr.get(), rnd.get(), rnd.get(), gr.get()};
        const GameResult r = play_game(seed, l);
        CHECK(r.decisions > 0);
        if (r.winner != NONE) {
            ++won;
            CHECK(r.vp[r.winner] >= 10);
        }
    }
    CHECK(won >= 95);
}

TEST_CASE("greedy never reads hidden information of other seats", "[eval]") {
    auto gr = make_greedy_policy();
    auto rnd = make_random_policy();
    const Lineup all_greedy{gr.get(), gr.get(), gr.get(), gr.get()};
    int sampled = 0;
    int mask_mismatch = 0;
    int choice_mismatch = 0;
    int ticks = 0;

    auto run = [&](uint64_t seed, const Lineup& seats, Policy& judge) {
        const Board board = random_board(seed);
        GameState s = initial_state(board);
        BeliefState belief = initial_belief();
        for (Decision d = next_decision(board, s); d.kind != DecisionKind::TERMINAL;
             d = next_decision(board, s)) {
            EventList ev;
            if (d.kind == DecisionKind::CHANCE) {
                ev = apply_chance(board, s, sample_chance(d.chance, s, seed));
            } else {
                const uint64_t rng = policy_rng(seed, d, s);
                const Action a = seats[d.player]->choose(board, s, d, rng);
                if (++ticks % 7 == 0) {
                    const Action base = judge.choose(board, s, d, rng);
                    for (uint64_t k = 1; k <= 3; ++k) {
                        const GameState r =
                            resample_hidden(board, s, belief, d.player, seed * 1000 + k);
                        const Decision dr = next_decision(board, r);
                        ++sampled;
                        const bool kind_ok =
                            dr.kind == DecisionKind::PLAYER && dr.player == d.player;
                        if (!kind_ok || !same_mask(dr.mask, d.mask) || dr.phase != d.phase) {
                            if (++mask_mismatch <= 3)
                                WARN("mask mismatch seed " << seed << " ply " << s.pub.ply
                                                           << " phase "
                                                           << static_cast<int>(d.phase));
                            continue;
                        }
                        const Action b = judge.choose(board, r, dr, rng);
                        if (b.type != base.type || b.arg != base.arg) {
                            if (++choice_mismatch <= 5)
                                INFO("seed " << seed << " ply " << s.pub.ply << " seat "
                                             << int(d.player) << " phase "
                                             << static_cast<int>(d.phase) << " base "
                                             << static_cast<int>(base.type) << "/"
                                             << int(base.arg) << " resampled "
                                             << static_cast<int>(b.type) << "/" << int(b.arg));
                            CHECK(false);
                        }
                    }
                }
                ev = apply(board, s, a);
            }
            observe(belief, s.pub, ev);
        }
    };

    for (uint64_t seed = 1; seed <= 20; ++seed) run(seed, all_greedy, *gr);
    // The random policy only looks at the mask, so it must pass trivially.
    const Lineup all_random{rnd.get(), rnd.get(), rnd.get(), rnd.get()};
    for (uint64_t seed = 1; seed <= 2; ++seed) run(seed, all_random, *rnd);

    INFO("sampled " << sampled);
    CHECK(sampled >= 300);
    CHECK(mask_mismatch == 0);
    CHECK(choice_mismatch == 0);
}

TEST_CASE("games and matches are deterministic", "[eval]") {
    auto rnd = make_random_policy();
    auto gr = make_greedy_policy();
    const Lineup l{gr.get(), rnd.get(), gr.get(), rnd.get()};
    for (uint64_t seed = 1; seed <= 5; ++seed)
        CHECK(same_result(play_game(seed, l), play_game(seed, l)));
    CHECK(same_result(play_game(3, l, true), play_game(3, l, true)));

    const MatchResult a = run_match(l, 3, 10);
    const MatchResult b = run_match(l, 3, 10);
    REQUIRE(a.results.size() == b.results.size());
    for (size_t i = 0; i < a.results.size(); ++i)
        CHECK(same_result(a.results[i], b.results[i]));
    for (int i = 0; i < NUM_PLAYERS; ++i) {
        CHECK(a.entrant[i].wins == b.entrant[i].wins);
        CHECK(a.entrant[i].mean_vp == b.entrant[i].mean_vp);
    }
}

TEST_CASE("run_match rotates seats and tallies consistently", "[eval]") {
    auto rnd = make_random_policy();
    auto gr = make_greedy_policy();
    const Lineup l{gr.get(), rnd.get(), rnd.get(), rnd.get()};
    const int n_seeds = 5;
    const MatchResult m = run_match(l, n_seeds, 7);
    REQUIRE(m.games == 4 * n_seeds);
    REQUIRE(m.results.size() == static_cast<size_t>(m.games));
    array<int, NUM_PLAYERS> wins{};
    array<int, NUM_PLAYERS> entrant0_seat{};
    int capped = 0;
    for (int n = 0; n < n_seeds; ++n) {
        for (int k = 0; k < NUM_PLAYERS; ++k) {
            const GameResult& r = m.results[n * NUM_PLAYERS + k];
            CHECK(r.seed == 7u + n);
            ++entrant0_seat[k];  // rotation k puts entrant 0 in seat k
            // Replaying that rotation directly gives the same result.
            Lineup seats{};
            for (int i = 0; i < NUM_PLAYERS; ++i) seats[(i + k) % NUM_PLAYERS] = l[i];
            CHECK(same_result(r, play_game(r.seed, seats)));
            if (r.winner == NONE) {
                ++capped;
            } else {
                ++wins[(r.winner + NUM_PLAYERS - k) % NUM_PLAYERS];
            }
        }
    }
    for (int k = 0; k < NUM_PLAYERS; ++k) CHECK(entrant0_seat[k] == n_seeds);
    int total = 0;
    for (int i = 0; i < NUM_PLAYERS; ++i) {
        CHECK(m.entrant[i].games == 4 * n_seeds);
        CHECK(m.entrant[i].wins == wins[i]);
        total += m.entrant[i].wins;
        CHECK(m.entrant[i].mean_vp >= 0);
        CHECK(m.entrant[i].mean_vp <= 12);
    }
    CHECK(m.capped == capped);
    CHECK(total + m.capped == m.games);
}

TEST_CASE("Wilson interval", "[eval]") {
    auto interval = [](int wins, int games) {
        EntrantStats e;
        e.wins = wins;
        e.games = games;
        return e;
    };
    const EntrantStats half = interval(50, 100);
    CHECK(std::abs(half.ci_low() - 0.404) < 1e-3);
    CHECK(std::abs(half.ci_high() - 0.596) < 1e-3);
    const EntrantStats none = interval(0, 10);
    CHECK(std::abs(none.ci_low()) < 1e-9);
    CHECK(none.ci_high() > 0.2);
    CHECK(none.ci_high() < 0.35);
    const EntrantStats all = interval(10, 10);
    CHECK(std::abs(all.ci_high() - 1.0) < 1e-9);
    CHECK(all.ci_low() > 0.65);
    const EntrantStats empty = interval(0, 0);
    CHECK(empty.win_rate() == 0);
    for (int games : {1, 7, 40, 400})
        for (int wins = 0; wins <= games; wins += (games > 40 ? 37 : 1)) {
            const EntrantStats e = interval(wins, games);
            CHECK(e.ci_low() <= e.win_rate() + 1e-12);
            CHECK(e.win_rate() <= e.ci_high() + 1e-12);
            CHECK(e.ci_low() >= -1e-12);
            CHECK(e.ci_high() <= 1 + 1e-12);
        }
}

TEST_CASE("greedy beats random", "[eval]") {
    auto rnd = make_random_policy();
    auto gr = make_greedy_policy();
    const MatchResult m = run_match({gr.get(), rnd.get(), rnd.get(), rnd.get()}, 100);
    REQUIRE(m.games == 400);
    INFO("greedy " << m.entrant[0].wins << "/400, ci [" << m.entrant[0].ci_low() << ", "
                   << m.entrant[0].ci_high() << "], capped " << m.capped);
    CHECK(m.entrant[0].ci_low() > 0.5);
    CHECK(m.entrant[0].mean_vp > m.entrant[1].mean_vp);
    CHECK(m.capped <= 20);
}

TEST_CASE("greedy mirror match has no gross seat bias", "[eval]") {
    // With one policy in every seat the 4 rotations of run_match replay the same game, so play
    // 200 distinct seeds directly. Turn order gives seat 0 a real but mild edge (~30% vs ~22%
    // over 5000 games); the bound only catches gross breakage.
    auto gr = make_greedy_policy();
    Policy* g = gr.get();
    array<int, NUM_PLAYERS> seat_wins{};
    int capped = 0;
    for (uint64_t seed = 1; seed <= 200; ++seed) {
        const GameResult r = play_game(seed, {g, g, g, g});
        if (r.winner == NONE) ++capped;
        else ++seat_wins[r.winner];
    }
    CHECK(capped <= 10);
    for (int seat = 0; seat < NUM_PLAYERS; ++seat) {
        INFO("seat " << seat << " wins " << seat_wins[seat] << "/200");
        CHECK(seat_wins[seat] >= 20);
        CHECK(seat_wins[seat] < 90);
    }
}
