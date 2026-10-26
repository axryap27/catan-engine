#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "catan/action.h"
#include "catan/board.h"
#include "catan/chance.h"
#include "catan/decision.h"
#include "catan/event.h"
#include "catan/invariants.h"
#include "catan/mask.h"
#include "catan/rules.h"
#include "catan/state.h"
#include "catan/types.h"
#include "support/fixtures.h"
#include "support/reference.h"

using namespace std;
using namespace catan;
using namespace catan::test;

namespace {

// Games per run: CATAN_FUZZ_GAMES, default 30 so a plain ctest stays fast (CI uses 10000).
int fuzz_games() {
    const char* env = getenv("CATAN_FUZZ_GAMES");
    const int n = env ? atoi(env) : 0;
    return n > 0 ? n : 30;
}

// Odd seeds play the beginner layout, even seeds a random one.
Board board_for(uint64_t seed) { return seed % 2 ? standard_board() : random_board(seed); }

// Remembers the first failed check of a game; later ones are noise.
struct Failure {
    string msg;
    bool failed() const { return !msg.empty(); }
    void fail(uint16_t ply, const string& what) {
        if (!failed()) msg = "ply " + to_string(ply) + ": " + what;
    }
};

// Net change of every hand and dev count that `events` imply.
struct Delta {
    array<array<int, NUM_RES>, NUM_PLAYERS> hand{};
    array<int, NUM_PLAYERS> dev{};
};

Delta expected_delta(const EventList& events) {
    Delta d;
    for (const Event& e : events) {
        switch (e.type) {
            case EventType::GAIN: d.hand[e.player][e.a] += e.b; break;
            case EventType::SPEND:
            case EventType::DISCARD: d.hand[e.player][e.a] -= e.b; break;
            case EventType::STEAL:
                d.hand[e.player][e.a] += 1;
                d.hand[e.target][e.a] -= 1;
                break;
            case EventType::MONOPOLY:
                d.hand[e.player][e.a] += e.b;
                d.hand[e.target][e.a] -= e.b;
                break;
            case EventType::DEV_BOUGHT: d.dev[e.player] += 1; break;
            case EventType::DEV_PLAYED: d.dev[e.player] -= 1; break;
            default: break;
        }
    }
    return d;
}

void check_player_decision(const Board& board, const GameState& before, const Decision& d,
                           Failure& f) {
    const uint16_t ply = before.pub.ply;
    if (d.phase != before.pub.phase) f.fail(ply, "decision phase differs from state phase");
    // The decider: whoever still owes a discard, else the current player.
    if (d.phase == Phase::DISCARD) {
        if (d.player >= NUM_PLAYERS || before.pub.pending_discard[d.player] == 0)
            f.fail(ply, "discard decider owes nothing");
    } else if (d.player != before.pub.current_player) {
        f.fail(ply, "decider is not the current player");
    }
    // Mask against the independent reference, over every syntactically possible action.
    vector<Action> expected;
    for (const Action& a : all_actions()) {
        const bool ref = reference_is_legal(board, before, a);
        if (ref) expected.push_back(a);
        if (is_legal(d.mask, d.phase, a) != ref) {
            f.fail(ply, "is_legal disagrees with reference for type " + to_string(idx(a.type)) +
                            " arg " + to_string(a.arg) + " (reference " + to_string(ref) + ")");
            return;
        }
    }
    if (expected.empty()) f.fail(ply, "no legal action");
    const int n = count_legal(d.mask, d.phase);
    if (n != d.n_legal) f.fail(ply, "count_legal != n_legal");
    if (n != static_cast<int>(expected.size())) {
        f.fail(ply, "count_legal " + to_string(n) + " != reference " + to_string(expected.size()));
        return;
    }
    // nth_legal enumerates exactly the legal set, ascending and distinct.
    for (int i = 0; i < n; ++i) {
        const Action a = nth_legal(d.mask, d.phase, i);
        if (a.type != expected[i].type || a.arg != expected[i].arg) {
            f.fail(ply, "nth_legal(" + to_string(i) + ") is not the " + to_string(i) +
                            "-th legal action");
            return;
        }
    }
}

void check_chance_decision(const Decision& d, Failure& f, uint16_t ply) {
    int sum = 0;
    for (int i = 0; i < d.chance.n_outcomes; ++i) sum += d.chance.weight[i];
    if (sum != d.chance.total) f.fail(ply, "chance weights do not sum to total");
    if (d.chance.total == 0) f.fail(ply, "chance total is zero");
    if (d.chance.kind == ChanceKind::DICE && d.chance.total != 36) f.fail(ply, "dice total != 36");
}

void check_transition(const Board& board, const GameState& before, const Decision& d,
                      const GameState& after, const EventList& events, uint64_t seed,
                      Failure& f) {
    const uint16_t ply = before.pub.ply;
    if (const char* v = invariant_violation(board, after)) f.fail(ply, string("invariant: ") + v);
    if (after.pub.ply != before.pub.ply + 1) f.fail(ply, "ply did not advance by one");

    if (d.kind == DecisionKind::CHANCE) {
        check_chance_decision(d, f, ply);
        const uint8_t o = sample_chance(d.chance, before, seed);
        if (o >= d.chance.n_outcomes || d.chance.weight[o] == 0)
            f.fail(ply, "sampled a zero-weight outcome");
    } else {
        check_player_decision(board, before, d, f);
    }

    // Hands and dev counts change exactly as the events say.
    const Delta want = expected_delta(events);
    for (int p = 0; p < NUM_PLAYERS; ++p) {
        for (int r = 0; r < NUM_RES; ++r) {
            if (after.hid.hands[p][r] - before.hid.hands[p][r] != want.hand[p][r])
                f.fail(ply, "hand of seat " + to_string(p) + " res " + to_string(r) +
                                " does not match events");
        }
        if (after.pub.dev_count[p] - before.pub.dev_count[p] != want.dev[p])
            f.fail(ply, "dev_count of seat " + to_string(p) + " does not match events");
    }

    // Road lengths are recomputed from scratch after anything that can change them.
    bool built = false;
    for (const Event& e : events)
        built |= e.type == EventType::ROAD_BUILT || e.type == EventType::SETTLEMENT_BUILT;
    if (built) {
        for (uint8_t q = 0; q < NUM_PLAYERS; ++q) {
            if (after.pub.road_len[q] != reference_road_length(after.pub, q))
                f.fail(ply, "road_len of seat " + to_string(q) + " is " +
                                to_string(after.pub.road_len[q]) + ", reference " +
                                to_string(reference_road_length(after.pub, q)));
        }
    }
}

// Plays one full game checking every step; returns the failure ("" if none). Fills the final
// hash and ply for the determinism check.
string fuzz_game(uint64_t seed, uint64_t& hash, uint16_t& ply_out) {
    const Board board = board_for(seed);
    GameState s = initial_state(board);
    Failure f;
    if (const char* v = invariant_violation(board, s)) f.fail(0, string("initial: ") + v);

    int game_overs = 0;
    GameState last_after = s;
    EventList last_events;
    play_random(board, s, seed, 1 << 30,
                [&](const GameState& before, const Decision& d, const GameState& after,
                    const EventList& events) {
                    if (!f.failed()) check_transition(board, before, d, after, events, seed, f);
                    for (const Event& e : events) game_overs += e.type == EventType::GAME_OVER;
                    last_after = after;
                    last_events = events;
                });

    const uint16_t ply = s.pub.ply;
    if (next_decision(board, s).kind != DecisionKind::TERMINAL) f.fail(ply, "game did not end");
    if (s.pub.phase != Phase::TERMINAL) f.fail(ply, "final phase is not TERMINAL");
    bool over_in_last = false;
    uint8_t winner = NONE;
    for (const Event& e : last_events) {
        if (e.type == EventType::GAME_OVER) {
            over_in_last = true;
            winner = e.player;
        }
    }
    if (!over_in_last || game_overs != 1) f.fail(ply, "GAME_OVER not exactly once, in last step");
    if (winner != NONE) {
        const int vp =
            s.pub.public_vp[winner] + s.hid.dev_held[winner][idx(DevCard::VICTORY_POINT)];
        if (vp < VICTORY_POINTS_TO_WIN) f.fail(ply, "winner has fewer than 10 VP");
        if (winner != s.pub.current_player) f.fail(ply, "winner is not the current player");
    } else if (s.pub.turn < TURN_CAP) {
        f.fail(ply, "no winner before the turn cap");
    }
    hash = state_hash(s);
    ply_out = s.pub.ply;
    return f.msg;
}

}  // namespace

TEST_CASE("fuzz: random games satisfy every per-step check", "[fuzz]") {
    const int games = fuzz_games();
    for (int i = 1; i <= games; ++i) {
        const uint64_t seed = static_cast<uint64_t>(i);
        uint64_t hash = 0;
        uint16_t ply = 0;
        const string msg = fuzz_game(seed, hash, ply);
        if (!msg.empty()) FAIL("seed " << seed << " (" << (seed % 2 ? "standard" : "random")
                                       << " board), " << msg);

        // Replaying the same seed reproduces the game exactly.
        const Board board = board_for(seed);
        GameState again = initial_state(board);
        play_random(board, again, seed);
        if (state_hash(again) != hash || again.pub.ply != ply)
            FAIL("seed " << seed << ": replay diverged");
    }
}

TEST_CASE("fuzz: games reach every phase and every action type", "[fuzz]") {
    array<bool, 11> phase_seen{};
    array<bool, NUM_ACTION_TYPES> type_seen{};
    array<bool, 4> chance_seen{};
    const int games = 100;
    for (int i = 1; i <= games; ++i) {
        const uint64_t seed = static_cast<uint64_t>(i);
        const Board board = board_for(seed);
        GameState s = initial_state(board);
        play_random(board, s, seed, 1 << 30,
                    [&](const GameState& before, const Decision& d, const GameState&,
                        const EventList&) {
                        phase_seen[static_cast<int>(before.pub.phase)] = true;
                        if (d.kind == DecisionKind::CHANCE)
                            chance_seen[static_cast<int>(d.chance.kind)] = true;
                        else
                            for (int t = 0; t < NUM_ACTION_TYPES; ++t)
                                if (type_legal(d.mask, static_cast<ActionType>(t)))
                                    type_seen[t] = true;
                    });
        phase_seen[static_cast<int>(s.pub.phase)] = true;
    }
    for (int p = 0; p < 11; ++p) CHECK(phase_seen[p]);
    // PROPOSE_TRADE and RESPOND_TRADE are declared but always masked.
    for (int t = 0; t < NUM_ACTION_TYPES; ++t) {
        const bool trade =
            t == idx(ActionType::PROPOSE_TRADE) || t == idx(ActionType::RESPOND_TRADE);
        CHECK(type_seen[t] == !trade);
    }
    CHECK(chance_seen[static_cast<int>(ChanceKind::DICE)]);
    CHECK(chance_seen[static_cast<int>(ChanceKind::DEV_DRAW)]);
    CHECK(chance_seen[static_cast<int>(ChanceKind::STEAL)]);
}
