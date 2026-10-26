#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "catan/board.h"
#include "catan/decision.h"
#include "catan/event.h"
#include "catan/rules.h"
#include "catan/state.h"
#include "support/fixtures.h"

using namespace std;
using namespace catan;
using namespace catan::test;

namespace {

// Golden replays pin whole games. They change ONLY when gameplay changes: a deliberate rules,
// setup, chance or move-ordering change must regenerate them (run with CATAN_GOLDEN_PRINT=1
// to print fresh values). An unexpected diff means the engine's behavior drifted.
struct Golden {
    uint64_t seed;
    bool standard;  // standard_board() instead of random_board(seed)
    uint64_t hash;
    uint16_t ply;
    uint16_t turn;
    uint8_t winner;  // seat, NONE if the turn cap ended the game
};

constexpr uint8_t NOBODY = NONE;

constexpr array<Golden, 7> GOLDENS = {{
    {1, false, 0x4988f73e02852a53ull, 1733, 352, 0},
    {2, false, 0x1ccc3d1e296214fbull, 2524, 549, 1},
    {3, false, 0x944f636b0047e8c9ull, 997, 213, 1},
    {42, false, 0x7e94d47ee545caefull, 1379, 289, 1},
    {1000, false, 0xef814b2573a0d7c2ull, 1758, 395, 3},
    {123456789, false, 0x8ae23c5d86cb2f2eull, 897, 170, 2},
    {7, true, 0x970d831133cfa2c0ull, 1127, 232, 0},
}};

Board board_of(const Golden& g) { return g.standard ? standard_board() : random_board(g.seed); }

// Plays a full game; `winner` is the GAME_OVER event's player.
GameState play_game(const Board& board, uint64_t seed, uint8_t& winner) {
    GameState s = initial_state(board);
    winner = NOBODY;
    play_random(board, s, seed, 1 << 30,
                [&](const GameState&, const Decision&, const GameState&, const EventList& ev) {
                    for (const Event& e : ev)
                        if (e.type == EventType::GAME_OVER) winner = e.player;
                });
    return s;
}

// Order-sensitive fold of per-ply hashes: any divergence changes every later value.
uint64_t fold(uint64_t acc, uint64_t h) {
    return (acc ^ h) * 0x100000001B3ull + 0x9E3779B97F4A7C15ull;
}

bool printing() { return getenv("CATAN_GOLDEN_PRINT") != nullptr; }

}  // namespace

TEST_CASE("golden: full random games replay to pinned final states", "[golden]") {
    for (const Golden& g : GOLDENS) {
        const Board board = board_of(g);
        uint8_t winner = NOBODY;
        const GameState s = play_game(board, g.seed, winner);
        if (printing())
            printf("    {%llu, %s, 0x%016llxull, %u, %u, %u},\n", (unsigned long long)g.seed,
                   g.standard ? "true" : "false", (unsigned long long)state_hash(s), s.pub.ply,
                   s.pub.turn, winner);
        INFO("seed " << g.seed << (g.standard ? " standard" : " random"));
        CHECK(state_hash(s) == g.hash);
        CHECK(s.pub.ply == g.ply);
        CHECK(s.pub.turn == g.turn);
        CHECK(s.pub.phase == Phase::TERMINAL);
        CHECK(winner == g.winner);
    }
}

TEST_CASE("golden: setup and the first plies are pinned", "[golden]") {
    const Board board = random_board(1);
    const GameState setup = random_setup(board, 1);
    // Rolling hash of the first 50 plies of seed 1, so a regression is localized early.
    GameState s = initial_state(board);
    uint64_t acc = 0;
    play_random(board, s, 1, 50, [&](const GameState&, const Decision&, const GameState& after,
                                      const EventList&) { acc = fold(acc, state_hash(after)); });
    if (printing())
        printf("setup 0x%016llxull rolling 0x%016llxull\n",
               (unsigned long long)state_hash(setup), (unsigned long long)acc);
    CHECK(state_hash(setup) == 0x684ba6b6e849a706ull);
    CHECK(s.pub.ply == 50);
    CHECK(acc == 0x1e2837654ce4c242ull);
}

TEST_CASE("golden: state_hash sees every byte and ignores identity", "[golden]") {
    const Board board = random_board(5);
    GameState s = initial_state(board);
    play_random(board, s, 5, 200);
    const uint64_t h = state_hash(s);

    const GameState copy = s;
    CHECK(state_hash(copy) == h);

    unsigned char bytes[sizeof(GameState)];
    memcpy(bytes, &s, sizeof s);
    int unchanged = 0;
    for (size_t i = 0; i < sizeof bytes; ++i) {
        GameState t;
        unsigned char flipped[sizeof(GameState)];
        memcpy(flipped, bytes, sizeof bytes);
        flipped[i] ^= 0x01;
        memcpy(&t, flipped, sizeof t);
        if (state_hash(t) == h) ++unchanged;
    }
    CHECK(unchanged == 0);
}
