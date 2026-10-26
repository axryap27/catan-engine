#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>

#include "catan/belief.h"
#include "catan/board.h"
#include "catan/event.h"
#include "catan/invariants.h"
#include "catan/rng.h"
#include "catan/rules.h"
#include "catan/state.h"
#include "support/fixtures.h"

using namespace std;
using namespace catan;
using namespace catan::test;

namespace {

constexpr int VP = idx(DevCard::VICTORY_POINT);

Board board_for(uint64_t seed) { return seed % 2 ? standard_board() : random_board(seed); }

bool same_bytes(const GameState& a, const GameState& b) { return memcmp(&a, &b, sizeof a) == 0; }

// Remembers the first failure of a long loop; per-step REQUIREs would swamp Catch.
struct Failure {
    string msg;
    void fail(uint64_t seed, uint16_t ply, const string& what) {
        if (msg.empty()) msg = "seed " + to_string(seed) + " ply " + to_string(ply) + ": " + what;
    }
};

const char* bounds_violation(const BeliefState& b, const GameState& s) {
    for (uint8_t o = 0; o < NUM_PLAYERS; ++o) {
        for (uint8_t t = 0; t < NUM_PLAYERS; ++t) {
            for (int r = 0; r < NUM_RES; ++r) {
                const int truth = s.hid.hands[t][r];
                if (hand_min(b, o, t, r) > truth) return "hand_min above truth";
                if (hand_max(b, s.pub, o, t, r) < truth) return "hand_max below truth";
                if (o == t && hand_max(b, s.pub, o, t, r) != truth) return "own max not exact";
            }
        }
    }
    return nullptr;
}

const char* vp_violation(const GameState& s, const BeliefState& b) {
    const PublicState& pub = s.pub;
    if (pub.phase == Phase::TERMINAL) return nullptr;
    for (int t = 0; t < NUM_PLAYERS; ++t) {
        const int vp = s.hid.dev_held[t][VP];
        if (t == pub.current_player && pub.public_vp[t] + vp >= 10) return "current VP >= 10";
        if (t != pub.current_player && b.vp_at_turn_end[t] + vp >= 10) return "VP at turn end";
    }
    return nullptr;
}

// Belief of a crafted state: every observer sees only its own hand.
BeliefState opaque_belief(const GameState& s) {
    BeliefState b = initial_belief();
    for (int o = 0; o < NUM_PLAYERS; ++o) {
        for (int t = 0; t < NUM_PLAYERS; ++t) {
            if (o == t) {
                b.hand[o][t].known = s.hid.hands[t];
            } else {
                b.hand[o][t].unknown = s.pub.hand_size[t];
            }
        }
    }
    return b;
}

// Everybody knows every hand exactly.
BeliefState open_belief(const array<array<uint8_t, NUM_RES>, NUM_PLAYERS>& hands) {
    BeliefState b = initial_belief();
    for (int o = 0; o < NUM_PLAYERS; ++o) {
        for (int t = 0; t < NUM_PLAYERS; ++t) b.hand[o][t].known = hands[t];
    }
    return b;
}

PublicState pub_with_sizes(const BeliefState& b) {
    PublicState pub{};
    for (int t = 0; t < NUM_PLAYERS; ++t) {
        int n = b.hand[0][t].unknown;
        for (int r = 0; r < NUM_RES; ++r) n += b.hand[0][t].known[r];
        pub.hand_size[t] = static_cast<uint8_t>(n);
    }
    return pub;
}

EventList one(Event e) {
    EventList ev;
    ev.push(e);
    return ev;
}

constexpr uint8_t B = idx(Resource::BRICK), L = idx(Resource::LUMBER), W = idx(Resource::WOOL),
                  G = idx(Resource::GRAIN), O = idx(Resource::ORE);

// Full checks of one resampled state against its truth.
void check_resample(const Board& board, const GameState& truth, const BeliefState& b,
                    uint8_t o, const GameState& r, Failure& f, uint64_t seed) {
    const uint16_t ply = truth.pub.ply;
    if (const char* why = invariant_violation(board, r)) f.fail(seed, ply, why);
    if (memcmp(&r.pub, &truth.pub, sizeof r.pub) != 0) f.fail(seed, ply, "pub changed");
    if (r.hid.hands[o] != truth.hid.hands[o]) f.fail(seed, ply, "observer hand changed");
    if (r.hid.dev_held[o] != truth.hid.dev_held[o]) f.fail(seed, ply, "observer devs changed");
    if (truth.pub.current_player == o &&
        r.hid.dev_bought_this_turn != truth.hid.dev_bought_this_turn) {
        f.fail(seed, ply, "observer dev_bought_this_turn changed");
    }
    if (const char* why = belief_violation(b, r, o)) f.fail(seed, ply, why);
    if (const char* why = vp_violation(r, b)) f.fail(seed, ply, why);
    if (const char* why = belief_violation(resampled_belief(b, r, o), r)) f.fail(seed, ply, why);
}

}  // namespace

TEST_CASE("after setup every hand is known to everyone", "[belief]") {
    for (uint64_t seed = 1; seed <= 20; ++seed) {
        const Board board = board_for(seed);
        GameState s = initial_state(board);
        BeliefState b = initial_belief();
        play_random(board, s, seed, SETUP_STEPS,
                    [&](const GameState&, const Decision&, const GameState& after,
                        const EventList& ev) { observe(b, after.pub, ev); });
        REQUIRE(s.pub.phase == Phase::ROLL);
        REQUIRE(belief_violation(b, s) == nullptr);
        for (int o = 0; o < NUM_PLAYERS; ++o) {
            for (int t = 0; t < NUM_PLAYERS; ++t) {
                CHECK(b.hand[o][t].unknown == 0);
                CHECK(b.hand[o][t].known == s.hid.hands[t]);
            }
        }
    }
}

TEST_CASE("beliefs stay sound and bounds hold through random games", "[belief]") {
    Failure f;
    long steps = 0, steals_seen = 0, unknown_after_steal = 0;
    for (uint64_t seed = 1; seed <= 300 && f.msg.empty(); ++seed) {
        const Board board = board_for(seed);
        GameState s = initial_state(board);
        BeliefState b = initial_belief();
        play_random(board, s, seed, 1 << 30,
                    [&](const GameState&, const Decision&, const GameState& after,
                        const EventList& ev) {
                        observe(b, after.pub, ev);
                        ++steps;
                        if (const char* why = belief_violation(b, after)) {
                            f.fail(seed, after.pub.ply, why);
                        }
                        if (const char* why = bounds_violation(b, after)) {
                            f.fail(seed, after.pub.ply, why);
                        }
                        for (const Event& e : ev) {
                            if (e.type != EventType::STEAL) continue;
                            ++steals_seen;
                            for (int o = 0; o < NUM_PLAYERS; ++o) {
                                if (o != e.player && o != e.target) {
                                    unknown_after_steal += b.hand[o][e.player].unknown > 0;
                                }
                            }
                        }
                    });
    }
    INFO(f.msg);
    REQUIRE(f.msg.empty());
    REQUIRE(steps > 100000);
    REQUIRE(steals_seen > 0);
    // Each steal leaves both non-participants unsure of the thief (unless deduced).
    REQUIRE(unknown_after_steal > steals_seen);
}

TEST_CASE("GAIN, SPEND and DISCARD update every observer", "[belief]") {
    BeliefState b = initial_belief();
    PublicState pub{};
    pub.hand_size[1] = 3;
    observe(b, pub, one(Event{EventType::GAIN, 1, NONE, W, 3}));
    for (int o = 0; o < NUM_PLAYERS; ++o) {
        CHECK(b.hand[o][1].known[W] == 3);
        CHECK(b.hand[o][1].unknown == 0);
    }

    // Observer 2 knows one brick and two unknown cards of player 1; player 1 spends 2 brick.
    b = initial_belief();
    b.hand[2][1].known[B] = 1;
    b.hand[2][1].unknown = 2;
    for (int o : {0, 1, 3}) b.hand[o][1].known[B] = 3;
    pub = pub_with_sizes(b);
    pub.hand_size[1] = 1;
    observe(b, pub, one(Event{EventType::SPEND, 1, NONE, B, 2}));
    CHECK(b.hand[2][1].known[B] == 0);
    CHECK(b.hand[2][1].unknown == 1);
    CHECK(b.hand[0][1].known[B] == 1);

    pub.hand_size[1] = 0;
    observe(b, pub, one(Event{EventType::DISCARD, 1, NONE, B, 1}));
    CHECK(b.hand[2][1].unknown == 0);
    CHECK(b.hand[0][1].known[B] == 0);
}

TEST_CASE("a steal is exact for thief and victim and blurs the hands for others", "[belief]") {
    array<array<uint8_t, NUM_RES>, NUM_PLAYERS> hands{};
    hands[1] = {2, 0, 1, 0, 0};  // victim: brick 2, wool 1
    hands[0] = {0, 0, 0, 1, 0};  // thief: grain 1
    BeliefState b = open_belief(hands);
    PublicState pub = pub_with_sizes(b);
    pub.hand_size[0] = 2;
    pub.hand_size[1] = 2;
    observe(b, pub, one(Event{EventType::STEAL, 0, 1, W, 1}));
    for (int o : {0, 1}) {
        CHECK(b.hand[o][1].known == array<uint8_t, NUM_RES>{2, 0, 0, 0, 0});
        CHECK(b.hand[o][1].unknown == 0);
        CHECK(b.hand[o][0].known == array<uint8_t, NUM_RES>{0, 0, 1, 1, 0});
        CHECK(b.hand[o][0].unknown == 0);
    }
    for (int o : {2, 3}) {
        // Either a brick or the wool went: one brick is certain, one card is unknown.
        CHECK(b.hand[o][1].known == array<uint8_t, NUM_RES>{1, 0, 0, 0, 0});
        CHECK(b.hand[o][1].unknown == 1);
        CHECK(b.hand[o][0].known == array<uint8_t, NUM_RES>{0, 0, 0, 1, 0});
        CHECK(b.hand[o][0].unknown == 1);
    }

    // A hand of one known type: the stolen card's type is deduced.
    hands = {};
    hands[2] = {0, 0, 0, 0, 3};
    b = open_belief(hands);
    pub = pub_with_sizes(b);
    pub.hand_size[2] = 2;
    pub.hand_size[3] = 1;
    observe(b, pub, one(Event{EventType::STEAL, 3, 2, O, 1}));
    for (int o = 0; o < NUM_PLAYERS; ++o) {
        CHECK(b.hand[o][2].known[O] == 2);
        CHECK(b.hand[o][2].unknown == 0);
        CHECK(b.hand[o][3].known[O] == 1);
        CHECK(b.hand[o][3].unknown == 0);
    }

    // A wholly unknown victim loses an unknown card; partly known hands drop each known type.
    b = initial_belief();
    b.hand[0][1].unknown = 2;
    b.hand[0][2].known = {1, 0, 0, 0, 0};
    b.hand[0][2].unknown = 1;
    for (int o = 1; o < NUM_PLAYERS; ++o) {
        b.hand[o][1].known = {0, 2, 0, 0, 0};
        b.hand[o][2].known = {1, 0, 0, 0, 1};
    }
    pub = pub_with_sizes(b);
    pub.hand_size[1] = 1;
    pub.hand_size[3] = 1;
    observe(b, pub, one(Event{EventType::STEAL, 3, 1, L, 1}));
    CHECK(b.hand[0][1].known == array<uint8_t, NUM_RES>{});
    CHECK(b.hand[0][1].unknown == 1);
    CHECK(b.hand[0][3].unknown == 1);
    pub.hand_size[2] = 1;
    pub.hand_size[3] = 2;
    observe(b, pub, one(Event{EventType::STEAL, 3, 2, O, 1}));
    CHECK(b.hand[0][2].known == array<uint8_t, NUM_RES>{});
    CHECK(b.hand[0][2].unknown == 1);
    CHECK(b.hand[0][3].unknown == 2);
}

TEST_CASE("monopoly removes known cards first and zeroes known[r]", "[belief]") {
    BeliefState b = initial_belief();
    // Observer 2's view of player 1: brick 1 known, 3 unknown (truth: brick 2, grain 2).
    b.hand[2][1].known[B] = 1;
    b.hand[2][1].unknown = 3;
    for (int o : {0, 1, 3}) b.hand[o][1].known = {2, 0, 0, 2, 0};
    PublicState pub = pub_with_sizes(b);
    pub.hand_size[1] = 2;
    pub.hand_size[0] = 2;
    EventList ev;
    ev.push(Event{EventType::MONOPOLY, 0, 1, B, 2});
    ev.push(Event{EventType::MONOPOLY, 0, 2, B, 0});
    ev.push(Event{EventType::MONOPOLY, 0, 3, B, 0});
    observe(b, pub, ev);
    for (int o = 0; o < NUM_PLAYERS; ++o) {
        CHECK(b.hand[o][1].known[B] == 0);
        CHECK(b.hand[o][0].known[B] == 2);
    }
    CHECK(b.hand[2][1].unknown == 2);
    CHECK(b.hand[0][1].known[G] == 2);
}

TEST_CASE("dev card and turn bookkeeping", "[belief]") {
    BeliefState b = initial_belief();
    PublicState pub{};
    pub.public_vp = {3, 4, 5, 6};
    EventList ev;
    ev.push(Event{EventType::DEV_BOUGHT, 2, NONE, idx(DevCard::KNIGHT), 0});
    ev.push(Event{EventType::DEV_BOUGHT, 2, NONE, idx(DevCard::VICTORY_POINT), 0});
    ev.push(Event{EventType::DEV_PLAYED, 2, NONE, idx(DevCard::MONOPOLY), 0});
    observe(b, pub, ev);
    CHECK(b.bought_this_turn == 2);
    CHECK(b.dev_played[idx(DevCard::MONOPOLY)] == 1);
    observe(b, pub, one(Event{EventType::TURN_ENDED, 2, NONE, 0, 0}));
    CHECK(b.bought_this_turn == 0);
    CHECK(b.vp_at_turn_end == array<uint8_t, NUM_PLAYERS>{0, 0, 5, 0});
}

TEST_CASE("resample_hidden is consistent at random points of random games", "[belief]") {
    Failure f;
    long points = 0, nontrivial = 0, differed = 0, continued = 0, constrained = 0;
    for (uint64_t seed = 1; seed <= 40 && f.msg.empty(); ++seed) {
        const Board board = board_for(seed);
        GameState s = initial_state(board);
        BeliefState b = initial_belief();
        play_random(board, s, seed, 1 << 30,
                    [&](const GameState&, const Decision&, const GameState& after,
                        const EventList& ev) {
                        observe(b, after.pub, ev);
                        const Phase ph = after.pub.phase;
                        const bool special = ph == Phase::DISCARD || ph == Phase::EXCHANGE_RECV;
                        const uint32_t odds = special ? 4 : 40;
                        if (random_below(seed, 77, after.pub.ply, odds) != 0) return;
                        ++points;
                        for (uint8_t o = 0; o < NUM_PLAYERS; ++o) {
                            const uint64_t rs = seed << 20 | after.pub.ply;
                            const GameState r1 = resample_hidden(board, after, b, o, rs);
                            const GameState r2 = resample_hidden(board, after, b, o, rs + 1);
                            check_resample(board, after, b, o, r1, f, seed);
                            check_resample(board, after, b, o, r2, f, seed);
                            if (!same_bytes(r1, resample_hidden(board, after, b, o, rs))) {
                                f.fail(seed, after.pub.ply, "not deterministic");
                            }
                            int unknown = 0;
                            for (int t = 0; t < NUM_PLAYERS; ++t) {
                                if (t == o) continue;
                                unknown += b.hand[o][t].unknown + after.pub.dev_count[t];
                            }
                            if (unknown >= 3) {
                                ++nontrivial;
                                differed += !same_bytes(r1, r2);
                            }
                            const bool hidden_mover =
                                special && next_decision(board, after).player != o;
                            constrained += hidden_mover;
                            const uint32_t keep = hidden_mover ? 4 : 32;
                            if (random_below(seed, 78 + o, after.pub.ply, keep) == 0) {
                                // Play on, tracking the rebased belief, which must stay sound.
                                GameState g = r1;
                                BeliefState gb = resampled_belief(b, r1, o);
                                play_random(board, g, seed + o, 200,
                                            [&](const GameState&, const Decision&,
                                                const GameState& a2, const EventList& e2) {
                                                observe(gb, a2.pub, e2);
                                                if (const char* why = belief_violation(gb, a2)) {
                                                    f.fail(seed, a2.pub.ply, why);
                                                }
                                            });
                                ++continued;
                            }
                        }
                    });
    }
    INFO(f.msg);
    REQUIRE(f.msg.empty());
    REQUIRE(points > 1000);
    REQUIRE(constrained > 100);
    REQUIRE(continued > 100);
    INFO("nontrivial " << nontrivial << " differed " << differed);
    REQUIRE(differed * 10 > nontrivial * 6);  // ~75%: small hands with skewed pools often repeat
}

TEST_CASE("resampled hands follow the hypergeometric distribution", "[belief]") {
    const Board board = standard_board();
    GameState s = initial_state(board);
    set_main(s, 0);
    give(s, 0, Resource::ORE, 2);
    give(s, 1, Resource::BRICK, 2);  // two cards observer 0 cannot see
    give(s, 2, Resource::LUMBER, 2);
    give(s, 2, Resource::ORE, 1);    // three more
    const BeliefState b = opaque_belief(s);
    REQUIRE(belief_violation(b, s) == nullptr);
    // Pool for 0: brick 2, lumber 2, ore 1. Player 1 draws 2 of 5: P(k bricks) = 3, 6, 1 / 10.
    const int N = 20000;
    array<int, 3> bricks{};
    int ore_at_1 = 0;
    for (uint64_t seed = 0; seed < N; ++seed) {
        const GameState r = resample_hidden(board, s, b, 0, seed);
        REQUIRE(r.hid.hands[1][B] + r.hid.hands[1][L] + r.hid.hands[1][O] == 2);
        ++bricks[r.hid.hands[1][B]];
        ore_at_1 += r.hid.hands[1][O];
    }
    CHECK(abs(bricks[0] / double(N) - 0.3) < 0.02);
    CHECK(abs(bricks[1] / double(N) - 0.6) < 0.02);
    CHECK(abs(bricks[2] / double(N) - 0.1) < 0.02);
    CHECK(abs(ore_at_1 / double(N) - 0.4) < 0.02);  // E[ore] = 2 * 1/5
}

TEST_CASE("resample repairs deals the rejection loop cannot find", "[belief]") {
    const Board board = standard_board();

    SECTION("VP cards over an opponent's cap") {
        GameState s = initial_state(board);
        set_main(s, 0);
        give_dev(s, 0, DevCard::KNIGHT, 14);
        give_dev(s, 1, DevCard::ROAD_BUILDING, 2);
        give_dev(s, 1, DevCard::YEAR_OF_PLENTY, 2);
        give_dev(s, 1, DevCard::MONOPOLY, 1);
        BeliefState b = opaque_belief(s);
        b.vp_at_turn_end[1] = 9;  // player 1 can hold no VP card
        REQUIRE(belief_violation(b, s) == nullptr);
        // Pool: 5 VP, 2 RB, 2 YoP, 2 Mono; player 1 takes 5 of 11 with no VP 1.3% of the time.
        for (uint64_t seed = 0; seed < 300; ++seed) {
            const GameState r = resample_hidden(board, s, b, 0, seed);
            REQUIRE(invariant_violation(board, r) == nullptr);
            REQUIRE(belief_violation(b, r, 0) == nullptr);
            REQUIRE(r.hid.dev_held[1][VP] == 0);
            REQUIRE(r.hid.dev_deck[VP] == 5);
        }
    }

    SECTION("VP cards bought this turn by the current player") {
        GameState s = initial_state(board);
        set_main(s, 1);
        give_dev(s, 0, DevCard::KNIGHT, 14);
        give_dev(s, 1, DevCard::VICTORY_POINT, 2);
        s.hid.dev_bought_this_turn[VP] = 2;  // both bought this turn
        give_dev(s, 1, DevCard::MONOPOLY, 2);
        BeliefState b = opaque_belief(s);
        b.vp_at_turn_end[1] = 9;  // older cards can hold no VP
        b.bought_this_turn = 2;
        REQUIRE(belief_violation(b, s) == nullptr);
        for (uint64_t seed = 0; seed < 300; ++seed) {
            const GameState r = resample_hidden(board, s, b, 0, seed);
            REQUIRE(invariant_violation(board, r) == nullptr);
            REQUIRE(belief_violation(b, r, 0) == nullptr);
            REQUIRE(r.hid.dev_held[1][VP] - r.hid.dev_bought_this_turn[VP] == 0);
        }
    }

    SECTION("a discarder past its first card keeps enough cards above the floor") {
        GameState s = initial_state(board);
        set_main(s, 0, 7);
        give(s, 1, Resource::ORE, 2);
        give(s, 1, Resource::BRICK, 6);
        give(s, 2, Resource::LUMBER, 19);
        give(s, 2, Resource::GRAIN, 19);
        s.pub.phase = Phase::DISCARD;
        s.pub.pending_discard[1] = 2;
        s.pub.discard_floor = O;
        const BeliefState b = opaque_belief(s);
        REQUIRE(invariant_violation(board, s) == nullptr);
        for (uint64_t seed = 0; seed < 300; ++seed) {
            const GameState r = resample_hidden(board, s, b, 0, seed);
            REQUIRE(invariant_violation(board, r) == nullptr);
            REQUIRE(belief_violation(b, r, 0) == nullptr);
            REQUIRE(r.hid.hands[1][O] == 2);
            GameState g = r;
            play_random(board, g, seed, 50);
        }
    }

    SECTION("a player mid bank exchange holds the cards it gives") {
        GameState s = initial_state(board);
        set_main(s, 3);
        give(s, 3, Resource::WOOL, 4);
        give(s, 3, Resource::BRICK, 3);
        give(s, 2, Resource::LUMBER, 19);
        give(s, 2, Resource::GRAIN, 15);
        s.pub.phase = Phase::EXCHANGE_RECV;
        s.pub.exchange_give = W;
        const BeliefState b = opaque_belief(s);
        REQUIRE(invariant_violation(board, s) == nullptr);
        for (uint64_t seed = 0; seed < 300; ++seed) {
            const GameState r = resample_hidden(board, s, b, 0, seed);
            REQUIRE(invariant_violation(board, r) == nullptr);
            REQUIRE(belief_violation(b, r, 0) == nullptr);
            REQUIRE(r.hid.hands[3][W] == 4);
            GameState g = r;
            play_random(board, g, seed, 50);
        }
    }
}
