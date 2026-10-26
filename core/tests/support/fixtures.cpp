#include "support/fixtures.h"

#include <algorithm>

#include "catan/chance.h"
#include "catan/mask.h"
#include "catan/rng.h"
#include "catan/rules.h"
#include "catan/topology.h"
#include "support/reference.h"

using namespace std;

namespace catan::test {

const vector<Action>& all_actions() {
    static const vector<Action> actions = [] {
        vector<Action> out;
        for (int t = 0; t < NUM_ACTION_TYPES; ++t) {
            for (int a = 0; a < NUM_EDGES; ++a) {
                out.push_back({static_cast<ActionType>(t), static_cast<uint8_t>(a)});
            }
        }
        return out;
    }();
    return actions;
}

GameState random_setup(const Board& board, uint64_t seed) {
    GameState s = initial_state(board);
    for (int i = 0; i < SETUP_STEPS; ++i) {
        const Decision d = next_decision(board, s);
        const auto n = random_below(seed, 99, s.pub.ply, d.n_legal);
        apply(board, s, nth_legal(d.mask, d.phase, static_cast<int>(n)));
    }
    return s;
}

int play_random(const Board& board, GameState& s, uint64_t seed, int max_decisions,
                const StepHook& hook) {
    int taken = 0;
    for (Decision d = next_decision(board, s);
         d.kind != DecisionKind::TERMINAL && taken < max_decisions; d = next_decision(board, s)) {
        const GameState before = s;
        EventList events;
        if (d.kind == DecisionKind::CHANCE) {
            events = apply_chance(board, s, sample_chance(d.chance, s, seed));
        } else {
            const auto n = random_below(seed, 1000, s.pub.ply, d.n_legal);
            events = apply(board, s, nth_legal(d.mask, d.phase, static_cast<int>(n)));
        }
        ++taken;
        if (hook) hook(before, d, s, events);
    }
    return taken;
}

void set_main(GameState& s, uint8_t p, uint8_t dice) {
    PublicState& pub = s.pub;
    pub.phase = Phase::MAIN;
    pub.current_player = p;
    pub.dice = dice;
    pub.pending_chance = 0;
    pub.steal_victim = NONE;
    pub.pending_discard.fill(0);
    pub.discard_floor = 0;
    pub.free_roads = 0;
    pub.yop_remaining = 0;
    pub.exchange_give = NONE;
    pub.played_dev_this_turn = 0;
    s.hid.dev_bought_this_turn.fill(0);
}

void give(GameState& s, uint8_t p, Resource r, int n) {
    const int i = idx(r);
    s.pub.bank[i] = static_cast<uint8_t>(s.pub.bank[i] - n);
    s.hid.hands[p][i] = static_cast<uint8_t>(s.hid.hands[p][i] + n);
    s.pub.hand_size[p] = static_cast<uint8_t>(s.pub.hand_size[p] + n);
}

void clear_hands(GameState& s) {
    for (uint8_t p = 0; p < NUM_PLAYERS; ++p) {
        for (int r = 0; r < NUM_RES; ++r) give(s, p, static_cast<Resource>(r), -s.hid.hands[p][r]);
    }
}

void give_dev(GameState& s, uint8_t p, DevCard c, int n) {
    const int i = idx(c);
    s.hid.dev_deck[i] = static_cast<uint8_t>(s.hid.dev_deck[i] - n);
    s.pub.dev_deck_remaining = static_cast<uint8_t>(s.pub.dev_deck_remaining - n);
    s.hid.dev_held[p][i] = static_cast<uint8_t>(s.hid.dev_held[p][i] + n);
    s.pub.dev_count[p] = static_cast<uint8_t>(s.pub.dev_count[p] + n);
}

void place_settlement(GameState& s, uint8_t p, int v) {
    s.pub.vertex[v] = make_settlement(p);
    --s.pub.pieces_left[p][PIECE_SETTLEMENT];
    ++s.pub.public_vp[p];
}

void place_city(GameState& s, uint8_t p, int v) {
    s.pub.vertex[v] = make_city(p);
    --s.pub.pieces_left[p][PIECE_CITY];
    ++s.pub.pieces_left[p][PIECE_SETTLEMENT];
    ++s.pub.public_vp[p];
}

void place_road(GameState& s, uint8_t p, int e) {
    edge_set_add(s.pub.roads[p], e);
    --s.pub.pieces_left[p][PIECE_ROAD];
}

void clear_board(GameState& s) {
    PublicState& pub = s.pub;
    for (int v = 0; v < NUM_VERTICES; ++v) {
        const uint8_t b = pub.vertex[v];
        if (!vertex_occupied(b)) continue;
        const uint8_t p = vertex_owner(b);
        if (vertex_is_city(b)) {
            ++pub.pieces_left[p][PIECE_CITY];
            pub.public_vp[p] = static_cast<uint8_t>(pub.public_vp[p] - 2);
        } else {
            ++pub.pieces_left[p][PIECE_SETTLEMENT];
            --pub.public_vp[p];
        }
        pub.vertex[v] = 0;
    }
    for (uint8_t p = 0; p < NUM_PLAYERS; ++p) {
        pub.roads[p] = {};
        pub.pieces_left[p][PIECE_ROAD] = ROADS_PER_PLAYER;
        pub.road_len[p] = 0;
    }
    if (pub.longest_road_holder != NONE) {
        pub.public_vp[pub.longest_road_holder] =
            static_cast<uint8_t>(pub.public_vp[pub.longest_road_holder] - 2);
        pub.longest_road_holder = NONE;
    }
}

void refresh_awards(GameState& s) {
    PublicState& pub = s.pub;
    int best = 0;
    for (uint8_t p = 0; p < NUM_PLAYERS; ++p) {
        pub.road_len[p] = static_cast<uint8_t>(reference_road_length(pub, p));
        best = max(best, static_cast<int>(pub.road_len[p]));
    }
    // Fresh assignment (no history): a unique leader at >= 5 holds it, else nobody.
    uint8_t lr = NONE;
    int leaders = 0;
    for (uint8_t p = 0; p < NUM_PLAYERS; ++p) {
        if (best >= 5 && pub.road_len[p] == best) {
            lr = p;
            ++leaders;
        }
    }
    if (leaders != 1) lr = NONE;
    if (pub.longest_road_holder != NONE) {
        pub.public_vp[pub.longest_road_holder] =
            static_cast<uint8_t>(pub.public_vp[pub.longest_road_holder] - 2);
    }
    pub.longest_road_holder = lr;
    if (lr != NONE) pub.public_vp[lr] = static_cast<uint8_t>(pub.public_vp[lr] + 2);
}

uint8_t edge_between(int a, int b) {
    const Topology& topo = topology();
    for (uint8_t e : topo.vertex_edges[a]) {
        if (e == NONE) continue;
        const auto& ends = topo.edge_vertices[e];
        if ((ends[0] == a && ends[1] == b) || (ends[0] == b && ends[1] == a)) return e;
    }
    return NONE;
}

vector<Event> events_of(const EventList& events, EventType t) {
    vector<Event> out;
    for (const Event& e : events) {
        if (e.type == t) out.push_back(e);
    }
    return out;
}

}  // namespace catan::test
