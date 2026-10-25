#include <array>

#include "rules_internal.h"

using namespace std;

namespace catan::detail {
namespace {

// The settlement placed this setup round: p's only settlement without an incident road of p's.
// In round 2 the round-1 settlement already carries its road, so this needs no extra state.
uint8_t unroaded_settlement(const PublicState& pub, uint8_t p) {
    const Topology& topo = topology();
    const EdgeSet& roads = pub.roads[p];
    for (int v = 0; v < NUM_VERTICES; ++v) {
        if (pub.vertex[v] != make_settlement(p)) continue;
        const EdgeSet& inc = topo.vertex_edge_mask[v];
        if (((inc[0] & roads[0]) | (inc[1] & roads[1])) == 0) return static_cast<uint8_t>(v);
    }
    return NONE;
}

// Second-round setup income: one card per adjacent producing hex, aggregated per resource.
void grant_setup_income(const Board& board, GameState& s, uint8_t p, uint8_t v,
                        EventList& events) {
    array<uint8_t, NUM_RES> gained{};
    for (uint8_t h : topology().vertex_hexes[v]) {
        if (h == NONE || board.hex_type[h] == HexType::DESERT) continue;
        ++gained[static_cast<int>(board.hex_type[h])];
    }
    for (int r = 0; r < NUM_RES; ++r) {
        if (gained[r] == 0) continue;
        bank_to_hand(s, p, r, gained[r]);
        events.push(event(EventType::GAIN, p, static_cast<uint8_t>(r), gained[r]));
    }
}

}  // namespace

Decision setup_decision(const PublicState& pub) {
    const uint8_t p = pub.current_player;
    CATAN_CHECK(pub.setup_step < SETUP_STEPS);
    CATAN_CHECK(p == setup_player(pub.setup_step));

    Decision d = player_decision(p, Phase::SETUP);
    if (pub.setup_step % 2 == 0) {
        d.mask.settlement = distance_legal(pub);
        if (d.mask.settlement != 0) set_type(d.mask, ActionType::BUILD_SETTLEMENT);
    } else {
        const uint8_t v = unroaded_settlement(pub, p);
        CATAN_CHECK(v != NONE);
        const EdgeSet& inc = topology().vertex_edge_mask[v];
        const EdgeSet taken = all_roads(pub);
        d.mask.edge = {inc[0] & ~taken[0], inc[1] & ~taken[1]};
        if ((d.mask.edge[0] | d.mask.edge[1]) != 0) set_type(d.mask, ActionType::BUILD_ROAD);
    }
    return finish(d);
}

void apply_setup(const Board& board, GameState& s, Action action, EventList& events) {
    PublicState& pub = s.pub;
    const uint8_t p = pub.current_player;

    if (action.type == ActionType::BUILD_SETTLEMENT) {
        const uint8_t v = action.arg;
        pub.vertex[v] = make_settlement(p);
        --pub.pieces_left[p][PIECE_SETTLEMENT];
        ++pub.public_vp[p];
        events.push(event(EventType::SETTLEMENT_BUILT, p, v));
        if (pub.setup_step >= SETUP_STEPS / 2) grant_setup_income(board, s, p, v, events);
    } else {
        const uint8_t e = action.arg;
        edge_set_add(pub.roads[p], e);
        --pub.pieces_left[p][PIECE_ROAD];
        events.push(event(EventType::ROAD_BUILT, p, e));
        // road_len is maintained from PR6.
    }

    ++pub.setup_step;
    if (pub.setup_step < SETUP_STEPS) {
        pub.current_player = setup_player(pub.setup_step);
    } else {
        pub.phase = Phase::ROLL;
        pub.current_player = 0;
        pub.turn = 0;
    }
}

}  // namespace catan::detail
