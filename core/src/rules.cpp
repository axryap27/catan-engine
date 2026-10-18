#include "catan/rules.h"

#include <array>
#include <cstdio>
#include <cstdlib>

#include "catan/check.h"
#include "catan/mask.h"
#include "catan/topology.h"

using namespace std;

namespace catan {
namespace {

constexpr VertexSet ALL54 = (uint64_t{1} << NUM_VERTICES) - 1;

// Positional rather than designated init: -Wmissing-field-initializers flags omitted fields.
Event event(EventType type, uint8_t player, uint8_t a, uint8_t b = 0) {
    return Event{type, player, NONE, a, b};
}

Decision decision(DecisionKind kind, uint8_t player, Phase phase) {
    Decision d{};
    d.kind = kind;
    d.player = player;
    d.phase = phase;
    return d;
}

// Phases and chance kinds that later PRs fill in land here.
[[noreturn]] void unimplemented(const char* what, int value) {
    fprintf(stderr, "catan: %s %d not implemented\n", what, value);
    abort();
}
[[noreturn]] void unimplemented(Phase phase) { unimplemented("phase", static_cast<int>(phase)); }
[[noreturn]] void unimplemented(ChanceKind kind) {
    unimplemented("chance kind", static_cast<int>(kind));
}

EdgeSet all_roads(const PublicState& pub) {
    EdgeSet all{};
    for (const EdgeSet& r : pub.roads) {
        all[0] |= r[0];
        all[1] |= r[1];
    }
    return all;
}

// Vertices free under the distance rule: unoccupied with no occupied neighbour.
VertexSet distance_legal(const PublicState& pub) {
    const Topology& topo = topology();
    VertexSet occupied = 0, blocked = 0;
    for (int v = 0; v < NUM_VERTICES; ++v) {
        if (!vertex_occupied(pub.vertex[v])) continue;
        occupied |= vertex_bit(v);
        blocked |= topo.vertex_neighbor_mask[v];
    }
    return ALL54 & ~occupied & ~blocked;
}

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

Decision chance_decision(const PublicState& pub) {
    const auto kind = static_cast<ChanceKind>(pub.pending_chance);
    switch (kind) {
        case ChanceKind::NONE:
        case ChanceKind::DICE:
        case ChanceKind::DEV_DRAW:
        case ChanceKind::STEAL: unimplemented(kind);
    }
    unimplemented(kind);
}

Decision setup_decision(const PublicState& pub) {
    const uint8_t p = pub.current_player;
    CATAN_CHECK(pub.setup_step < SETUP_STEPS);
    CATAN_CHECK(p == setup_player(pub.setup_step));

    Decision d = decision(DecisionKind::PLAYER, p, Phase::SETUP);
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
    d.n_legal = static_cast<uint16_t>(count_legal(d.mask, Phase::SETUP));
    return d;
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
        const uint8_t n = gained[r];
        if (n == 0) continue;
        CATAN_CHECK(s.pub.bank[r] >= n);
        s.pub.bank[r] = static_cast<uint8_t>(s.pub.bank[r] - n);
        s.hid.hands[p][r] = static_cast<uint8_t>(s.hid.hands[p][r] + n);
        s.pub.hand_size[p] = static_cast<uint8_t>(s.pub.hand_size[p] + n);
        events.push(event(EventType::GAIN, p, static_cast<uint8_t>(r), n));
    }
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

}  // namespace

Decision next_decision([[maybe_unused]] const Board& board, const GameState& state) {
    const PublicState& pub = state.pub;
    if (pub.pending_chance != 0) return chance_decision(pub);

    switch (pub.phase) {
        case Phase::SETUP: return setup_decision(pub);
        case Phase::TERMINAL:
            return decision(DecisionKind::TERMINAL, NONE, Phase::TERMINAL);
        case Phase::ROLL:
        case Phase::DISCARD:
        case Phase::MOVE_ROBBER:
        case Phase::STEAL:
        case Phase::MAIN:
        case Phase::ROAD_BUILDING:
        case Phase::YOP_PICK:
        case Phase::MONOPOLY_PICK:
        case Phase::EXCHANGE_RECV: unimplemented(pub.phase);
    }
    unimplemented(pub.phase);
}

EventList apply(const Board& board, GameState& state, Action action) {
    CATAN_CHECK(state.pub.pending_chance == 0);
    CATAN_CHECK(is_legal(next_decision(board, state).mask, state.pub.phase, action));

    EventList events;
    switch (state.pub.phase) {
        case Phase::SETUP: apply_setup(board, state, action, events); break;
        case Phase::ROLL:
        case Phase::DISCARD:
        case Phase::MOVE_ROBBER:
        case Phase::STEAL:
        case Phase::MAIN:
        case Phase::ROAD_BUILDING:
        case Phase::YOP_PICK:
        case Phase::MONOPOLY_PICK:
        case Phase::EXCHANGE_RECV:
        case Phase::TERMINAL: unimplemented(state.pub.phase);  // TERMINAL has no actions
    }
    state.pub.ply = static_cast<uint16_t>(state.pub.ply + 1);
    return events;
}

EventList apply_chance([[maybe_unused]] const Board& board, GameState& state,
                       [[maybe_unused]] uint8_t outcome) {
    CATAN_CHECK(state.pub.pending_chance != 0);

    // Cleared before dispatch, so handlers that owe another chance (none yet) can set it again.
    const auto kind = static_cast<ChanceKind>(state.pub.pending_chance);
    state.pub.pending_chance = 0;

    EventList events;
    switch (kind) {
        case ChanceKind::NONE:
        case ChanceKind::DICE:
        case ChanceKind::DEV_DRAW:
        case ChanceKind::STEAL: unimplemented(kind);
    }
    state.pub.ply = static_cast<uint16_t>(state.pub.ply + 1);
    return events;
}

}  // namespace catan
