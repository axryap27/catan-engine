#pragma once
#include <cstdint>

#include "catan/action.h"
#include "catan/types.h"

// Legal-move masks as bitsets. `core` computes them with every decision; `encode` expands them
// to floats. A type bit is set only if that type has at least one legal parameter.

namespace catan {

struct ActionMask {
    uint16_t type = 0;          // bit t: ActionType t is legal
    EdgeSet edge{};             // BUILD_ROAD
    VertexSet settlement = 0;   // BUILD_SETTLEMENT
    VertexSet city = 0;         // BUILD_CITY (same vertex head, different mask)
    uint32_t tile = 0;          // MOVE_ROBBER, bit per hex
    uint8_t devcard = 0;        // PLAY_DEV, bit per DevCard
    uint8_t player = 0;         // STEAL, bit per relative seat
    uint8_t resource = 0;       // the type using the resource head in this phase, bit per Resource
};

constexpr bool type_legal(const ActionMask& m, ActionType t) { return (m.type >> idx(t)) & 1; }
constexpr void set_type(ActionMask& m, ActionType t) {
    m.type = static_cast<uint16_t>(m.type | (1u << idx(t)));
}

// `phase` selects the parameter head (see param_head).
bool is_legal(const ActionMask& mask, Phase phase, Action action);

// Number of distinct legal (type, arg) actions. 1 means the move is forced.
int count_legal(const ActionMask& mask, Phase phase);

// The n-th legal action, 0 <= n < count_legal, in canonical order: by type, then arg ascending.
// A uniform n gives a uniform random legal action.
Action nth_legal(const ActionMask& mask, Phase phase, int n);

}  // namespace catan
