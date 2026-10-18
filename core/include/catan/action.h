#pragma once
#include <cstdint>

#include "catan/types.h"

// An action is a type plus at most one parameter. Choices that need two dependent picks
// (Road Building, Year of Plenty, Monopoly, bank exchange, multi-card discards) are split
// into a sequence of decisions, one per pick, so every action stays (type, arg).

namespace catan {

// Order is fixed: it is the type head's output order. New types are only ever appended.
enum class ActionType : uint8_t {
    ROLL,
    BUILD_ROAD,
    BUILD_SETTLEMENT,
    BUILD_CITY,
    BUY_DEV,
    PLAY_DEV,
    BANK_EXCHANGE,
    MOVE_ROBBER,
    STEAL,
    DISCARD,
    END_TURN,
    PROPOSE_TRADE,  // declared, always masked
    RESPOND_TRADE,  // declared, always masked
};
inline constexpr int NUM_ACTION_TYPES = 13;

// Which parameter head an action's `arg` indexes.
enum class Head : uint8_t { NONE, VERTEX, EDGE, TILE, DEVCARD, PLAYER, RESOURCE };

struct Action {
    ActionType type;
    // VERTEX: vertex id. EDGE: edge id. TILE: hex id. DEVCARD: DevCard. RESOURCE: Resource.
    // PLAYER: seat relative to the decider, clockwise: 0 = next seat, 1 = across, 2 = previous.
    // NONE: 0.
    uint8_t arg = 0;
};

constexpr int idx(ActionType t) { return static_cast<int>(t); }

// The parameter head for `type` when chosen in `phase`. Sub-phases keep the type that started
// them: ROAD_BUILDING -> BUILD_ROAD (edge), EXCHANGE_RECV -> BANK_EXCHANGE (resource),
// YOP_PICK and MONOPOLY_PICK -> PLAY_DEV, which there names a resource, not a card.
constexpr Head param_head(Phase phase, ActionType type) {
    if (phase == Phase::YOP_PICK || phase == Phase::MONOPOLY_PICK) return Head::RESOURCE;
    switch (type) {
        case ActionType::BUILD_ROAD: return Head::EDGE;
        case ActionType::BUILD_SETTLEMENT:
        case ActionType::BUILD_CITY: return Head::VERTEX;
        case ActionType::PLAY_DEV: return Head::DEVCARD;
        case ActionType::MOVE_ROBBER: return Head::TILE;
        case ActionType::STEAL: return Head::PLAYER;
        case ActionType::BANK_EXCHANGE:
        case ActionType::DISCARD: return Head::RESOURCE;
        default: return Head::NONE;
    }
}

// Absolute seat for a relative PLAYER arg, and back.
constexpr uint8_t seat_from_relative(uint8_t decider, uint8_t rel) {
    return static_cast<uint8_t>((decider + 1 + rel) % NUM_PLAYERS);
}
constexpr uint8_t relative_from_seat(uint8_t decider, uint8_t seat) {
    return static_cast<uint8_t>((seat + NUM_PLAYERS - decider - 1) % NUM_PLAYERS);
}

}  // namespace catan
