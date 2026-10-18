#pragma once
#include <array>
#include <cstdint>

#include "catan/mask.h"
#include "catan/types.h"

namespace catan {

enum class DecisionKind : uint8_t { PLAYER, CHANCE, TERMINAL };

// Stored in PublicState::pending_chance while a chance outcome is owed.
enum class ChanceKind : uint8_t { NONE, DICE, DEV_DRAW, STEAL };

// Outcome weights for a CHANCE decision. Outcome meaning by kind:
//   DICE: outcome i is a roll of i + 2 (weights 1..6..1 out of 36)
//   DEV_DRAW: outcome is a DevCard, weighted by the deck's remaining counts
//   STEAL: outcome is a Resource, weighted by the victim's hand
struct ChanceDist {
    ChanceKind kind = ChanceKind::NONE;
    uint8_t n_outcomes = 0;
    uint16_t total = 0;  // sum of weights
    std::array<uint8_t, 11> weight{};
};

struct Decision {
    DecisionKind kind;
    uint8_t player = NONE;  // who decides; not always current_player (discards). NONE unless PLAYER
    Phase phase;
    uint16_t n_legal = 0;   // PLAYER only; 1 means forced
    ActionMask mask;        // PLAYER only
    ChanceDist chance;      // CHANCE only
};

}  // namespace catan
