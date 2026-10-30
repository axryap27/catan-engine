#pragma once
#include <cstdint>
#include <string>

#include "catan/action.h"
#include "catan/board.h"
#include "catan/event.h"
#include "catan/state.h"

// Human-readable views of a game, for the CLI tools, the Python bindings and debugging. Plain
// ASCII with no color codes, deterministic, so output can be diffed and pinned in tests.
// Players are written P0..P3 (absolute seats).

namespace catan {

struct RenderOptions {
    bool vertex_ids = false;  // label every empty corner with its vertex id
    bool edge_ids = false;    // label every empty road spot with its edge id
};

// The board as an ASCII map of pointy-top hexes laid out as in topology.h (rows of 3, 4, 5,
// 4, 3). Each hex shows its resource and dice number, and the robber when present. Each corner
// shows a settlement or city with its owner, each edge a road with its owner, and each harbor
// its type next to its two coastal corners. Every line has the same width; no trailing
// whitespace beyond that width.
std::string render_board(const Board& board, const PublicState& pub, RenderOptions opts = {});

// render_board, then the turn state (turn, phase, current player, last roll, pending discards,
// bank, dev deck size) and one panel per player: VP, hand size, resources, dev cards, knights,
// road length, pieces left, awards. `observer` == NONE shows every hand and dev card (god
// view); otherwise only the observer's own resources and dev cards are shown, and opponents
// show counts only.
std::string render_state(const Board& board, const GameState& state, uint8_t observer = NONE,
                         RenderOptions opts = {});

// One action in words, e.g. "build settlement at v17", "steal from P2", "discard grain",
// "trade grain to the bank", "receive ore", "take grain" (Year of Plenty), "end turn".
// `decider` turns the relative PLAYER arg into an absolute seat. Distinct legal actions of one
// decision always get distinct descriptions.
std::string describe_action(Action action, Phase phase, uint8_t decider);

// One event in words, e.g. "P0 builds a settlement at v12". `observer` hides what that seat
// could not see: a STEAL's resource unless the observer is thief or victim, a DEV_BOUGHT's card
// unless the observer bought it. NONE shows the full truth.
std::string describe_event(const Event& event, uint8_t observer = NONE);

// Lower-case names: "brick", "knight", "main", "build_settlement", "desert", "3:1", ...
const char* resource_name(int resource);
const char* dev_card_name(int card);
const char* phase_name(Phase phase);
const char* action_type_name(ActionType type);
const char* hex_type_name(HexType type);
const char* port_type_name(PortType type);  // "2:1 brick" ... "3:1"

}  // namespace catan
