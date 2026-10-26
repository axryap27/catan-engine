#include "support/reference.h"

#include <algorithm>
#include <vector>

#include "catan/rules.h"
#include "catan/topology.h"

using namespace std;

namespace catan::test {

namespace {

// Extends a trail that has just arrived at `v` over edge list `mine`, marking used edges.
// Passing through v is allowed only if no opponent builds there.
int longest_from(const PublicState& pub, uint8_t p, const vector<int>& mine, vector<bool>& used,
                 int v) {
    const uint8_t b = pub.vertex[v];
    if (vertex_occupied(b) && vertex_owner(b) != p) return 0;
    const Topology& topo = topology();
    int best = 0;
    for (size_t i = 0; i < mine.size(); ++i) {
        if (used[i]) continue;
        const auto& ends = topo.edge_vertices[mine[i]];
        if (ends[0] != v && ends[1] != v) continue;
        used[i] = true;
        best = max(best, 1 + longest_from(pub, p, mine, used, ends[0] == v ? ends[1] : ends[0]));
        used[i] = false;
    }
    return best;
}

// --- reference legality: plain loops over raw state, written from the rules ---

bool owns_road(const PublicState& pub, uint8_t p, int e) { return edge_set_has(pub.roads[p], e); }

bool road_on(const PublicState& pub, int e) {
    for (uint8_t q = 0; q < NUM_PLAYERS; ++q) {
        if (owns_road(pub, q, e)) return true;
    }
    return false;
}

bool has_road_at(const PublicState& pub, uint8_t p, int v) {
    for (uint8_t e : topology().vertex_edges[v]) {
        if (e != NONE && owns_road(pub, p, e)) return true;
    }
    return false;
}

// p may lay a road on e (cost aside): e is free and one end is p's building, or an empty
// vertex reached by p's road. An opponent's building at an end cuts the network there.
bool road_connects(const PublicState& pub, uint8_t p, int e) {
    if (road_on(pub, e)) return false;
    for (uint8_t v : topology().edge_vertices[e]) {
        const uint8_t b = pub.vertex[v];
        if (vertex_occupied(b) && vertex_owner(b) == p) return true;
        if (!vertex_occupied(b) && has_road_at(pub, p, v)) return true;
    }
    return false;
}

bool any_road_connects(const PublicState& pub, uint8_t p) {
    for (int e = 0; e < NUM_EDGES; ++e) {
        if (road_connects(pub, p, e)) return true;
    }
    return false;
}

// Empty vertex with every neighbour empty.
bool distance_ok(const PublicState& pub, int v) {
    if (vertex_occupied(pub.vertex[v])) return false;
    for (uint8_t n : topology().vertex_vertices[v]) {
        if (n != NONE && vertex_occupied(pub.vertex[n])) return false;
    }
    return true;
}

bool affords(const GameState& s, uint8_t p, const array<uint8_t, NUM_RES>& cost) {
    for (int r = 0; r < NUM_RES; ++r) {
        if (s.hid.hands[p][r] < cost[r]) return false;
    }
    return true;
}

bool bank_has_any(const PublicState& pub) {
    for (int r = 0; r < NUM_RES; ++r) {
        if (pub.bank[r] > 0) return true;
    }
    return false;
}

int bank_rate(const Board& board, const PublicState& pub, uint8_t p, int r) {
    int rate = 4;
    for (int v = 0; v < NUM_VERTICES; ++v) {
        if (vertex_owner(pub.vertex[v]) != p || board.vertex_port[v] == NONE) continue;
        if (board.vertex_port[v] == static_cast<uint8_t>(PortType::GENERIC)) rate = min(rate, 3);
        else if (board.vertex_port[v] == r) rate = 2;
    }
    return rate;
}

bool dev_playable(const GameState& s, uint8_t p, int c) {
    const PublicState& pub = s.pub;
    if (c >= NUM_DEV || pub.played_dev_this_turn != 0) return false;
    if (c == idx(DevCard::VICTORY_POINT)) return false;
    if (s.hid.dev_held[p][c] <= s.hid.dev_bought_this_turn[c]) return false;
    if (c == idx(DevCard::ROAD_BUILDING)) {
        return pub.pieces_left[p][PIECE_ROAD] > 0 && any_road_connects(pub, p);
    }
    if (c == idx(DevCard::YEAR_OF_PLENTY)) return bank_has_any(pub);
    return true;
}

// The setup settlement awaiting its road: p's settlement with no road of p's touching it.
int unroaded_settlement(const PublicState& pub, uint8_t p) {
    for (int v = 0; v < NUM_VERTICES; ++v) {
        if (pub.vertex[v] == make_settlement(p) && !has_road_at(pub, p, v)) return v;
    }
    return -1;
}

bool setup_legal(const PublicState& pub, Action a) {
    const uint8_t p = setup_player(pub.setup_step);
    if (pub.setup_step % 2 == 0) {
        return a.type == ActionType::BUILD_SETTLEMENT && a.arg < NUM_VERTICES &&
               distance_ok(pub, a.arg);
    }
    if (a.type != ActionType::BUILD_ROAD || a.arg >= NUM_EDGES || road_on(pub, a.arg)) {
        return false;
    }
    const int v = unroaded_settlement(pub, p);
    const auto& ends = topology().edge_vertices[a.arg];
    return v >= 0 && (ends[0] == v || ends[1] == v);
}

bool main_legal(const Board& board, const GameState& s, Action a) {
    const PublicState& pub = s.pub;
    const uint8_t p = pub.current_player;
    const auto& hand = s.hid.hands[p];
    switch (a.type) {
        case ActionType::END_TURN: return a.arg == 0;
        case ActionType::BUILD_ROAD:
            return a.arg < NUM_EDGES && affords(s, p, ROAD_COST) &&
                   pub.pieces_left[p][PIECE_ROAD] > 0 && road_connects(pub, p, a.arg);
        case ActionType::BUILD_SETTLEMENT:
            return a.arg < NUM_VERTICES && affords(s, p, SETTLEMENT_COST) &&
                   pub.pieces_left[p][PIECE_SETTLEMENT] > 0 && distance_ok(pub, a.arg) &&
                   has_road_at(pub, p, a.arg);
        case ActionType::BUILD_CITY:
            return a.arg < NUM_VERTICES && affords(s, p, CITY_COST) &&
                   pub.pieces_left[p][PIECE_CITY] > 0 && pub.vertex[a.arg] == make_settlement(p);
        case ActionType::BUY_DEV:
            return a.arg == 0 && affords(s, p, DEV_CARD_COST) && pub.dev_deck_remaining > 0;
        case ActionType::PLAY_DEV: return dev_playable(s, p, a.arg);
        case ActionType::BANK_EXCHANGE: {
            if (a.arg >= NUM_RES || hand[a.arg] < bank_rate(board, pub, p, a.arg)) return false;
            for (int r = 0; r < NUM_RES; ++r) {
                if (r != a.arg && pub.bank[r] > 0) return true;
            }
            return false;
        }
        default: return false;
    }
}

bool discard_legal(const GameState& s, Action a) {
    const PublicState& pub = s.pub;
    if (a.type != ActionType::DISCARD || a.arg >= NUM_RES) return false;
    uint8_t q = NONE;
    for (int i = 0; i < NUM_PLAYERS && q == NONE; ++i) {
        const auto seat = static_cast<uint8_t>((pub.current_player + i) % NUM_PLAYERS);
        if (pub.pending_discard[seat] > 0) q = seat;
    }
    if (q == NONE) return false;
    const auto& hand = s.hid.hands[q];
    int at_or_above = 0;
    for (int r = a.arg; r < NUM_RES; ++r) at_or_above += hand[r];
    return a.arg >= pub.discard_floor && hand[a.arg] > 0 &&
           at_or_above >= pub.pending_discard[q];
}

bool steal_legal(const PublicState& pub, Action a) {
    if (a.type != ActionType::STEAL || a.arg >= NUM_PLAYERS - 1) return false;
    const uint8_t me = pub.current_player;
    const uint8_t victim = seat_from_relative(me, a.arg);
    if (victim == me || pub.hand_size[victim] == 0) return false;
    for (uint8_t v : topology().hex_vertices[pub.robber_hex]) {
        if (vertex_owner(pub.vertex[v]) == victim) return true;
    }
    return false;
}

}  // namespace

bool reference_is_legal(const Board& board, const GameState& state, Action action) {
    const PublicState& pub = state.pub;
    const uint8_t p = pub.current_player;
    const Action a = action;
    switch (pub.phase) {
        case Phase::SETUP: return setup_legal(pub, a);
        case Phase::ROLL:
            if (a.type == ActionType::ROLL) return a.arg == 0;
            return a.type == ActionType::PLAY_DEV && dev_playable(state, p, a.arg);
        case Phase::MAIN: return main_legal(board, state, a);
        case Phase::EXCHANGE_RECV:
            return a.type == ActionType::BANK_EXCHANGE && a.arg < NUM_RES &&
                   a.arg != pub.exchange_give && pub.bank[a.arg] > 0;
        case Phase::DISCARD: return discard_legal(state, a);
        case Phase::MOVE_ROBBER:
            return a.type == ActionType::MOVE_ROBBER && a.arg < NUM_HEXES &&
                   a.arg != pub.robber_hex;
        case Phase::STEAL: return steal_legal(pub, a);
        case Phase::ROAD_BUILDING:
            return a.type == ActionType::BUILD_ROAD && a.arg < NUM_EDGES &&
                   pub.pieces_left[p][PIECE_ROAD] > 0 && road_connects(pub, p, a.arg);
        case Phase::YOP_PICK:
            return a.type == ActionType::PLAY_DEV && a.arg < NUM_RES && pub.bank[a.arg] > 0;
        case Phase::MONOPOLY_PICK: return a.type == ActionType::PLAY_DEV && a.arg < NUM_RES;
        case Phase::TERMINAL: return false;
    }
    return false;
}

int reference_road_length(const PublicState& pub, uint8_t p) {
    vector<int> mine;
    for (int e = 0; e < NUM_EDGES; ++e) {
        if (edge_set_has(pub.roads[p], e)) mine.push_back(e);
    }
    // Every trail starts with some edge walked in some direction; the start vertex itself is
    // never passed through, so an opponent building there does not matter.
    const Topology& topo = topology();
    int best = 0;
    vector<bool> used(mine.size(), false);
    for (size_t i = 0; i < mine.size(); ++i) {
        for (int dir = 0; dir < 2; ++dir) {
            used[i] = true;
            const int start = topo.edge_vertices[mine[i]][dir];
            best = max(best, 1 + longest_from(pub, p, mine, used, start));
            used[i] = false;
        }
    }
    return best;
}

}  // namespace catan::test
