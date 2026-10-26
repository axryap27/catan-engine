#include "catan/invariants.h"

#include <bit>

#include "rules_internal.h"

using namespace std;

namespace catan {

namespace {

constexpr int ALL_PHASES = static_cast<int>(Phase::TERMINAL);
constexpr int KNIGHT_TOTAL = DEV_DECK_COUNTS[0];

const char* check_resources(const PublicState& pub, const HiddenState& hid) {
    for (int r = 0; r < NUM_RES; ++r) {
        int total = pub.bank[r];
        for (int p = 0; p < NUM_PLAYERS; ++p) total += hid.hands[p][r];
        if (total != RESOURCE_PER_TYPE) return "resource conservation";
    }
    for (int p = 0; p < NUM_PLAYERS; ++p) {
        int n = 0;
        for (int r = 0; r < NUM_RES; ++r) n += hid.hands[p][r];
        if (n != pub.hand_size[p]) return "hand_size mismatch";
    }
    return nullptr;
}

const char* check_dev(const PublicState& pub, const HiddenState& hid) {
    int deck = 0;
    for (int c = 0; c < NUM_DEV; ++c) deck += hid.dev_deck[c];
    if (deck != pub.dev_deck_remaining) return "dev_deck_remaining mismatch";

    int held_by_type[NUM_DEV] = {};
    int played = 0;
    for (int p = 0; p < NUM_PLAYERS; ++p) {
        int n = 0;
        for (int c = 0; c < NUM_DEV; ++c) {
            n += hid.dev_held[p][c];
            held_by_type[c] += hid.dev_held[p][c];
        }
        if (n != pub.dev_count[p]) return "dev_count mismatch";
        played += pub.knights_played[p];
    }
    for (int c = 0; c < NUM_DEV; ++c) {
        const int in_play = hid.dev_deck[c] + held_by_type[c];
        if (c == idx(DevCard::KNIGHT)) {
            if (in_play + played != KNIGHT_TOTAL) return "knight conservation";
        } else if (in_play > DEV_DECK_COUNTS[c]) {
            return "dev card conservation";
        }
    }
    const uint8_t cur = pub.current_player;
    for (int c = 0; c < NUM_DEV; ++c) {
        if (hid.dev_bought_this_turn[c] > hid.dev_held[cur][c]) return "dev_bought_this_turn";
    }
    return nullptr;
}

const char* check_board(const PublicState& pub) {
    const Topology& topo = topology();
    int roads[NUM_PLAYERS] = {}, settlements[NUM_PLAYERS] = {}, cities[NUM_PLAYERS] = {};
    int vp[NUM_PLAYERS] = {};
    for (int v = 0; v < NUM_VERTICES; ++v) {
        const uint8_t b = pub.vertex[v];
        if (!vertex_occupied(b)) {
            if (b != 0) return "malformed vertex";
            continue;
        }
        if ((b >> 4) != 0) return "malformed vertex";
        for (uint8_t n : topo.vertex_vertices[v]) {
            if (n != NONE && vertex_occupied(pub.vertex[n])) return "distance rule";
        }
        const uint8_t o = vertex_owner(b);
        if (vertex_is_city(b)) {
            ++cities[o];
            vp[o] += 2;
        } else {
            ++settlements[o];
            vp[o] += 1;
        }
    }

    EdgeSet seen{};
    for (int p = 0; p < NUM_PLAYERS; ++p) {
        const EdgeSet& r = pub.roads[p];
        if (r[1] >> (NUM_EDGES - 64)) return "edge out of range";
        if ((seen[0] & r[0]) || (seen[1] & r[1])) return "road owned twice";
        seen[0] |= r[0];
        seen[1] |= r[1];
        roads[p] = popcount(r[0]) + popcount(r[1]);
    }
    for (int p = 0; p < NUM_PLAYERS; ++p) {
        if (pub.pieces_left[p][PIECE_ROAD] + roads[p] != ROADS_PER_PLAYER) return "road pieces";
        if (pub.pieces_left[p][PIECE_SETTLEMENT] + settlements[p] != SETTLEMENTS_PER_PLAYER) {
            return "settlement pieces";
        }
        if (pub.pieces_left[p][PIECE_CITY] + cities[p] != CITIES_PER_PLAYER) return "city pieces";
        if (pub.longest_road_holder == p) vp[p] += 2;
        if (pub.largest_army_holder == p) vp[p] += 2;
        if (pub.public_vp[p] != vp[p]) return "public_vp mismatch";
    }
    return nullptr;
}

const char* check_awards(const PublicState& pub) {
    for (uint8_t p = 0; p < NUM_PLAYERS; ++p) {
        if (pub.road_len[p] != detail::road_length(pub, p)) return "road_len stale";
    }
    const uint8_t lr = pub.longest_road_holder;
    if (lr != NONE) {
        if (lr >= NUM_PLAYERS) return "longest road holder out of range";
        if (pub.road_len[lr] < 5) return "longest road holder below 5";
        for (int p = 0; p < NUM_PLAYERS; ++p) {
            if (pub.road_len[p] > pub.road_len[lr]) return "longest road holder not longest";
        }
    }
    const uint8_t la = pub.largest_army_holder;
    if (la != NONE) {
        if (la >= NUM_PLAYERS) return "largest army holder out of range";
        if (pub.knights_played[la] < 3) return "largest army holder below 3 knights";
        for (int p = 0; p < NUM_PLAYERS; ++p) {
            if (pub.knights_played[p] > pub.knights_played[la]) {
                return "largest army holder not largest";
            }
        }
    } else {
        for (int p = 0; p < NUM_PLAYERS; ++p) {
            if (pub.knights_played[p] >= 3) return "largest army unclaimed";
        }
    }
    return nullptr;
}

const char* check_phase(const PublicState& pub) {
    if (static_cast<int>(pub.phase) > ALL_PHASES) return "phase out of range";
    if (pub.current_player >= NUM_PLAYERS) return "current_player out of range";
    if (pub.robber_hex >= NUM_HEXES) return "robber_hex out of range";
    if (pub.pending_chance > static_cast<uint8_t>(ChanceKind::STEAL)) {
        return "pending_chance out of range";
    }
    const bool steal_pending = pub.pending_chance == static_cast<uint8_t>(ChanceKind::STEAL);
    if (steal_pending != (pub.steal_victim != NONE)) return "steal_victim mismatch";
    if (steal_pending && pub.steal_victim >= NUM_PLAYERS) return "steal_victim out of range";
    if (steal_pending && pub.hand_size[pub.steal_victim] == 0) return "steal from empty hand";
    if (pub.pending_chance == static_cast<uint8_t>(ChanceKind::DEV_DRAW) &&
        pub.dev_deck_remaining == 0) {
        return "dev draw from empty deck";
    }

    int owed = 0;
    for (int p = 0; p < NUM_PLAYERS; ++p) owed += pub.pending_discard[p];
    if (pub.phase != Phase::DISCARD && owed != 0) return "pending_discard outside DISCARD";
    if (pub.phase == Phase::DISCARD && owed == 0) return "DISCARD with nobody owing";
    if (pub.phase != Phase::ROAD_BUILDING && pub.free_roads != 0) {
        return "free_roads outside ROAD_BUILDING";
    }
    if (pub.phase != Phase::YOP_PICK && pub.yop_remaining != 0) {
        return "yop_remaining outside YOP_PICK";
    }
    if ((pub.phase == Phase::EXCHANGE_RECV) != (pub.exchange_give != NONE)) {
        return "exchange_give mismatch";
    }
    if (pub.exchange_give != NONE && pub.exchange_give >= NUM_RES) {
        return "exchange_give out of range";
    }

    switch (pub.phase) {
        case Phase::SETUP:
            if (pub.setup_step >= SETUP_STEPS) return "setup_step out of range";
            if (pub.dice != 0) return "dice set during SETUP";
            break;
        case Phase::ROLL:
            if (pub.dice != 0) return "dice set before roll";
            break;
        case Phase::MAIN:
        case Phase::EXCHANGE_RECV:
        case Phase::DISCARD:
            if (pub.dice < 2 || pub.dice > 12) return "dice out of range";
            break;
        default:
            if (pub.dice == 1 || pub.dice > 12) return "dice out of range";
            break;
    }
    return nullptr;
}

const char* check_liveness(const Board& board, const GameState& state) {
    const Decision d = next_decision(board, state);
    if (d.kind == DecisionKind::PLAYER) {
        if (d.player >= NUM_PLAYERS) return "decision player out of range";
        if (d.n_legal < 1 || d.mask.type == 0) return "no legal action";
    } else if (d.kind == DecisionKind::CHANCE) {
        if (d.chance.total == 0) return "empty chance distribution";
    }
    return nullptr;
}

}  // namespace

const char* invariant_violation(const Board& board, const GameState& state) {
    const PublicState& pub = state.pub;
    if (const char* why = check_phase(pub)) return why;
    if (const char* why = check_resources(pub, state.hid)) return why;
    if (const char* why = check_dev(pub, state.hid)) return why;
    if (const char* why = check_board(pub)) return why;
    if (const char* why = check_awards(pub)) return why;
    if (pub.phase == Phase::TERMINAL) return nullptr;
    return check_liveness(board, state);
}

}  // namespace catan
