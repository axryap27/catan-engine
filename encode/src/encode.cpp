#include "catan/encode.h"

#include <algorithm>
#include <bit>

#include "catan/check.h"
#include "catan/topology.h"

using namespace std;

namespace catan {
namespace {

static_assert(NUM_PLAYERS == 4, "relative seats use a mask instead of a modulo");
static_assert(sizeof(Phase) == 1 && static_cast<int>(Phase::TERMINAL) == 10,
              "GLOBAL_PHASE holds one flag per Phase");

constexpr int VP_CARD = idx(DevCard::VICTORY_POINT);
constexpr int NUM_TOKENS = static_cast<int>(TOKEN_VALUES.size());

// Seat of absolute seat `seat` as the observer sees it: 0 is the observer, k is k places
// clockwise after it. (Not action.h's relative_from_seat, which skips the decider.)
constexpr uint8_t rel_seat(uint8_t seat, uint8_t observer) {
    return static_cast<uint8_t>((seat + NUM_PLAYERS - observer) & (NUM_PLAYERS - 1));
}

// Position of a dice token in TOKEN_VALUES: 2..6 -> 0..4, 8..12 -> 5..9.
constexpr int token_index(uint8_t token) { return token < 7 ? token - 2 : token - 3; }

constexpr bool token_index_matches() {
    for (int i = 0; i < NUM_TOKENS; ++i) {
        if (token_index(TOKEN_VALUES[i]) != i) return false;
    }
    return true;
}
static_assert(token_index_matches());

// Dice combinations producing `token` (2 and 12: 1, ..., 6 and 8: 5).
constexpr int token_pips(uint8_t token) { return token < 7 ? token - 1 : 13 - token; }

// Hand sizes 0..5 get a bucket each, then 6-7, then 8 and more.
constexpr int hand_bucket(uint8_t count) { return count <= 5 ? count : count <= 7 ? 6 : 7; }

uint8_t saturating_add(uint8_t a, int b) { return static_cast<uint8_t>(min(a + b, 255)); }

// Sets dst[i] for every set bit i of `bits`.
template <typename Word>
void expand_bits(uint8_t* dst, Word bits) {
    while (bits != 0) {
        dst[countr_zero(bits)] = 1;
        bits = static_cast<Word>(bits & (bits - 1));
    }
}

template <typename Word>
constexpr Word low_bits(Word bits, int n) {
    return static_cast<Word>(bits & ((uint64_t{1} << n) - 1));
}

// Relative owner of every road, NONE where an edge is empty.
using EdgeOwners = array<uint8_t, NUM_EDGES>;

EdgeOwners road_owners(const PublicState& pub, uint8_t observer) {
    EdgeOwners owner;
    owner.fill(NONE);
    for (int p = 0; p < NUM_PLAYERS; ++p) {
        const uint8_t rel = rel_seat(static_cast<uint8_t>(p), observer);
        for (int w = 0; w < 2; ++w) {
            uint64_t bits = pub.roads[p][w];
            while (bits != 0) {
                owner[64 * w + countr_zero(bits)] = rel;
                bits &= bits - 1;
            }
        }
    }
    return owner;
}

void encode_tiles(const Board& board, const PublicState& pub, uint8_t observer, uint8_t* out) {
    const Topology& topo = topology();
    const EdgeOwners road = road_owners(pub, observer);

    for (int h = 0; h < NUM_HEXES; ++h) {
        uint8_t* t = out + h * TILE_FEATURES;
        t[TILE_HEX_TYPE + static_cast<int>(board.hex_type[h])] = 1;
        const uint8_t token = board.token[h];
        if (token != 0) {  // the desert has none
            t[TILE_TOKEN + token_index(token)] = 1;
            t[TILE_PIPS] = static_cast<uint8_t>(token_pips(token));
        }
        t[TILE_ROBBER] = pub.robber_hex == h;

        for (int i = 0; i < 6; ++i) {
            const uint8_t vertex = pub.vertex[topo.hex_vertices[h][i]];
            if (vertex_occupied(vertex)) {
                const int rel = rel_seat(vertex_owner(vertex), observer);
                t[TILE_CORNERS + 8 * i + 2 * rel + (vertex_is_city(vertex) ? 1 : 0)] = 1;
            }
            const uint8_t owner = road[topo.hex_edges[h][i]];
            if (owner != NONE) t[TILE_ROADS + 4 * i + owner] = 1;
        }
    }

    // A harbor sits on one coastal edge of one land hex.
    for (int s = 0; s < NUM_PORTS; ++s) {
        const PortSlot& slot = topo.port_slots[s];
        uint8_t* t = out + slot.hex * TILE_FEATURES;
        CATAN_CHECK(all_of(t + TILE_PORT_TYPE, t + TILE_FEATURES,
                           [](uint8_t b) { return b == 0; }));  // at most one harbor per tile
        t[TILE_PORT_TYPE + static_cast<int>(board.port_type[s])] = 1;
        const auto& edges = topo.hex_edges[slot.hex];
        const int side = static_cast<int>(find(edges.begin(), edges.end(), slot.edge) -
                                          edges.begin());
        CATAN_CHECK(side < 6);
        t[TILE_PORT_SIDE + side] = 1;
    }
}

void encode_self(const PublicState& pub, const PrivateView& own, uint8_t observer, uint8_t* out) {
    for (int r = 0; r < NUM_RES; ++r) {
        out[SELF_HAND_BUCKET + HAND_BUCKETS * r + hand_bucket(own.hand[r])] = 1;
        out[SELF_HAND + r] = own.hand[r];
    }
    for (int d = 0; d < NUM_DEV; ++d) {
        out[SELF_DEV_HELD + d] = own.dev_held[d];
        out[SELF_DEV_BOUGHT + d] = own.dev_bought_this_turn[d];
    }
    out[SELF_TOTAL_VP] = static_cast<uint8_t>(pub.public_vp[observer] + own.dev_held[VP_CARD]);
}

void encode_players(const Board& board, const PublicState& pub, const BeliefState& belief,
                    const PrivateView& own, uint8_t observer, uint8_t* out) {
    const Topology& topo = topology();

    for (int k = 0; k < NUM_PLAYERS; ++k) {
        const uint8_t p = static_cast<uint8_t>((observer + k) & (NUM_PLAYERS - 1));
        uint8_t* row = out + k * PLAYER_FEATURES;
        for (int r = 0; r < NUM_RES; ++r) {
            if (k == 0) {  // the observer's own hand is exact whatever the belief says
                row[PLAYER_HAND_MIN + r] = own.hand[r];
                row[PLAYER_HAND_MAX + r] = own.hand[r];
            } else {
                row[PLAYER_HAND_MIN + r] = hand_min(belief, observer, p, r);
                row[PLAYER_HAND_MAX + r] = hand_max(belief, pub, observer, p, r);
            }
        }
        row[PLAYER_HAND_SIZE] = pub.hand_size[p];
        row[PLAYER_DEV_COUNT] = pub.dev_count[p];
        row[PLAYER_KNIGHTS] = pub.knights_played[p];
        row[PLAYER_PUBLIC_VP] = pub.public_vp[p];
        row[PLAYER_ROAD_LEN] = pub.road_len[p];
        for (int i = 0; i < 3; ++i) row[PLAYER_PIECES_LEFT + i] = pub.pieces_left[p][i];
        row[PLAYER_PENDING_DISCARD] = pub.pending_discard[p];
        row[PLAYER_LONGEST_ROAD] = pub.longest_road_holder == p;
        row[PLAYER_LARGEST_ARMY] = pub.largest_army_holder == p;
        row[PLAYER_IS_CURRENT] = pub.current_player == p;
        row[PLAYER_VP_AT_TURN_END] = belief.vp_at_turn_end[p];
    }

    // Harbors and production come from the buildings, so one pass over the occupied vertices
    // fills both.
    for (int v = 0; v < NUM_VERTICES; ++v) {
        const uint8_t vertex = pub.vertex[v];
        if (!vertex_occupied(vertex)) continue;
        uint8_t* row = out + rel_seat(vertex_owner(vertex), observer) * PLAYER_FEATURES;

        const uint8_t port = board.vertex_port[v];
        if (port != NONE) row[PLAYER_PORTS + port] = 1;

        const int yield = vertex_is_city(vertex) ? 2 : 1;
        for (const uint8_t h : topo.vertex_hexes[v]) {
            if (h == NONE) break;  // NONE padding comes last
            if (h == pub.robber_hex || board.token[h] == 0) continue;
            uint8_t& cell = row[PLAYER_PRODUCTION + 10 * static_cast<int>(board.hex_type[h]) +
                                token_index(board.token[h])];
            cell = saturating_add(cell, yield);
        }
    }
}

void encode_global(const PublicState& pub, const BeliefState& belief, uint8_t* out) {
    out[GLOBAL_PHASE + static_cast<int>(pub.phase)] = 1;
    CATAN_CHECK(pub.dice == 0 || (pub.dice >= 2 && pub.dice <= 12));
    if (pub.dice != 0) out[GLOBAL_DICE + pub.dice - 2] = 1;
    for (int r = 0; r < NUM_RES; ++r) out[GLOBAL_BANK + r] = pub.bank[r];
    out[GLOBAL_DEV_DECK] = pub.dev_deck_remaining;
    for (int d = 0; d < NUM_DEV; ++d) out[GLOBAL_DEV_PLAYED + d] = belief.dev_played[d];
    out[GLOBAL_BOUGHT_THIS_TURN] = belief.bought_this_turn;
    out[GLOBAL_PLAYED_DEV] = pub.played_dev_this_turn != 0;
    out[GLOBAL_FREE_ROADS] = pub.free_roads;
    out[GLOBAL_YOP_REMAINING] = pub.yop_remaining;
    if (pub.phase == Phase::EXCHANGE_RECV) {
        CATAN_CHECK(pub.exchange_give < NUM_RES);
        out[GLOBAL_EXCHANGE_GIVE + pub.exchange_give] = 1;
    }
    out[GLOBAL_DISCARD_FLOOR] = pub.discard_floor;
    out[GLOBAL_SETUP_STEP] = pub.setup_step;
    out[GLOBAL_TURN] = static_cast<uint8_t>(min<int>(pub.turn, 255));
}

}  // namespace

PrivateView private_view(const GameState& state, uint8_t observer) {
    PrivateView view{};
    view.hand = state.hid.hands[observer];
    view.dev_held = state.hid.dev_held[observer];
    if (state.pub.current_player == observer) {
        view.dev_bought_this_turn = state.hid.dev_bought_this_turn;
    }
    return view;
}

void encode(const Board& board, const PublicState& pub, const BeliefState& belief,
            const PrivateView& own, uint8_t observer, Observation& out) {
    out.fill(0);
    uint8_t* const base = out.data();
    encode_tiles(board, pub, observer, base + OBS_TILES);
    encode_self(pub, own, observer, base + OBS_SELF);
    encode_players(board, pub, belief, own, observer, base + OBS_PLAYERS);
    encode_global(pub, belief, base + OBS_GLOBAL);
}

void encode(const Board& board, const GameState& state, const BeliefState& belief,
            uint8_t observer, Observation& out) {
    encode(board, state.pub, belief, private_view(state, observer), observer, out);
}

void encode_mask(const Decision& decision, MaskBuffer& out) {
    out.fill(0);
    if (decision.kind != DecisionKind::PLAYER) return;

    const ActionMask& m = decision.mask;
    uint8_t* const base = out.data();
    // Each head keeps only its low `n` bits, so stray bits never spill into the next section.
    expand_bits(base + MASK_TYPE, low_bits(m.type, NUM_ACTION_TYPES));
    expand_bits(base + MASK_SETTLEMENT, low_bits(m.settlement, NUM_VERTICES));
    expand_bits(base + MASK_CITY, low_bits(m.city, NUM_VERTICES));
    expand_bits(base + MASK_EDGE, m.edge[0]);
    expand_bits(base + MASK_EDGE + 64, low_bits(m.edge[1], NUM_EDGES - 64));
    expand_bits(base + MASK_TILE, low_bits(m.tile, NUM_HEXES));
    expand_bits(base + MASK_DEVCARD, low_bits(m.devcard, NUM_DEV));
    // relative to the decider, copied unchanged
    expand_bits(base + MASK_PLAYER, low_bits(m.player, NUM_PLAYERS - 1));
    expand_bits(base + MASK_RESOURCE, low_bits(m.resource, NUM_RES));
}

}  // namespace catan
