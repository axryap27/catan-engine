#include <algorithm>
#include <array>

#include "rules_internal.h"

using namespace std;

namespace catan::detail {
namespace {

inline constexpr int LONGEST_ROAD_MIN = 5;
inline constexpr int LARGEST_ARMY_MIN = 3;

// Longest trail extending from v over p's unused edges. Recursion depth is bounded by the
// 15 roads a player owns, so the stack stays small and nothing is allocated.
int extend_trail(const PublicState& pub, uint8_t p, int v, EdgeSet& used, bool at_start) {
    // An opponent's building ends the trail here: it may be an endpoint, never a pass-through.
    const uint8_t owner = vertex_owner(pub.vertex[v]);
    if (!at_start && owner != NONE && owner != p) return 0;

    const Topology& topo = topology();
    int best = 0;
    for (uint8_t e : topo.vertex_edges[v]) {
        if (e == NONE || !edge_set_has(pub.roads[p], e) || edge_set_has(used, e)) continue;
        const auto& ends = topo.edge_vertices[e];
        const int next = ends[0] == v ? ends[1] : ends[0];
        edge_set_add(used, e);
        best = max(best, 1 + extend_trail(pub, p, next, used, false));
        used[e / 64] &= ~(uint64_t{1} << (e % 64));
    }
    return best;
}

void move_award(PublicState& pub, uint8_t& holder, uint8_t to) {
    if (holder != NONE) {
        pub.public_vp[holder] = static_cast<uint8_t>(pub.public_vp[holder] - 2);
    }
    if (to != NONE) pub.public_vp[to] = static_cast<uint8_t>(pub.public_vp[to] + 2);
    holder = to;
}

}  // namespace

uint8_t road_length(const PublicState& pub, uint8_t p) {
    const Topology& topo = topology();
    const EdgeSet& roads = pub.roads[p];
    if (roads[0] == 0 && roads[1] == 0) return 0;

    // The longest trail starts at some vertex touched by p's roads; try each one.
    VertexSet starts = 0;
    for (int e = 0; e < NUM_EDGES; ++e) {
        if (!edge_set_has(roads, e)) continue;
        starts |= vertex_bit(topo.edge_vertices[e][0]) | vertex_bit(topo.edge_vertices[e][1]);
    }
    int best = 0;
    for (int v = 0; v < NUM_VERTICES; ++v) {
        if ((starts & vertex_bit(v)) == 0) continue;
        EdgeSet used{};
        best = max(best, extend_trail(pub, p, v, used, true));
    }
    return static_cast<uint8_t>(best);
}

void update_longest_road(GameState& s, EventList& events) {
    PublicState& pub = s.pub;
    int best = 0;
    for (int q = 0; q < NUM_PLAYERS; ++q) {
        pub.road_len[q] = road_length(pub, static_cast<uint8_t>(q));
        best = max(best, static_cast<int>(pub.road_len[q]));
    }

    const uint8_t holder = pub.longest_road_holder;
    uint8_t next = NONE;
    if (best >= LONGEST_ROAD_MIN) {
        if (holder != NONE && pub.road_len[holder] == best) {
            next = holder;  // a tie never takes the card from its holder
        } else {
            int leaders = 0;
            for (int q = 0; q < NUM_PLAYERS; ++q) {
                if (pub.road_len[q] != best) continue;
                ++leaders;
                next = static_cast<uint8_t>(q);
            }
            // Tied non-holders (after the holder's road was cut): the card is set aside.
            if (leaders != 1) next = NONE;
        }
    }

    if (next == holder) return;
    move_award(pub, pub.longest_road_holder, next);
    events.push(event(EventType::LONGEST_ROAD, next, 0));
}

void update_largest_army(GameState& s, EventList& events) {
    PublicState& pub = s.pub;
    const uint8_t p = pub.current_player;
    const uint8_t holder = pub.largest_army_holder;
    if (pub.knights_played[p] < LARGEST_ARMY_MIN || p == holder) return;
    if (holder != NONE && pub.knights_played[p] <= pub.knights_played[holder]) return;

    move_award(pub, pub.largest_army_holder, p);
    events.push(event(EventType::LARGEST_ARMY, p, 0));
}

}  // namespace catan::detail
