// Greedy heuristic baseline: the first opponent the learned agent has to beat.
//
// Each decision is scored from scratch with a few cheap features; nothing is remembered between
// calls, nothing is allocated. It reads only what its seat may know: the public state, its own
// hand and development cards (the engine's mask already encodes which of them are playable).
//
// The rules, in short:
//   * Vertex value = dice pips of adjacent hexes (robber hex discounted), plus a bonus for each
//     resource the player does not produce yet and for harbors matching its production.
//   * Expansion target = best open settlement spot minus a cost per road needed to reach it,
//     found by BFS over free edges from the player's network.
//   * MAIN, in order: city, settlement, road toward the target, a useful development card,
//     a bank exchange that completes the current goal (or sheds cards when a 7 would hurt),
//     a development card purchase, end turn.
//   * Robber: hurt the most productive leader without touching our own buildings; steal from
//     the leader. Discards and resource picks serve the current goal.
// `rng` is used only to break exact ties.

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>

#include "catan/policy.h"
#include "catan/rng.h"
#include "catan/rules.h"
#include "catan/topology.h"

using namespace std;

namespace catan {
namespace {

using Cost = array<uint8_t, NUM_RES>;
using Hand = array<int, NUM_RES>;

constexpr uint8_t FAR = 0xFF;             // BFS: unreachable vertex
constexpr double ROBBED_HEX = 0.25;       // share of a robbed hex's pips still counted
constexpr double NEW_RESOURCE_BONUS = 3;  // per resource a spot adds that we lack
constexpr double ROAD_COST_PIPS = 2.5;    // a road to a spot costs this much of its value
constexpr int DISCARD_RISK = 8;           // hand size that a 7 would halve
constexpr int MONOPOLY_MIN = 4;           // cards opponents must hold to play Monopoly
// Per-resource weight of a pip in a spot's value: ore and grain build cities and dev cards.
constexpr array<double, NUM_RES> RES_WEIGHT = {1.0, 1.0, 0.8, 1.2, 1.2};
constexpr Cost FALLBACK_NEED = {1, 1, 1, 2, 3};  // settlement + city: what's always useful

// Dice pips of a number token: probability of the roll times 36 (0 for the desert).
constexpr int pips(uint8_t token) {
    if (token == 0) return 0;
    return token < 7 ? token - 1 : 13 - token;
}

bool touches(const EdgeSet& a, const EdgeSet& b) {
    return ((a[0] & b[0]) | (a[1] & b[1])) != 0;
}

bool has_bit(uint64_t set, int i) { return (set >> i) & 1; }

int missing(const Hand& hand, const Cost& cost) {
    int m = 0;
    for (int r = 0; r < NUM_RES; ++r) m += max(0, cost[r] - hand[r]);
    return m;
}

enum class Goal : uint8_t { CITY, SETTLEMENT, ROAD, DEV };

// Everything a decision needs, computed once per call from public data and our own hand.
struct View {
    const Board& board;
    const GameState& s;
    const PublicState& pub;
    const Topology& topo;
    uint8_t me;
    uint64_t rng;

    Hand hand{};
    int hand_total = 0;
    array<Hand, NUM_PLAYERS> prod{};  // pips per resource, cities doubled, robbed hex excluded
    array<int, NUM_RES> rate{};       // our bank exchange rate per resource
    VertexSet open_spots = 0;         // distance-legal vertices
    VertexSet own_settlements = 0;

    // Road planner: roads needed from our network to each vertex, and the edge used to get there.
    array<uint8_t, NUM_VERTICES> dist{};
    array<uint8_t, NUM_VERTICES> via{};

    View(const Board& b, const GameState& st, uint8_t seat, uint64_t r)
        : board(b), s(st), pub(st.pub), topo(topology()), me(seat), rng(r) {
        for (int i = 0; i < NUM_RES; ++i) {
            hand[i] = s.hid.hands[me][i];
            hand_total += hand[i];
        }
        compute_production();
        compute_rates();
        compute_open_spots();
        plan_roads();
    }

    // Deterministic per-candidate tie-breaker in [0, 1e-6).
    double jitter(int arg) const {
        return static_cast<double>(splitmix64(rng ^ (static_cast<uint64_t>(arg) * 0x9E37)) >> 40)
               * 1e-6 / static_cast<double>(1 << 24);
    }

    void compute_production() {
        for (int v = 0; v < NUM_VERTICES; ++v) {
            const uint8_t b = pub.vertex[v];
            if (!vertex_occupied(b)) continue;
            const int mult = vertex_is_city(b) ? 2 : 1;
            for (uint8_t h : topo.vertex_hexes[v]) {
                if (h == NONE || h == pub.robber_hex || board.hex_type[h] == HexType::DESERT) {
                    continue;
                }
                prod[vertex_owner(b)][static_cast<int>(board.hex_type[h])] +=
                    mult * pips(board.token[h]);
            }
            if (b == make_settlement(me)) own_settlements |= vertex_bit(v);
        }
    }

    // Same rule as the engine: 2:1 for an owned resource harbor, 3:1 generic, else 4:1.
    void compute_rates() {
        rate.fill(4);
        bool generic = false;
        for (int v = 0; v < NUM_VERTICES; ++v) {
            const uint8_t port = board.vertex_port[v];
            if (port == NONE || vertex_owner(pub.vertex[v]) != me) continue;
            if (port == static_cast<uint8_t>(PortType::GENERIC)) generic = true;
            else rate[port] = 2;
        }
        if (generic) {
            for (int& r : rate) r = min(r, 3);
        }
    }

    void compute_open_spots() {
        VertexSet occupied = 0, blocked = 0;
        for (int v = 0; v < NUM_VERTICES; ++v) {
            if (!vertex_occupied(pub.vertex[v])) continue;
            occupied |= vertex_bit(v);
            blocked |= topo.vertex_neighbor_mask[v];
        }
        open_spots = ((uint64_t{1} << NUM_VERTICES) - 1) & ~occupied & ~blocked;
    }

    bool opponent_at(int v) const {
        const uint8_t o = vertex_owner(pub.vertex[v]);
        return o != NONE && o != me;
    }

    // BFS over unroaded edges, never through an opponent's building. Sources (distance 0) are
    // our buildings and the free vertices our roads reach, exactly where the engine lets us build.
    void plan_roads() {
        dist.fill(FAR);
        via.fill(NONE);
        EdgeSet taken{};
        for (const EdgeSet& r : pub.roads) {
            taken[0] |= r[0];
            taken[1] |= r[1];
        }
        array<uint8_t, NUM_VERTICES> queue{};
        int head = 0, tail = 0;
        for (int v = 0; v < NUM_VERTICES; ++v) {
            const uint8_t o = vertex_owner(pub.vertex[v]);
            if (o == me || (o == NONE && touches(topo.vertex_edge_mask[v], pub.roads[me]))) {
                dist[v] = 0;
                queue[tail++] = static_cast<uint8_t>(v);
            }
        }
        while (head < tail) {
            const int v = queue[head++];
            for (uint8_t e : topo.vertex_edges[v]) {
                if (e == NONE || edge_set_has(taken, e)) continue;
                const auto& ends = topo.edge_vertices[e];
                const int u = ends[0] == v ? ends[1] : ends[0];
                if (dist[u] != FAR || opponent_at(u)) continue;
                dist[u] = static_cast<uint8_t>(dist[v] + 1);
                via[u] = e;
                queue[tail++] = static_cast<uint8_t>(u);
            }
        }
    }

    // The road to build first on the planned path to `target` (dist[target] >= 1).
    uint8_t first_road(int target) const {
        int v = target;
        while (dist[v] > 1) {
            const auto& ends = topo.edge_vertices[via[v]];
            v = ends[0] == v ? ends[1] : ends[0];
        }
        return via[v];
    }

    // How much a settlement at v would add for us.
    double vertex_value(int v) const {
        Hand gain{};
        for (uint8_t h : topo.vertex_hexes[v]) {
            if (h == NONE || board.hex_type[h] == HexType::DESERT) continue;
            const double p = pips(board.token[h]) * (h == pub.robber_hex ? ROBBED_HEX : 1.0);
            gain[static_cast<int>(board.hex_type[h])] += static_cast<int>(p * 4);  // quarter pips
        }
        double value = 0;
        for (int r = 0; r < NUM_RES; ++r) {
            value += gain[r] / 4.0 * RES_WEIGHT[r];
            // Diversity: a resource we cannot roll for is worth far more than a duplicate.
            if (gain[r] > 0 && prod[me][r] == 0) value += NEW_RESOURCE_BONUS;
        }
        // Harbors: a 2:1 port pays off with production of its resource, a 3:1 a little always.
        const uint8_t port = board.vertex_port[v];
        if (port == static_cast<uint8_t>(PortType::GENERIC)) {
            value += 1.0;
        } else if (port != NONE) {
            value += 0.3 * min(prod[me][port] + gain[port] / 4, 10);
        }
        return value;
    }

    // Production a city at our settlement v would add (its own pips, robbed hex excluded).
    double city_value(int v) const {
        double value = 0;
        for (uint8_t h : topo.vertex_hexes[v]) {
            if (h == NONE || h == pub.robber_hex || board.hex_type[h] == HexType::DESERT) continue;
            value += pips(board.token[h]);
        }
        return value;
    }

    // Best open spot, scored by value minus the roads it costs. min_dist 1 skips spots we can
    // settle already (used to aim free roads). Returns NONE if no spot is reachable.
    int best_target(int min_dist) const {
        int best = NONE;
        double best_score = -1e9;
        for (int v = 0; v < NUM_VERTICES; ++v) {
            if (!has_bit(open_spots, v) || dist[v] == FAR || dist[v] < min_dist) continue;
            const double score = vertex_value(v) - ROAD_COST_PIPS * dist[v] + jitter(v);
            if (score > best_score) {
                best_score = score;
                best = v;
            }
        }
        return best;
    }

    int my_vp() const {
        return pub.public_vp[me] + s.hid.dev_held[me][idx(DevCard::VICTORY_POINT)];
    }

    // Cards of r all opponents hold together: everything not in the bank or our hand.
    int opponents_hold(int r) const { return RESOURCE_PER_TYPE - pub.bank[r] - hand[r]; }
};

struct Plan {
    Goal goal;
    Cost cost;
    int target;  // best expansion spot, NONE if none
};

// What we are saving for. Settlement or city, whichever is closer (settlement on ties: it
// grows production and the board presence); a road when the next spot needs one; else a
// development card.
Plan make_plan(const View& w) {
    const auto& pieces = w.pub.pieces_left[w.me];
    const int target = pieces[PIECE_SETTLEMENT] > 0 ? w.best_target(0) : NONE;
    const bool city_ok = pieces[PIECE_CITY] > 0 && w.own_settlements != 0;
    const bool settle_ok = target != NONE && w.dist[target] == 0;
    const bool road_ok = target != NONE && w.dist[target] > 0 && pieces[PIECE_ROAD] > 0;

    const int miss_city = city_ok ? missing(w.hand, CITY_COST) : 99;
    const int miss_settle = settle_ok ? missing(w.hand, SETTLEMENT_COST) : 99;
    if (settle_ok && miss_settle <= miss_city) return {Goal::SETTLEMENT, SETTLEMENT_COST, target};
    if (city_ok && (miss_city <= 1 || !road_ok)) return {Goal::CITY, CITY_COST, target};
    if (road_ok) return {Goal::ROAD, ROAD_COST, target};
    if (city_ok) return {Goal::CITY, CITY_COST, target};
    return {Goal::DEV, DEV_CARD_COST, target};
}

// Cards beyond what the goal needs.
Hand surplus(const View& w, const Cost& cost) {
    Hand s{};
    for (int r = 0; r < NUM_RES; ++r) s[r] = max(0, w.hand[r] - cost[r]);
    return s;
}

// Among the resources in `mask`, the one maximizing score(r) (ties by rng); NONE if empty.
template <class F>
uint8_t pick_resource(const View& w, uint8_t mask, F score) {
    uint8_t best = NONE;
    double best_score = -1e18;
    for (int r = 0; r < NUM_RES; ++r) {
        if (!has_bit(mask, r)) continue;
        const double sc = score(r) + w.jitter(r);
        if (sc > best_score) {
            best_score = sc;
            best = static_cast<uint8_t>(r);
        }
    }
    return best;
}

// The resource we most need next: the goal's shortfall first, then the general need.
double need_score(const View& w, const Cost& cost, int r) {
    return 10.0 * (cost[r] - w.hand[r]) + (FALLBACK_NEED[r] - w.hand[r]);
}

// ---- SETUP -------------------------------------------------------------------------------

Action setup_settlement(const View& w, const ActionMask& m) {
    int best = NONE;
    double best_score = -1e9;
    for (int v = 0; v < NUM_VERTICES; ++v) {
        if (!has_bit(m.settlement, v)) continue;
        const double score = w.vertex_value(v) + w.jitter(v);
        if (score > best_score) {
            best_score = score;
            best = v;
        }
    }
    return {ActionType::BUILD_SETTLEMENT, static_cast<uint8_t>(best)};
}

// Point the setup road at the best spot two or three roads away.
Action setup_road(const View& w, const ActionMask& m) {
    int best = NONE;
    double best_score = -1e9;
    for (int e = 0; e < NUM_EDGES; ++e) {
        if (!edge_set_has(m.edge, e)) continue;
        const auto& ends = w.topo.edge_vertices[e];
        const int u = vertex_occupied(w.pub.vertex[ends[0]]) ? ends[1] : ends[0];
        double score = 0;
        for (uint8_t n1 : w.topo.vertex_vertices[u]) {
            if (n1 == NONE) continue;
            if (has_bit(w.open_spots, n1)) score = max(score, w.vertex_value(n1));
            for (uint8_t n2 : w.topo.vertex_vertices[n1]) {
                if (n2 == NONE || !has_bit(w.open_spots, n2)) continue;
                score = max(score, w.vertex_value(n2) - ROAD_COST_PIPS);
            }
        }
        score += w.jitter(e);
        if (score > best_score) {
            best_score = score;
            best = e;
        }
    }
    return {ActionType::BUILD_ROAD, static_cast<uint8_t>(best)};
}

// ---- Robber ------------------------------------------------------------------------------

bool robber_hurts_me(const View& w) {
    const uint8_t h = w.pub.robber_hex;
    if (w.board.hex_type[h] == HexType::DESERT) return false;
    for (uint8_t v : w.topo.hex_vertices[h]) {
        if (vertex_owner(w.pub.vertex[v]) == w.me) return true;
    }
    return false;
}

// A Knight now takes or keeps Largest Army.
bool knight_wins_army(const View& w) {
    const int mine = w.pub.knights_played[w.me] + 1;
    if (mine < 3 || w.pub.largest_army_holder == w.me) return false;
    for (int q = 0; q < NUM_PLAYERS; ++q) {
        if (q != w.me && w.pub.knights_played[q] >= mine) return false;
    }
    return true;
}

// Opponent q's weight as a robber target: leaders matter more.
double threat(const View& w, int q) { return 1.0 + 0.5 * w.pub.public_vp[q]; }

// Block the hex costing opponents the most production, weighted toward the leader; never
// one touching our own buildings unless every hex does.
Action move_robber(const View& w, const ActionMask& m) {
    int best = NONE;
    double best_score = -1e18;
    for (int h = 0; h < NUM_HEXES; ++h) {
        if (!has_bit(m.tile, h)) continue;
        const int p = pips(w.board.token[h]);
        double score = 0;
        bool mine = false, victim = false;
        for (uint8_t v : w.topo.hex_vertices[h]) {
            const uint8_t b = w.pub.vertex[v];
            const uint8_t o = vertex_owner(b);
            if (o == NONE) continue;
            if (o == w.me) {
                mine = true;
                continue;
            }
            score += p * (vertex_is_city(b) ? 2 : 1) * threat(w, o);
            if (w.pub.hand_size[o] > 0) victim = true;
        }
        if (victim) score += 3;  // a steal is worth about a card
        if (mine) score -= 1000;
        score += w.jitter(h);
        if (score > best_score) {
            best_score = score;
            best = h;
        }
    }
    return {ActionType::MOVE_ROBBER, static_cast<uint8_t>(best)};
}

// Rob the leader; on a tie the one with more cards.
Action steal(const View& w, const ActionMask& m) {
    int best = NONE;
    double best_score = -1e18;
    for (int rel = 0; rel < NUM_PLAYERS - 1; ++rel) {
        if (!has_bit(m.player, rel)) continue;
        const uint8_t q = seat_from_relative(w.me, static_cast<uint8_t>(rel));
        const double score = 100.0 * w.pub.public_vp[q] + w.pub.hand_size[q] + w.jitter(rel);
        if (score > best_score) {
            best_score = score;
            best = rel;
        }
    }
    return {ActionType::STEAL, static_cast<uint8_t>(best)};
}

// ---- Development cards -------------------------------------------------------------------

// Resource Monopoly would take the most of, valuing what the goal needs a bit higher.
uint8_t monopoly_choice(const View& w, const Cost& cost) {
    return pick_resource(w, 0x1F, [&](int r) {
        const int n = w.opponents_hold(r);
        return n + (cost[r] > w.hand[r] ? 1.0 : 0.0) + (w.rate[r] == 2 ? 1.0 : 0.0);
    });
}

// The card worth playing now, or NONE. One per turn, so the most timely one wins:
// Year of Plenty when it completes the goal, Monopoly on a well-stocked resource,
// Road Building toward a spot, otherwise a Knight.
uint8_t dev_to_play(const View& w, const Plan& plan, uint8_t playable) {
    auto can = [&](DevCard c) { return has_bit(playable, idx(c)); };
    const int miss = missing(w.hand, plan.cost);
    if (can(DevCard::YEAR_OF_PLENTY) && miss >= 1 && miss <= 2) {
        return idx(DevCard::YEAR_OF_PLENTY);
    }
    if (can(DevCard::MONOPOLY) && w.opponents_hold(monopoly_choice(w, plan.cost)) >= MONOPOLY_MIN) {
        return idx(DevCard::MONOPOLY);
    }
    if (can(DevCard::ROAD_BUILDING) && w.best_target(1) != NONE) {
        return idx(DevCard::ROAD_BUILDING);
    }
    if (can(DevCard::KNIGHT)) return idx(DevCard::KNIGHT);
    return NONE;
}

// ---- MAIN --------------------------------------------------------------------------------

Action main_action(const View& w, const ActionMask& m) {
    const Plan plan = make_plan(w);

    // 1. City on the settlement producing the most.
    if (type_legal(m, ActionType::BUILD_CITY)) {
        int best = NONE;
        double best_score = -1;
        for (int v = 0; v < NUM_VERTICES; ++v) {
            if (!has_bit(m.city, v)) continue;
            const double score = w.city_value(v) + w.jitter(v);
            if (score > best_score) {
                best_score = score;
                best = v;
            }
        }
        return {ActionType::BUILD_CITY, static_cast<uint8_t>(best)};
    }

    // 2. Settlement on the most valuable reachable spot.
    if (type_legal(m, ActionType::BUILD_SETTLEMENT)) {
        int best = NONE;
        double best_score = -1e9;
        for (int v = 0; v < NUM_VERTICES; ++v) {
            if (!has_bit(m.settlement, v)) continue;
            const double score = w.vertex_value(v) + w.jitter(v);
            if (score > best_score) {
                best_score = score;
                best = v;
            }
        }
        return {ActionType::BUILD_SETTLEMENT, static_cast<uint8_t>(best)};
    }

    // 3. Road along the planned path when the best spot is not reachable yet.
    if (type_legal(m, ActionType::BUILD_ROAD) && plan.target != NONE && w.dist[plan.target] > 0) {
        const uint8_t e = w.first_road(plan.target);
        if (edge_set_has(m.edge, e)) return {ActionType::BUILD_ROAD, e};
    }

    // 4. A development card that helps right now.
    if (type_legal(m, ActionType::PLAY_DEV)) {
        const uint8_t card = dev_to_play(w, plan, m.devcard);
        if (card != NONE) return {ActionType::PLAY_DEV, card};
    }

    // 5. Bank exchange: give the biggest tradeable surplus when the trades available finish
    //    the goal this turn, or when a 7 would cost us half a big hand.
    if (type_legal(m, ActionType::BANK_EXCHANGE)) {
        const Hand extra = surplus(w, plan.cost);
        const int miss = missing(w.hand, plan.cost);
        int trades = 0;
        bool bank_ok = true;
        for (int r = 0; r < NUM_RES; ++r) {
            trades += extra[r] / w.rate[r];
            if (plan.cost[r] > w.hand[r] && w.pub.bank[r] == 0) bank_ok = false;
        }
        const bool completes = miss > 0 && trades >= miss && bank_ok;
        const bool shed = w.hand_total >= DISCARD_RISK && trades > 0;
        if (completes || shed) {
            const uint8_t give = pick_resource(w, m.resource, [&](int r) {
                return extra[r] >= w.rate[r] ? extra[r] / static_cast<double>(w.rate[r]) : -1e9;
            });
            if (give != NONE && extra[give] >= w.rate[give]) {
                return {ActionType::BANK_EXCHANGE, give};
            }
        }
    }

    // 6. Development card whenever nothing better was affordable. Saving its cards for the
    //    goal instead measured worse: Knights, VP cards and the robber pay off steadily.
    if (type_legal(m, ActionType::BUY_DEV)) return {ActionType::BUY_DEV, 0};

    return {ActionType::END_TURN, 0};
}

// ---- Dispatcher --------------------------------------------------------------------------

Action decide(const View& w, const Decision& d) {
    const ActionMask& m = d.mask;
    switch (d.phase) {
        case Phase::SETUP:
            if (type_legal(m, ActionType::BUILD_SETTLEMENT)) return setup_settlement(w, m);
            return setup_road(w, m);

        case Phase::ROLL:
            // Knight first if the robber sits on us or the Knight wins Largest Army.
            if (type_legal(m, ActionType::PLAY_DEV) && has_bit(m.devcard, idx(DevCard::KNIGHT)) &&
                (robber_hurts_me(w) || knight_wins_army(w))) {
                return {ActionType::PLAY_DEV, static_cast<uint8_t>(idx(DevCard::KNIGHT))};
            }
            return {ActionType::ROLL, 0};

        case Phase::MAIN: return main_action(w, m);

        case Phase::DISCARD: {
            // Drop what the goal needs least: the largest excess, then the largest pile.
            const Plan plan = make_plan(w);
            const uint8_t r = pick_resource(w, m.resource, [&](int r) {
                return 10.0 * (w.hand[r] - plan.cost[r]) + w.hand[r] - 0.1 * FALLBACK_NEED[r];
            });
            return {ActionType::DISCARD, r};
        }

        case Phase::MOVE_ROBBER: return move_robber(w, m);
        case Phase::STEAL: return steal(w, m);

        case Phase::ROAD_BUILDING: {
            // Free roads toward the best spot not yet reachable.
            const int target = w.best_target(1);
            if (target != NONE) {
                const uint8_t e = w.first_road(target);
                if (edge_set_has(m.edge, e)) return {ActionType::BUILD_ROAD, e};
            }
            break;  // no target: any legal road
        }

        case Phase::YOP_PICK: {
            const Plan plan = make_plan(w);
            const uint8_t r = pick_resource(w, m.resource,
                                            [&](int r) { return need_score(w, plan.cost, r); });
            return {ActionType::PLAY_DEV, r};
        }

        case Phase::MONOPOLY_PICK:
            return {ActionType::PLAY_DEV, monopoly_choice(w, make_plan(w).cost)};

        case Phase::EXCHANGE_RECV: {
            // The give cards are still in hand here; they were surplus, so the plan is unchanged.
            const Plan plan = make_plan(w);
            const uint8_t r = pick_resource(w, m.resource,
                                            [&](int r) { return need_score(w, plan.cost, r); });
            return {ActionType::BANK_EXCHANGE, r};
        }

        default: break;
    }
    return nth_legal(m, d.phase, static_cast<int>(w.rng % d.n_legal));
}

class GreedyPolicy final : public Policy {
public:
    const char* name() const override { return "greedy"; }

    Action choose(const Board& board, const GameState& state, const Decision& d,
                  uint64_t rng) override {
        const View w(board, state, d.player, rng);
        const Action a = decide(w, d);
        // Safety net: every rule above picks from the mask, but never return an illegal move.
        if (is_legal(d.mask, d.phase, a)) return a;
        return nth_legal(d.mask, d.phase, static_cast<int>(rng % d.n_legal));
    }
};

}  // namespace

unique_ptr<Policy> make_greedy_policy() { return make_unique<GreedyPolicy>(); }

}  // namespace catan
