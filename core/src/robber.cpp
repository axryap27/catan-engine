#include <array>

#include "rules_internal.h"

using namespace std;

namespace catan::detail {
namespace {

// First seat owing cards, scanning clockwise from the current player; NONE if nobody owes.
uint8_t discarder(const PublicState& pub) {
    for (int i = 0; i < NUM_PLAYERS; ++i) {
        const auto q = static_cast<uint8_t>((pub.current_player + i) % NUM_PLAYERS);
        if (pub.pending_discard[q] > 0) return q;
    }
    return NONE;
}

// Seats (bitmask) the thief can rob: a building on the robber's hex and a non-empty hand.
uint8_t victims(const PublicState& pub, uint8_t thief) {
    uint8_t mask = 0;
    for (uint8_t v : topology().hex_vertices[pub.robber_hex]) {
        const uint8_t q = vertex_owner(pub.vertex[v]);
        if (q == NONE || q == thief || pub.hand_size[q] == 0) continue;
        mask = static_cast<uint8_t>(mask | (1 << q));
    }
    return mask;
}

}  // namespace

void start_robber(GameState& s) {
    PublicState& pub = s.pub;
    bool owed = false;
    for (int q = 0; q < NUM_PLAYERS; ++q) {
        pub.pending_discard[q] = 0;
        if (pub.hand_size[q] > DISCARD_THRESHOLD) {
            pub.pending_discard[q] = static_cast<uint8_t>(pub.hand_size[q] / 2);
            owed = true;
        }
    }
    pub.discard_floor = 0;
    pub.phase = owed ? Phase::DISCARD : Phase::MOVE_ROBBER;
}

Decision discard_decision(const GameState& s) {
    const PublicState& pub = s.pub;
    const uint8_t q = discarder(pub);
    CATAN_CHECK(q != NONE);

    Decision d = player_decision(q, Phase::DISCARD);
    // One card per decision, non-decreasing by resource so orderings of the same set collapse.
    // Requiring enough cards at or above r keeps the player able to finish in that order.
    int suffix = 0;
    for (int r = NUM_RES - 1; r >= 0; --r) {
        suffix += s.hid.hands[q][r];
        const bool feasible = suffix >= pub.pending_discard[q];
        if (r >= pub.discard_floor && s.hid.hands[q][r] > 0 && feasible) {
            d.mask.resource = static_cast<uint8_t>(d.mask.resource | (1 << r));
        }
    }
    if (d.mask.resource != 0) set_type(d.mask, ActionType::DISCARD);
    return finish(d);
}

void apply_discard(GameState& s, Action action, EventList& events) {
    PublicState& pub = s.pub;
    const uint8_t q = discarder(pub);
    CATAN_CHECK(q != NONE);
    const uint8_t r = action.arg;

    hand_to_bank(s, q, r, 1);
    events.push(event(EventType::DISCARD, q, r, 1));
    if (--pub.pending_discard[q] == 0) {
        pub.discard_floor = 0;
    } else {
        pub.discard_floor = r;
    }
    if (discarder(pub) == NONE) pub.phase = Phase::MOVE_ROBBER;
}

Decision move_robber_decision(const GameState& s) {
    const PublicState& pub = s.pub;
    Decision d = player_decision(pub.current_player, Phase::MOVE_ROBBER);
    d.mask.tile = ((uint32_t{1} << NUM_HEXES) - 1) & ~(uint32_t{1} << pub.robber_hex);
    set_type(d.mask, ActionType::MOVE_ROBBER);
    return finish(d);
}

void apply_move_robber([[maybe_unused]] const Board& board, GameState& s, Action action,
                       EventList& events) {
    PublicState& pub = s.pub;
    pub.robber_hex = action.arg;
    events.push(event(EventType::ROBBER_MOVED, pub.current_player, action.arg));
    pub.phase = victims(pub, pub.current_player) == 0 ? after_robber(pub) : Phase::STEAL;
}

Decision steal_decision([[maybe_unused]] const Board& board, const GameState& s) {
    const PublicState& pub = s.pub;
    const uint8_t me = pub.current_player;
    Decision d = player_decision(me, Phase::STEAL);
    const uint8_t seats = victims(pub, me);
    for (uint8_t q = 0; q < NUM_PLAYERS; ++q) {
        if (!(seats & (1 << q))) continue;
        d.mask.player = static_cast<uint8_t>(d.mask.player | (1 << relative_from_seat(me, q)));
    }
    if (d.mask.player != 0) set_type(d.mask, ActionType::STEAL);
    return finish(d);
}

void apply_steal(GameState& s, Action action, [[maybe_unused]] EventList& events) {
    PublicState& pub = s.pub;
    pub.steal_victim = seat_from_relative(pub.current_player, action.arg);
    pub.pending_chance = static_cast<uint8_t>(ChanceKind::STEAL);
}

ChanceDist steal_dist(const GameState& s) {
    const uint8_t v = s.pub.steal_victim;
    CATAN_CHECK(v != NONE);
    ChanceDist dist;
    dist.kind = ChanceKind::STEAL;
    dist.n_outcomes = NUM_RES;
    dist.total = s.pub.hand_size[v];
    CATAN_CHECK(dist.total > 0);  // pick_outcome divides by total
    for (int r = 0; r < NUM_RES; ++r) dist.weight[r] = s.hid.hands[v][r];
    return dist;
}

void apply_steal_outcome(GameState& s, uint8_t outcome, EventList& events) {
    PublicState& pub = s.pub;
    const uint8_t thief = pub.current_player;
    const uint8_t victim = pub.steal_victim;
    const uint8_t r = outcome;
    CATAN_CHECK(victim != NONE && s.hid.hands[victim][r] > 0);

    // Card goes straight between hands; the bank is not involved.
    --s.hid.hands[victim][r];
    ++s.hid.hands[thief][r];
    --pub.hand_size[victim];
    ++pub.hand_size[thief];
    events.push(event(EventType::STEAL, thief, r, 1, victim));
    pub.steal_victim = NONE;
    pub.phase = after_robber(pub);
}

}  // namespace catan::detail
