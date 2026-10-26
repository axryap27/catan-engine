#include <algorithm>
#include <array>

#include "rules_internal.h"

using namespace std;

namespace catan::detail {
namespace {

bool any_edge(const EdgeSet& e) { return (e[0] | e[1]) != 0; }

uint8_t bank_resources(const PublicState& pub) {
    uint8_t mask = 0;
    for (int r = 0; r < NUM_RES; ++r) {
        if (pub.bank[r] > 0) mask = static_cast<uint8_t>(mask | (1u << r));
    }
    return mask;
}

}  // namespace

uint8_t playable_dev_cards(const GameState& s) {
    const PublicState& pub = s.pub;
    if (pub.played_dev_this_turn != 0) return 0;
    const uint8_t p = pub.current_player;

    uint8_t mask = 0;
    for (int c = 0; c < NUM_DEV; ++c) {
        if (c == idx(DevCard::VICTORY_POINT)) continue;
        // A copy bought this turn is locked, but an older copy of the same type is not.
        if (s.hid.dev_held[p][c] <= s.hid.dev_bought_this_turn[c]) continue;

        bool effective = true;
        switch (static_cast<DevCard>(c)) {
            case DevCard::ROAD_BUILDING:
                effective = pub.pieces_left[p][PIECE_ROAD] > 0 &&
                            any_edge(buildable_roads(pub, p));
                break;
            case DevCard::YEAR_OF_PLENTY: effective = bank_resources(pub) != 0; break;
            default: break;
        }
        if (effective) mask = static_cast<uint8_t>(mask | (1u << c));
    }
    return mask;
}

void apply_buy_dev(GameState& s, EventList& events) {
    pay(s, s.pub.current_player, DEV_CARD_COST, events);
    s.pub.pending_chance = static_cast<uint8_t>(ChanceKind::DEV_DRAW);
}

ChanceDist dev_draw_dist(const GameState& s) {
    ChanceDist dist;
    dist.kind = ChanceKind::DEV_DRAW;
    dist.n_outcomes = NUM_DEV;
    [[maybe_unused]] int sum = 0;  // read only by CATAN_CHECK
    for (int c = 0; c < NUM_DEV; ++c) {
        dist.weight[c] = s.hid.dev_deck[c];
        sum += s.hid.dev_deck[c];
    }
    dist.total = s.pub.dev_deck_remaining;
    CATAN_CHECK(dist.total > 0);  // pick_outcome divides by total
    CATAN_CHECK(sum == dist.total);
    return dist;
}

void apply_dev_draw(GameState& s, uint8_t outcome, EventList& events) {
    PublicState& pub = s.pub;
    const uint8_t p = pub.current_player;
    const uint8_t c = outcome;
    CATAN_CHECK(c < NUM_DEV && s.hid.dev_deck[c] > 0);

    --s.hid.dev_deck[c];
    --pub.dev_deck_remaining;
    ++s.hid.dev_held[p][c];
    ++pub.dev_count[p];
    ++s.hid.dev_bought_this_turn[c];
    events.push(event(EventType::DEV_BOUGHT, p, c));
}

void apply_play_dev(GameState& s, uint8_t card, EventList& events) {
    PublicState& pub = s.pub;
    const uint8_t p = pub.current_player;
    CATAN_CHECK(card < NUM_DEV && s.hid.dev_held[p][card] > 0);

    --s.hid.dev_held[p][card];
    --pub.dev_count[p];
    pub.played_dev_this_turn = 1;
    events.push(event(EventType::DEV_PLAYED, p, card));

    switch (static_cast<DevCard>(card)) {
        case DevCard::KNIGHT:
            ++pub.knights_played[p];
            update_largest_army(s, events);
            pub.phase = Phase::MOVE_ROBBER;
            break;
        case DevCard::ROAD_BUILDING:
            pub.free_roads = min<uint8_t>(2, pub.pieces_left[p][PIECE_ROAD]);
            pub.phase = Phase::ROAD_BUILDING;
            break;
        case DevCard::YEAR_OF_PLENTY:
            pub.yop_remaining = 2;
            pub.phase = Phase::YOP_PICK;
            break;
        case DevCard::MONOPOLY: pub.phase = Phase::MONOPOLY_PICK; break;
        default: CATAN_CHECK(false);
    }
}

Decision road_building_decision(const GameState& s) {
    const PublicState& pub = s.pub;
    const uint8_t p = pub.current_player;
    Decision d = player_decision(p, Phase::ROAD_BUILDING);
    d.mask.edge = buildable_roads(pub, p);
    if (any_edge(d.mask.edge)) set_type(d.mask, ActionType::BUILD_ROAD);
    return finish(d);
}

void apply_road_building(GameState& s, Action action, EventList& events) {
    PublicState& pub = s.pub;
    const uint8_t p = pub.current_player;
    const uint8_t e = action.arg;
    CATAN_CHECK(pub.free_roads > 0 && pub.pieces_left[p][PIECE_ROAD] > 0);

    edge_set_add(pub.roads[p], e);
    --pub.pieces_left[p][PIECE_ROAD];
    events.push(event(EventType::ROAD_BUILT, p, e));
    update_longest_road(s, events);
    --pub.free_roads;

    // End early when nothing more can be placed, so no decision is left without legal moves.
    if (pub.free_roads == 0 || pub.pieces_left[p][PIECE_ROAD] == 0 ||
        !any_edge(buildable_roads(pub, p))) {
        pub.free_roads = 0;
        pub.phase = resume_phase(pub);
    }
}

Decision yop_decision(const GameState& s) {
    Decision d = player_decision(s.pub.current_player, Phase::YOP_PICK);
    d.mask.resource = bank_resources(s.pub);
    if (d.mask.resource != 0) set_type(d.mask, ActionType::PLAY_DEV);
    return finish(d);
}

void apply_yop(GameState& s, Action action, EventList& events) {
    PublicState& pub = s.pub;
    const uint8_t p = pub.current_player;
    const uint8_t r = action.arg;

    bank_to_hand(s, p, r, 1);
    events.push(event(EventType::GAIN, p, r, 1));
    --pub.yop_remaining;
    if (pub.yop_remaining == 0 || bank_resources(pub) == 0) {
        pub.yop_remaining = 0;
        pub.phase = resume_phase(pub);
    }
}

Decision monopoly_decision(const GameState& s) {
    Decision d = player_decision(s.pub.current_player, Phase::MONOPOLY_PICK);
    d.mask.resource = static_cast<uint8_t>((1u << NUM_RES) - 1);
    set_type(d.mask, ActionType::PLAY_DEV);
    return finish(d);
}

void apply_monopoly(GameState& s, Action action, EventList& events) {
    PublicState& pub = s.pub;
    const uint8_t p = pub.current_player;
    const uint8_t r = action.arg;

    for (int i = 1; i < NUM_PLAYERS; ++i) {
        const auto q = static_cast<uint8_t>((p + i) % NUM_PLAYERS);
        const uint8_t n = s.hid.hands[q][r];
        s.hid.hands[q][r] = 0;
        s.hid.hands[p][r] = static_cast<uint8_t>(s.hid.hands[p][r] + n);
        pub.hand_size[q] = static_cast<uint8_t>(pub.hand_size[q] - n);
        pub.hand_size[p] = static_cast<uint8_t>(pub.hand_size[p] + n);
        // Emitted even for n == 0: beliefs learn that q holds none of r.
        events.push(event(EventType::MONOPOLY, p, r, n, q));
    }
    pub.phase = resume_phase(pub);
}

}  // namespace catan::detail
