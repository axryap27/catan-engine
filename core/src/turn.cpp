#include <algorithm>
#include <array>

#include "rules_internal.h"

using namespace std;

namespace catan::detail {
namespace {

bool touches(const EdgeSet& a, const EdgeSet& b) {
    return ((a[0] & b[0]) | (a[1] & b[1])) != 0;
}

// Bank trade rate per resource for p: 4:1, 3:1 with a generic harbor, 2:1 with that resource's.
array<uint8_t, NUM_RES> exchange_rates(const Board& board, const PublicState& pub, uint8_t p) {
    array<uint8_t, NUM_RES> rate;
    rate.fill(4);
    bool generic = false;
    for (int v = 0; v < NUM_VERTICES; ++v) {
        const uint8_t port = board.vertex_port[v];
        if (port == NONE || vertex_owner(pub.vertex[v]) != p) continue;
        if (port == static_cast<uint8_t>(PortType::GENERIC)) generic = true;
        else rate[port] = 2;
    }
    if (generic) {
        for (uint8_t& r : rate) r = min<uint8_t>(r, 3);
    }
    return rate;
}

// Pays out a non-7 roll. Demand is totalled first so a short bank is judged per resource.
void produce(const Board& board, GameState& s, EventList& events) {
    const PublicState& pub = s.pub;
    const Topology& topo = topology();
    array<array<uint8_t, NUM_RES>, NUM_PLAYERS> demand{};
    for (uint8_t h : board.hexes_for_roll[pub.dice]) {
        if (h == NONE || h == pub.robber_hex) continue;
        const int r = static_cast<int>(board.hex_type[h]);
        for (uint8_t v : topo.hex_vertices[h]) {
            const uint8_t b = pub.vertex[v];
            if (!vertex_occupied(b)) continue;
            uint8_t& owed = demand[vertex_owner(b)][r];
            owed = static_cast<uint8_t>(owed + (vertex_is_city(b) ? 2 : 1));
        }
    }

    // A short bank pays a sole claimant what is left, and nobody when two or more are owed.
    array<array<uint8_t, NUM_RES>, NUM_PLAYERS> grant{};
    for (int r = 0; r < NUM_RES; ++r) {
        int total = 0, claimants = 0;
        uint8_t sole = NONE;
        for (uint8_t p = 0; p < NUM_PLAYERS; ++p) {
            if (demand[p][r] == 0) continue;
            total += demand[p][r];
            ++claimants;
            sole = p;
        }
        if (total <= pub.bank[r]) {
            for (int p = 0; p < NUM_PLAYERS; ++p) grant[p][r] = demand[p][r];
        } else if (claimants == 1) {
            grant[sole][r] = pub.bank[r];
        }
    }

    for (uint8_t p = 0; p < NUM_PLAYERS; ++p) {
        for (int r = 0; r < NUM_RES; ++r) {
            if (grant[p][r] == 0) continue;
            bank_to_hand(s, p, r, grant[p][r]);
            events.push(event(EventType::GAIN, p, static_cast<uint8_t>(r), grant[p][r]));
        }
    }
}

void end_turn(GameState& s, EventList& events) {
    PublicState& pub = s.pub;
    events.push(event(EventType::TURN_ENDED, pub.current_player, 0));
    pub.turn = static_cast<uint16_t>(pub.turn + 1);
    if (pub.turn >= TURN_CAP) {
        pub.phase = Phase::TERMINAL;
        events.push(event(EventType::GAME_OVER, NONE, 0));
        return;
    }
    pub.current_player = static_cast<uint8_t>((pub.current_player + 1) % NUM_PLAYERS);
    pub.phase = Phase::ROLL;
    pub.dice = 0;  // resume_phase reads 0 as "not rolled yet"
    pub.played_dev_this_turn = 0;
    pub.trades_this_turn = 0;
    s.hid.dev_bought_this_turn.fill(0);
}

}  // namespace

EdgeSet buildable_roads(const PublicState& pub, uint8_t p) {
    // An opponent's building on a vertex cuts p's network there.
    const Topology& topo = topology();
    EdgeSet reach{};
    for (int v = 0; v < NUM_VERTICES; ++v) {
        const EdgeSet& inc = topo.vertex_edge_mask[v];
        const uint8_t owner = vertex_owner(pub.vertex[v]);
        if (owner == p || (owner == NONE && touches(inc, pub.roads[p]))) {
            reach[0] |= inc[0];
            reach[1] |= inc[1];
        }
    }
    const EdgeSet taken = all_roads(pub);
    return {reach[0] & ~taken[0], reach[1] & ~taken[1]};
}

Decision roll_decision(const GameState& s) {
    Decision d = player_decision(s.pub.current_player, Phase::ROLL);
    set_type(d.mask, ActionType::ROLL);
    // A development card may be played before rolling (usually a Knight).
    d.mask.devcard = playable_dev_cards(s);
    if (d.mask.devcard != 0) set_type(d.mask, ActionType::PLAY_DEV);
    return finish(d);
}

void apply_roll(GameState& s, Action action, EventList& events) {
    if (action.type == ActionType::PLAY_DEV) {
        apply_play_dev(s, action.arg, events);
        return;
    }
    // DICE_ROLLED is emitted by apply_dice once the outcome arrives.
    s.pub.pending_chance = static_cast<uint8_t>(ChanceKind::DICE);
}

ChanceDist dice_dist() {
    ChanceDist c;
    c.kind = ChanceKind::DICE;
    c.n_outcomes = 11;
    c.weight = {1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    c.total = 36;
    return c;
}

void apply_dice(const Board& board, GameState& s, uint8_t outcome, EventList& events) {
    PublicState& pub = s.pub;
    pub.dice = static_cast<uint8_t>(outcome + 2);
    events.push(event(EventType::DICE_ROLLED, pub.current_player, pub.dice));
    if (pub.dice == 7) {
        start_robber(s);
        return;
    }
    produce(board, s, events);
    pub.phase = Phase::MAIN;
}

Decision main_decision(const Board& board, const GameState& s) {
    const PublicState& pub = s.pub;
    const Topology& topo = topology();
    const uint8_t p = pub.current_player;
    const EdgeSet& roads = pub.roads[p];
    Decision d = player_decision(p, Phase::MAIN);
    set_type(d.mask, ActionType::END_TURN);

    VertexSet own_settlements = 0, road_touched = 0;
    for (int v = 0; v < NUM_VERTICES; ++v) {
        if (touches(topo.vertex_edge_mask[v], roads)) road_touched |= vertex_bit(v);
        if (pub.vertex[v] == make_settlement(p)) own_settlements |= vertex_bit(v);
    }

    const auto& pieces = pub.pieces_left[p];
    if (can_afford(s, p, ROAD_COST) && pieces[PIECE_ROAD] > 0) {
        d.mask.edge = buildable_roads(pub, p);
        if ((d.mask.edge[0] | d.mask.edge[1]) != 0) set_type(d.mask, ActionType::BUILD_ROAD);
    }
    if (can_afford(s, p, SETTLEMENT_COST) && pieces[PIECE_SETTLEMENT] > 0) {
        d.mask.settlement = distance_legal(pub) & road_touched;
        if (d.mask.settlement != 0) set_type(d.mask, ActionType::BUILD_SETTLEMENT);
    }
    if (can_afford(s, p, CITY_COST) && pieces[PIECE_CITY] > 0) {
        d.mask.city = own_settlements;
        if (d.mask.city != 0) set_type(d.mask, ActionType::BUILD_CITY);
    }
    if (can_afford(s, p, DEV_CARD_COST) && pub.dev_deck_remaining > 0) {
        set_type(d.mask, ActionType::BUY_DEV);
    }
    d.mask.devcard = playable_dev_cards(s);
    if (d.mask.devcard != 0) set_type(d.mask, ActionType::PLAY_DEV);

    uint8_t bank_has = 0;
    for (int r = 0; r < NUM_RES; ++r) {
        if (pub.bank[r] > 0) bank_has = static_cast<uint8_t>(bank_has | (1u << r));
    }
    const auto rate = exchange_rates(board, pub, p);
    for (int r = 0; r < NUM_RES; ++r) {
        if (s.hid.hands[p][r] >= rate[r] && (bank_has & ~(1u << r)) != 0) {
            d.mask.resource = static_cast<uint8_t>(d.mask.resource | (1u << r));
        }
    }
    if (d.mask.resource != 0) set_type(d.mask, ActionType::BANK_EXCHANGE);
    return finish(d);
}

void apply_main(const Board&, GameState& s, Action action, EventList& events) {
    PublicState& pub = s.pub;
    const uint8_t p = pub.current_player;
    const uint8_t arg = action.arg;
    switch (action.type) {
        case ActionType::BUILD_ROAD:
            pay(s, p, ROAD_COST, events);
            edge_set_add(pub.roads[p], arg);
            --pub.pieces_left[p][PIECE_ROAD];
            events.push(event(EventType::ROAD_BUILT, p, arg));
            update_longest_road(s, events);
            break;
        case ActionType::BUILD_SETTLEMENT:
            pay(s, p, SETTLEMENT_COST, events);
            pub.vertex[arg] = make_settlement(p);
            --pub.pieces_left[p][PIECE_SETTLEMENT];
            ++pub.public_vp[p];
            events.push(event(EventType::SETTLEMENT_BUILT, p, arg));
            update_longest_road(s, events);  // a settlement can cut an opponent's road
            break;
        case ActionType::BUILD_CITY:
            pay(s, p, CITY_COST, events);
            pub.vertex[arg] = make_city(p);
            --pub.pieces_left[p][PIECE_CITY];
            ++pub.pieces_left[p][PIECE_SETTLEMENT];  // the replaced settlement returns to supply
            ++pub.public_vp[p];
            events.push(event(EventType::CITY_BUILT, p, arg));
            break;
        case ActionType::BANK_EXCHANGE:
            // Step 1 of 2: only the resource given is chosen; cards move on the receive pick.
            pub.exchange_give = arg;
            pub.phase = Phase::EXCHANGE_RECV;
            break;
        case ActionType::BUY_DEV: apply_buy_dev(s, events); break;
        case ActionType::PLAY_DEV: apply_play_dev(s, arg, events); break;
        case ActionType::END_TURN: end_turn(s, events); break;
        default: CATAN_CHECK(false);
    }
}

Decision exchange_recv_decision(const Board&, const GameState& s) {
    const PublicState& pub = s.pub;
    Decision d = player_decision(pub.current_player, Phase::EXCHANGE_RECV);
    for (int r = 0; r < NUM_RES; ++r) {
        if (r == pub.exchange_give || pub.bank[r] == 0) continue;
        d.mask.resource = static_cast<uint8_t>(d.mask.resource | (1u << r));
    }
    if (d.mask.resource != 0) set_type(d.mask, ActionType::BANK_EXCHANGE);
    return finish(d);
}

void apply_exchange_recv(const Board& board, GameState& s, Action action, EventList& events) {
    PublicState& pub = s.pub;
    const uint8_t p = pub.current_player;
    const uint8_t give = pub.exchange_give;
    const uint8_t rate = exchange_rates(board, pub, p)[give];
    hand_to_bank(s, p, give, rate);
    events.push(event(EventType::SPEND, p, give, rate));
    bank_to_hand(s, p, action.arg, 1);
    events.push(event(EventType::GAIN, p, action.arg, 1));
    pub.exchange_give = NONE;
    pub.phase = Phase::MAIN;
}

// Only the current player is checked: a player can win only on their own turn. Run after
// END_TURN too, so 10 points reached on someone else's turn win as the new turn begins.
void check_victory(GameState& s, EventList& events) {
    PublicState& pub = s.pub;
    if (pub.phase == Phase::TERMINAL) return;
    const uint8_t p = pub.current_player;
    const int vp = pub.public_vp[p] + s.hid.dev_held[p][idx(DevCard::VICTORY_POINT)];
    if (vp < VICTORY_POINTS_TO_WIN) return;
    pub.phase = Phase::TERMINAL;
    events.push(event(EventType::GAME_OVER, p, 0));
}

}  // namespace catan::detail
