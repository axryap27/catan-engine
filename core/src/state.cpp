#include "catan/state.h"

namespace catan {

GameState initial_state(uint8_t robber_hex) {
    GameState s{};
    PublicState& p = s.pub;

    p.bank.fill(RESOURCE_PER_TYPE);
    for (auto& pieces : p.pieces_left) {
        pieces[PIECE_ROAD] = ROADS_PER_PLAYER;
        pieces[PIECE_SETTLEMENT] = SETTLEMENTS_PER_PLAYER;
        pieces[PIECE_CITY] = CITIES_PER_PLAYER;
    }
    p.dev_deck_remaining = DEV_DECK_SIZE;
    p.robber_hex = robber_hex;
    p.phase = Phase::SETUP;
    p.longest_road_holder = NONE;
    p.largest_army_holder = NONE;
    p.exchange_give = NONE;
    p.trade_target = NONE;

    s.hid.dev_deck = DEV_DECK_COUNTS;
    return s;
}

}  // namespace catan
