#pragma once
#include <array>
#include <cstdint>

#include "catan/types.h"

// apply() reports what happened as typed events carrying the full truth. Consumers decide who
// may see what: `belief` hides STEAL's resource from non-participants and DEV_BOUGHT's card
// from everyone but the buyer. The same stream is the replay and debug log.

namespace catan {

enum class EventType : uint8_t {
    SETTLEMENT_BUILT,  // player, a = vertex
    CITY_BUILT,        // player, a = vertex
    ROAD_BUILT,        // player, a = edge
    GAIN,              // player, a = resource, b = count; bank -> hand (production, setup, YoP)
    SPEND,             // player, a = resource, b = count; hand -> bank (building, bank trade)
    DISCARD,           // player, a = resource, b = count
    DICE_ROLLED,       // a = sum
    ROBBER_MOVED,      // player, a = hex
    STEAL,             // player = thief, target = victim, a = resource (hidden from others)
    DEV_BOUGHT,        // player, a = DevCard (hidden from others)
    DEV_PLAYED,        // player, a = DevCard
    MONOPOLY,          // player, target = victim, a = resource, b = count
    LONGEST_ROAD,      // player = new holder or NONE
    LARGEST_ARMY,      // player = new holder
    TURN_ENDED,        // player
    GAME_OVER,         // player = winner, NONE if the turn cap ended the game
};

struct Event {
    EventType type;
    uint8_t player = NONE;
    uint8_t target = NONE;
    uint8_t a = 0;
    uint8_t b = 0;
};

// Fixed capacity: the largest single apply is a roll producing for every player.
struct EventList {
    static constexpr int CAPACITY = 32;
    std::array<Event, CAPACITY> items{};
    uint8_t size = 0;

    void push(Event e) { items[size++] = e; }
    const Event* begin() const { return items.data(); }
    const Event* end() const { return items.data() + size; }
};

}  // namespace catan
