#include "catan/chance.h"

#include <cstdlib>

#include "catan/rng.h"

using namespace std;

namespace catan {

uint8_t pick_outcome(const ChanceDist& dist, uint64_t r) {
    // Modulo bias is about total / 2^64, negligible.
    uint64_t x = r % dist.total;
    for (int i = 0; i < dist.n_outcomes; ++i) {
        if (x < dist.weight[i]) return static_cast<uint8_t>(i);
        x -= dist.weight[i];
    }
    return static_cast<uint8_t>(dist.n_outcomes - 1);  // unreachable if total matches weights
}

uint8_t sample_chance(const ChanceDist& dist, const GameState& state, uint64_t seed) {
    Stream stream;
    uint64_t index;
    switch (dist.kind) {
        case ChanceKind::DICE:
            stream = Stream::DICE;
            index = state.pub.turn;
            break;
        case ChanceKind::DEV_DRAW:
            stream = Stream::DEV_DRAW;
            index = static_cast<uint64_t>(DEV_DECK_SIZE - state.pub.dev_deck_remaining);
            break;
        case ChanceKind::STEAL:
            stream = Stream::STEAL;
            index = state.pub.ply;
            break;
        default: abort();  // driver bug: no chance pending
    }
    return pick_outcome(dist, random_u64(seed, stream, index));
}

}  // namespace catan
