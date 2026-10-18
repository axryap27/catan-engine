#include "catan/mask.h"

#include <bit>

using namespace std;

namespace catan {
namespace {

// The legal args of one action type as a 128-bit set (lo = bits 0..63, hi = bits 64..127).
// `none` marks heads without an arg: exactly one action, arg 0.
struct Params {
    uint64_t lo = 0;
    uint64_t hi = 0;
    bool none = false;
    int size = 0;  // number of valid arg values for the head
};

Params params_for(const ActionMask& m, Phase phase, ActionType t) {
    switch (param_head(phase, t)) {
        case Head::NONE: return {0, 0, true, 1};
        case Head::VERTEX:
            return {t == ActionType::BUILD_SETTLEMENT ? m.settlement : m.city, 0, false,
                    NUM_VERTICES};
        case Head::EDGE: return {m.edge[0], m.edge[1], false, NUM_EDGES};
        case Head::TILE: return {m.tile, 0, false, NUM_HEXES};
        case Head::DEVCARD: return {m.devcard, 0, false, 8};
        case Head::PLAYER: return {m.player, 0, false, 8};
        case Head::RESOURCE: return {m.resource, 0, false, 8};
    }
    return {};
}

int param_count(const Params& p) {
    return p.none ? 1 : popcount(p.lo) + popcount(p.hi);
}

// Index of the n-th (0-based) set bit of `word`.
int nth_set_bit(uint64_t word, int n) {
    for (int i = 0; i < n; ++i) word &= word - 1;
    return countr_zero(word);
}

}  // namespace

bool is_legal(const ActionMask& mask, Phase phase, Action action) {
    int t = idx(action.type);
    if (t < 0 || t >= NUM_ACTION_TYPES || !type_legal(mask, action.type)) return false;
    Params p = params_for(mask, phase, action.type);
    if (p.none) return action.arg == 0;
    if (action.arg >= p.size) return false;
    uint64_t word = action.arg < 64 ? p.lo : p.hi;
    return (word >> (action.arg % 64)) & 1;
}

int count_legal(const ActionMask& mask, Phase phase) {
    int total = 0;
    for (int t = 0; t < NUM_ACTION_TYPES; ++t) {
        auto type = static_cast<ActionType>(t);
        if (type_legal(mask, type)) total += param_count(params_for(mask, phase, type));
    }
    return total;
}

Action nth_legal(const ActionMask& mask, Phase phase, int n) {
    for (int t = 0; t < NUM_ACTION_TYPES; ++t) {
        auto type = static_cast<ActionType>(t);
        if (!type_legal(mask, type)) continue;
        Params p = params_for(mask, phase, type);
        int c = param_count(p);
        if (n >= c) {
            n -= c;
            continue;
        }
        if (p.none) return {type, 0};
        int lo_count = popcount(p.lo);
        int arg = n < lo_count ? nth_set_bit(p.lo, n) : 64 + nth_set_bit(p.hi, n - lo_count);
        return {type, static_cast<uint8_t>(arg)};
    }
    return {ActionType::END_TURN, 0};  // unreachable given 0 <= n < count_legal
}

}  // namespace catan
