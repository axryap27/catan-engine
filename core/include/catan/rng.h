#pragma once
#include <cstdint>

namespace catan {

// Game state never holds RNG state; drivers derive every random outcome from
// (seed, stream, index), so games replay exactly.

enum class Stream : uint64_t { BOARD = 1, DICE = 2, DEV_DRAW = 3, STEAL = 4, RESAMPLE = 5 };

// Standard splitmix64 finalizer.
constexpr uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EB;
    return x ^ (x >> 31);
}

// Combine seed, stream, and index to produce a deterministic u64.
constexpr uint64_t random_u64(uint64_t seed, uint64_t stream, uint64_t index) {
    return splitmix64(splitmix64(splitmix64(seed) ^ stream) ^ index);
}

// Combine seed, stream, and index using Stream enum variant.
constexpr uint64_t random_u64(uint64_t seed, Stream stream, uint64_t index) {
    return random_u64(seed, static_cast<uint64_t>(stream), index);
}

// Return a uniform random value in [0, bound).
// Modulo bias is ~bound / 2^64, negligible for game-sized bounds.
constexpr uint32_t random_below(uint64_t seed, uint64_t stream, uint64_t index, uint32_t bound) {
    return static_cast<uint32_t>(random_u64(seed, stream, index) % bound);
}

// Return a uniform random value in [0, bound) using Stream enum variant.
constexpr uint32_t random_below(uint64_t seed, Stream stream, uint64_t index, uint32_t bound) {
    return random_below(seed, static_cast<uint64_t>(stream), index, bound);
}

}  // namespace catan
