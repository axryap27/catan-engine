#include <catch2/catch_test_macros.hpp>
#include <array>

#include "catan/rng.h"

using namespace std;
using namespace catan;

// Verify well-known first output of splitmix64(0)
static_assert(splitmix64(0) == 0xE220A8397B1DCDAF, "splitmix64(0) mismatch");
static_assert(splitmix64(1) == 0x910A2DEC89025CC1, "splitmix64(1) mismatch");

TEST_CASE("rng: golden values", "[rng]") {
    // splitmix64 golden values
    REQUIRE(splitmix64(0) == 0xE220A8397B1DCDAF);
    REQUIRE(splitmix64(1) == 0x910A2DEC89025CC1);

    // random_u64 golden values
    REQUIRE(random_u64(0, 1, 0) == 0x44E5B98100C67FB0);
    REQUIRE(random_u64(42, 2, 7) == 0xEF5E260BC49C3C6F);
    REQUIRE(random_u64(123456789, 3, 1000) == 0x733F5993AAE70CCF);

    // random_below golden value
    REQUIRE(random_below(42, Stream::DICE, 7, 6) == 5);
}

TEST_CASE("rng: random_below uniformity for bound 6", "[rng]") {
    // Draw 60,000 values from random_below with bound 6
    array<int, 6> buckets = {};
    const int total_draws = 60000;

    for (int i = 0; i < total_draws; ++i) {
        uint32_t val = random_below(42, 2, static_cast<uint64_t>(i), 6);
        REQUIRE(val < 6);
        buckets[val]++;
    }

    // Check each bucket is within 10% of the expected 10,000
    for (int i = 0; i < 6; ++i) {
        REQUIRE(buckets[i] >= 9000);
        REQUIRE(buckets[i] <= 11000);
    }
}

TEST_CASE("rng: random_below bounds 1, 2, 6, 19", "[rng]") {
    const int num_draws = 10000;

    // Test bound 1: always 0
    for (int i = 0; i < num_draws; ++i) {
        uint32_t val = random_below(42, 2, static_cast<uint64_t>(i), 1);
        REQUIRE(val == 0);
    }

    // Test bound 2: always 0 or 1
    for (int i = 0; i < num_draws; ++i) {
        uint32_t val = random_below(42, 2, static_cast<uint64_t>(i), 2);
        REQUIRE(val < 2);
    }

    // Test bound 6: already tested thoroughly above
    for (int i = 0; i < num_draws; ++i) {
        uint32_t val = random_below(42, 2, static_cast<uint64_t>(i), 6);
        REQUIRE(val < 6);
    }

    // Test bound 19: always in [0, 19)
    for (int i = 0; i < num_draws; ++i) {
        uint32_t val = random_below(42, 2, static_cast<uint64_t>(i), 19);
        REQUIRE(val < 19);
    }
}

TEST_CASE("rng: different streams with same seed and index", "[rng]") {
    const uint64_t seed = 42;
    const uint64_t index = 7;

    // Different streams should give different values
    uint64_t val1 = random_u64(seed, static_cast<uint64_t>(Stream::BOARD), index);
    uint64_t val2 = random_u64(seed, static_cast<uint64_t>(Stream::DICE), index);
    uint64_t val3 = random_u64(seed, static_cast<uint64_t>(Stream::DEV_DRAW), index);
    uint64_t val4 = random_u64(seed, static_cast<uint64_t>(Stream::STEAL), index);

    REQUIRE(val1 != val2);
    REQUIRE(val1 != val3);
    REQUIRE(val1 != val4);
    REQUIRE(val2 != val3);
    REQUIRE(val2 != val4);
    REQUIRE(val3 != val4);
}
