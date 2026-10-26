// Uniform-random baseline: picks any legal action with equal probability.
#include <memory>

#include "catan/mask.h"
#include "catan/policy.h"

using namespace std;

namespace catan {

namespace {

class RandomPolicy final : public Policy {
public:
    const char* name() const override { return "random"; }
    Action choose(const Board&, const GameState&, const Decision& d, uint64_t rng) override {
        return nth_legal(d.mask, d.phase, static_cast<int>(rng % d.n_legal));
    }
};

}  // namespace

unique_ptr<Policy> make_random_policy() { return make_unique<RandomPolicy>(); }

}  // namespace catan
