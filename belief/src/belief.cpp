#include "catan/belief.h"

#include <algorithm>
#include <array>
#include <utility>

#include "catan/check.h"
#include "catan/rng.h"
#include "catan/rules.h"

using namespace std;

namespace catan {
namespace {

constexpr int MAX_TRIES = 64;
constexpr int UNCAPPED = 99;
constexpr int VP = idx(DevCard::VICTORY_POINT);

// Removes n cards of a public type r: known ones first, the rest were unknown.
void remove_typed(HandBelief& h, int r, int n) {
    const int k = min<int>(h.known[r], n);
    h.known[r] = static_cast<uint8_t>(h.known[r] - k);
    h.unknown = static_cast<uint8_t>(h.unknown - (n - k));
}

// Number of resource types with a known card, and the last such type.
int known_types(const HandBelief& h, int& last) {
    int m = 0;
    for (int r = 0; r < NUM_RES; ++r) {
        if (h.known[r] == 0) continue;
        ++m;
        last = r;
    }
    return m;
}

// One card of unseen type left the hand. Any known type may have lost a card, so each known
// count drops by one; the cards that were not taken become unknown.
void remove_untyped(HandBelief& h) {
    int dropped = 0;
    for (int r = 0; r < NUM_RES; ++r) {
        if (h.known[r] == 0) continue;
        --h.known[r];
        ++dropped;
    }
    h.unknown = static_cast<uint8_t>(h.unknown + dropped - 1);
}

void observe_steal(BeliefState& b, uint8_t thief, uint8_t victim, int r) {
    for (int o = 0; o < NUM_PLAYERS; ++o) {
        HandBelief& v = b.hand[o][victim];
        HandBelief& t = b.hand[o][thief];
        int only = 0;
        if (o == thief || o == victim) {
            remove_typed(v, r, 1);
            ++t.known[r];
        } else if (v.unknown == 0 && known_types(v, only) == 1) {
            remove_typed(v, only, 1);  // a one-type hand: the card's type is deduced
            ++t.known[only];
        } else {
            remove_untyped(v);
            ++t.unknown;
        }
    }
}

// Exchange rate p pays for resource r: 4:1, 3:1 with a generic harbor, 2:1 with r's harbor.
int exchange_rate(const Board& board, const PublicState& pub, uint8_t p, int r) {
    int rate = 4;
    for (int v = 0; v < NUM_VERTICES; ++v) {
        const uint8_t port = board.vertex_port[v];
        if (port == NONE || vertex_owner(pub.vertex[v]) != p) continue;
        rate = min(rate, port == static_cast<uint8_t>(PortType::GENERIC) ? 3 : port == r ? 2 : 4);
    }
    return rate;
}

// A publicly implied lower bound on one opponent's hand: `player` holds at least `count` cards
// whose type is in the bitmask `types`.
struct HandNeed {
    uint8_t player = NONE;
    uint8_t types = 0;
    int count = 0;
};

HandNeed hand_need(const Board& board, const PublicState& pub, uint8_t observer) {
    HandNeed need;
    if (pub.phase == Phase::DISCARD) {
        // The discarder is the first seat owing, clockwise from the current player.
        for (int i = 0; i < NUM_PLAYERS && need.player == NONE; ++i) {
            const auto q = static_cast<uint8_t>((pub.current_player + i) % NUM_PLAYERS);
            if (pub.pending_discard[q] > 0) need.player = q;
        }
        need.types = static_cast<uint8_t>(((1u << NUM_RES) - 1) & ~((1u << pub.discard_floor) - 1));
        if (need.player != NONE) need.count = pub.pending_discard[need.player];
    } else if (pub.phase == Phase::EXCHANGE_RECV) {
        need.player = pub.current_player;
        need.types = static_cast<uint8_t>(1u << pub.exchange_give);
        need.count = exchange_rate(board, pub, need.player, pub.exchange_give);
    }
    if (need.player == observer) need = HandNeed{};
    return need;
}

struct Rng {
    uint64_t seed;
    uint64_t counter = 0;
    uint32_t below(uint32_t n) { return random_below(seed, Stream::RESAMPLE, counter++, n); }
};

template <size_t N>
void shuffle_prefix(array<uint8_t, N>& a, int n, Rng& rng) {
    for (int i = n - 1; i > 0; --i) {
        swap(a[i], a[rng.below(static_cast<uint32_t>(i + 1))]);
    }
}

// Cards are dealt as contiguous segments of one shuffled array: [start[h], start[h] + len[h]).
template <size_t N, int H>
struct Deal {
    array<uint8_t, N> card{};
    int n = 0;
    array<int, H> start{};
    array<int, H> len{};

    int count(int h, int type) const {
        int c = 0;
        for (int i = start[h]; i < start[h] + len[h]; ++i) c += card[i] == type;
        return c;
    }
    int count_mask(int h, uint8_t types) const {
        int c = 0;
        for (int i = start[h]; i < start[h] + len[h]; ++i) c += (types >> card[i]) & 1;
        return c;
    }
    // First position in h's segment whose card is (or is not) in `types`; -1 if none.
    int find(int h, uint8_t types, bool in) const {
        for (int i = start[h]; i < start[h] + len[h]; ++i) {
            if ((((types >> card[i]) & 1) != 0) == in) return i;
        }
        return -1;
    }
};

using ResDeal = Deal<NUM_RES * RESOURCE_PER_TYPE, NUM_PLAYERS>;
using DevDeal = Deal<DEV_DECK_SIZE, NUM_PLAYERS + 1>;  // holder NUM_PLAYERS is the deck
constexpr int DECK = NUM_PLAYERS;
constexpr uint8_t VP_MASK = 1u << VP;

// Deals opponents' unknown resource cards. Returns dealt counts per opponent and type.
array<array<uint8_t, NUM_RES>, NUM_PLAYERS> deal_resources(const Board& board,
                                                           const GameState& truth,
                                                           const BeliefState& belief,
                                                           uint8_t o, Rng& rng) {
    const auto unseen = unseen_resources(belief, truth.pub, o);
    ResDeal d;
    for (int r = 0; r < NUM_RES; ++r) {
        for (int i = 0; i < unseen[r]; ++i) d.card[d.n++] = static_cast<uint8_t>(r);
    }
    int pos = 0;
    for (int t = 0; t < NUM_PLAYERS; ++t) {
        d.start[t] = pos;
        d.len[t] = t == o ? 0 : belief.hand[o][t].unknown;
        pos += d.len[t];
    }
    // Every unseen card sits in some opponent's unknown slots: both are what o cannot account
    // for (hand sizes minus known cards, 19 minus bank minus known cards).
    CATAN_CHECK(pos == d.n);

    const HandNeed need = hand_need(board, truth.pub, o);
    int missing = 0;  // cards of the needed types q must get from the deal
    if (need.player != NONE) {
        missing = need.count;
        for (int r = 0; r < NUM_RES; ++r) {
            if ((need.types >> r) & 1) missing -= belief.hand[o][need.player].known[r];
        }
    }
    for (int tries = 0; tries < MAX_TRIES; ++tries) {
        shuffle_prefix(d.card, d.n, rng);
        if (missing <= 0 || d.count_mask(need.player, need.types) >= missing) break;
    }
    // Repair: swap a non-needed card of q for a needed one elsewhere. The truth's deal meets
    // the need, so while short, q has a card outside `types` and another hand one inside.
    if (missing > 0) {
        const int q = need.player;
        while (d.count_mask(q, need.types) < missing) {
            const int i = d.find(q, need.types, false);
            int j = -1;
            for (int t = 0; t < NUM_PLAYERS && j < 0; ++t) {
                if (t != q) j = d.find(t, need.types, true);
            }
            CATAN_CHECK(i >= 0 && j >= 0);
            if (i < 0 || j < 0) break;
            swap(d.card[i], d.card[j]);
        }
    }

    array<array<uint8_t, NUM_RES>, NUM_PLAYERS> dealt{};
    for (int t = 0; t < NUM_PLAYERS; ++t) {
        for (int i = d.start[t]; i < d.start[t] + d.len[t]; ++i) ++dealt[t][d.card[i]];
    }
    return dealt;
}

// Deals opponents' dev cards and the deck into `out`. The first bought_this_turn cards of the
// current player's segment are the ones bought this turn.
void deal_dev(const GameState& truth, const BeliefState& belief, uint8_t o, Rng& rng,
              GameState& out) {
    const PublicState& pub = truth.pub;
    DevDeal d;
    [[maybe_unused]] int knights = 0;
    for (int p = 0; p < NUM_PLAYERS; ++p) knights += pub.knights_played[p];
    CATAN_CHECK(knights == belief.dev_played[idx(DevCard::KNIGHT)]);
    for (int c = 0; c < NUM_DEV; ++c) {
        const int left = DEV_DECK_COUNTS[c] - belief.dev_played[c] - truth.hid.dev_held[o][c];
        CATAN_CHECK(left >= 0);
        for (int i = 0; i < left; ++i) d.card[d.n++] = static_cast<uint8_t>(c);
    }
    int pos = 0;
    for (int t = 0; t < NUM_PLAYERS; ++t) {
        d.start[t] = pos;
        d.len[t] = t == o ? 0 : pub.dev_count[t];
        pos += d.len[t];
    }
    d.start[DECK] = pos;
    d.len[DECK] = d.n - pos;
    CATAN_CHECK(d.len[DECK] == pub.dev_deck_remaining);

    // VP caps per holder, and for the current player a separate cap on cards held before this
    // turn (positions k and on).
    const uint8_t cur = pub.current_player;
    const int k = cur == o ? 0 : belief.bought_this_turn;
    CATAN_CHECK(k <= d.len[cur]);
    const bool won = pub.phase == Phase::TERMINAL && pub.turn < TURN_CAP;
    const bool live = pub.phase != Phase::TERMINAL;
    array<int, NUM_PLAYERS + 1> cap;
    cap.fill(UNCAPPED);
    int old_cap = UNCAPPED;
    for (int t = 0; t < NUM_PLAYERS; ++t) {
        if (t == o) continue;
        const int before = VICTORY_POINTS_TO_WIN - 1 - belief.vp_at_turn_end[t];
        if (t == cur && won) continue;
        if (t == cur && live) {
            // Old cards are capped by the last turn end, all of them by the VP shown now.
            old_cap = before;
            cap[t] = min(VICTORY_POINTS_TO_WIN - 1 - pub.public_vp[t], before + k);
        } else {
            cap[t] = before;
        }
    }
    auto old_vp = [&] {
        int c = 0;
        for (int i = d.start[cur] + k; i < d.start[cur] + d.len[cur]; ++i) c += d.card[i] == VP;
        return c;
    };
    auto ok = [&] {
        for (int h = 0; h <= DECK; ++h) {
            if (d.count(h, VP) > cap[h]) return false;
        }
        return cur == o || old_vp() <= old_cap;
    };

    for (int tries = 0; tries < MAX_TRIES; ++tries) {
        shuffle_prefix(d.card, d.n, rng);
        if (ok()) break;
    }
    if (!ok()) {
        // Move surplus VP cards to holders with room, trading back a non-VP card. If none had
        // room, the VP cards would exceed every holder's capacity, and the truth could not
        // exist.
        for (int t = 0; t < NUM_PLAYERS; ++t) {
            while (d.count(t, VP) > cap[t]) {
                const int i = d.find(t, VP_MASK, true);
                int j = -1;
                for (int h = 0; h <= DECK && j < 0; ++h) {
                    if (h != t && d.count(h, VP) < cap[h]) j = d.find(h, VP_MASK, false);
                }
                CATAN_CHECK(i >= 0 && j >= 0);
                if (i < 0 || j < 0) break;
                swap(d.card[i], d.card[j]);
            }
        }
        // Then mark VP cards as this turn's purchases until the older ones fit.
        while (cur != o && old_vp() > old_cap) {
            const int s = d.start[cur];
            int i = -1, j = -1;
            for (int x = s; x < s + k && i < 0; ++x) i = d.card[x] != VP ? x : -1;
            for (int x = s + k; x < s + d.len[cur] && j < 0; ++x) j = d.card[x] == VP ? x : -1;
            CATAN_CHECK(i >= 0 && j >= 0);
            if (i < 0 || j < 0) break;
            swap(d.card[i], d.card[j]);
        }
    }

    out.hid.dev_deck.fill(0);
    for (int i = d.start[DECK]; i < d.start[DECK] + d.len[DECK]; ++i) ++out.hid.dev_deck[d.card[i]];
    for (int t = 0; t < NUM_PLAYERS; ++t) {
        if (t == o) continue;
        out.hid.dev_held[t].fill(0);
        for (int i = d.start[t]; i < d.start[t] + d.len[t]; ++i) ++out.hid.dev_held[t][d.card[i]];
    }
    if (cur != o) {
        out.hid.dev_bought_this_turn.fill(0);
        for (int i = d.start[cur]; i < d.start[cur] + k; ++i) {
            ++out.hid.dev_bought_this_turn[d.card[i]];
        }
    }
}

}  // namespace

BeliefState initial_belief() { return BeliefState{}; }

void observe(BeliefState& b, [[maybe_unused]] const PublicState& after, const EventList& events) {
    for (const Event& e : events) {
        switch (e.type) {
            case EventType::GAIN:
                for (int o = 0; o < NUM_PLAYERS; ++o) {
                    uint8_t& k = b.hand[o][e.player].known[e.a];
                    k = static_cast<uint8_t>(k + e.b);
                }
                break;
            case EventType::SPEND:
            case EventType::DISCARD:
                for (int o = 0; o < NUM_PLAYERS; ++o) remove_typed(b.hand[o][e.player], e.a, e.b);
                break;
            case EventType::STEAL: observe_steal(b, e.player, e.target, e.a); break;
            case EventType::MONOPOLY:
                // n is the victim's whole holding of a, so the known-first removal leaves
                // known[a] == 0.
                for (int o = 0; o < NUM_PLAYERS; ++o) {
                    remove_typed(b.hand[o][e.target], e.a, e.b);
                    uint8_t& k = b.hand[o][e.player].known[e.a];
                    k = static_cast<uint8_t>(k + e.b);
                }
                break;
            case EventType::DEV_PLAYED: ++b.dev_played[e.a]; break;
            case EventType::DEV_BOUGHT: ++b.bought_this_turn; break;
            case EventType::TURN_ENDED:
                b.vp_at_turn_end[e.player] = after.public_vp[e.player];
                b.bought_this_turn = 0;
                break;
            default: break;
        }
    }
#if CATAN_CHECKED
    for (int o = 0; o < NUM_PLAYERS; ++o) {
        for (int t = 0; t < NUM_PLAYERS; ++t) {
            const HandBelief& h = b.hand[o][t];
            int n = h.unknown;
            for (int r = 0; r < NUM_RES; ++r) n += h.known[r];
            CATAN_CHECK(n == after.hand_size[t]);
            CATAN_CHECK(o != t || h.unknown == 0);
        }
    }
#endif
}

const char* belief_violation(const BeliefState& b, const GameState& truth, uint8_t observer) {
    const PublicState& pub = truth.pub;
    const HiddenState& hid = truth.hid;
    for (int o = 0; o < NUM_PLAYERS; ++o) {
        if (observer != NONE && o != observer) continue;
        for (int t = 0; t < NUM_PLAYERS; ++t) {
            const HandBelief& h = b.hand[o][t];
            int n = h.unknown;
            for (int r = 0; r < NUM_RES; ++r) {
                n += h.known[r];
                if (h.known[r] > hid.hands[t][r]) return "known exceeds the true hand";
                if (o == t && h.known[r] != hid.hands[t][r]) return "own hand not exact";
            }
            if (n != pub.hand_size[t]) return "known + unknown != hand_size";
            if (o == t && h.unknown != 0) return "own hand has unknown cards";
        }
    }

    int knights = 0;
    for (int p = 0; p < NUM_PLAYERS; ++p) knights += pub.knights_played[p];
    if (b.dev_played[idx(DevCard::KNIGHT)] != knights) return "dev_played knights mismatch";
    for (int c = 0; c < NUM_DEV; ++c) {
        int n = b.dev_played[c] + hid.dev_deck[c];
        for (int p = 0; p < NUM_PLAYERS; ++p) n += hid.dev_held[p][c];
        if (n != DEV_DECK_COUNTS[c]) return "dev_played mismatch";
    }
    // At the turn cap the game ends inside END_TURN, before the hidden counter is reset.
    if (pub.phase != Phase::TERMINAL) {
        int bought = 0;
        for (int c = 0; c < NUM_DEV; ++c) bought += hid.dev_bought_this_turn[c];
        if (bought != b.bought_this_turn) return "bought_this_turn mismatch";
    }

    const uint8_t cur = pub.current_player;
    const bool won = pub.phase == Phase::TERMINAL && pub.turn < TURN_CAP;
    const bool live = pub.phase != Phase::TERMINAL;
    for (int t = 0; t < NUM_PLAYERS; ++t) {
        const int vp = hid.dev_held[t][VP];
        const int old = t == cur && live ? vp - hid.dev_bought_this_turn[VP] : vp;
        if (t == cur && won) continue;
        if (b.vp_at_turn_end[t] + old >= VICTORY_POINTS_TO_WIN) return "VP cards above last turn";
        if (t == cur && live && pub.public_vp[t] + vp >= VICTORY_POINTS_TO_WIN) {
            return "current player should have won";
        }
    }
    return nullptr;
}

array<uint8_t, NUM_RES> unseen_resources(const BeliefState& b, const PublicState& pub,
                                         uint8_t observer) {
    array<uint8_t, NUM_RES> out;
    for (int r = 0; r < NUM_RES; ++r) {
        int n = RESOURCE_PER_TYPE - pub.bank[r];
        for (int t = 0; t < NUM_PLAYERS; ++t) n -= b.hand[observer][t].known[r];
        CATAN_CHECK(n >= 0);
        out[r] = static_cast<uint8_t>(n);
    }
    return out;
}

uint8_t hand_min(const BeliefState& b, uint8_t observer, uint8_t target, int r) {
    return b.hand[observer][target].known[r];
}

uint8_t hand_max(const BeliefState& b, const PublicState& pub, uint8_t observer, uint8_t target,
                 int r) {
    const HandBelief& h = b.hand[observer][target];
    const uint8_t unseen = unseen_resources(b, pub, observer)[r];
    return static_cast<uint8_t>(h.known[r] + min(h.unknown, unseen));
}

GameState resample_hidden(const Board& board, const GameState& truth, const BeliefState& belief,
                          uint8_t observer, uint64_t seed) {
    GameState out = truth;
    Rng rng{seed};
    const auto dealt = deal_resources(board, truth, belief, observer, rng);
    for (int t = 0; t < NUM_PLAYERS; ++t) {
        if (t == observer) continue;
        for (int r = 0; r < NUM_RES; ++r) {
            out.hid.hands[t][r] = static_cast<uint8_t>(belief.hand[observer][t].known[r] +
                                                       dealt[t][r]);
        }
    }
    deal_dev(truth, belief, observer, rng, out);
    return out;
}

BeliefState resampled_belief(const BeliefState& belief, const GameState& resampled,
                             uint8_t observer) {
    BeliefState out = belief;
    for (int o = 0; o < NUM_PLAYERS; ++o) {
        out.hand[o] = belief.hand[observer];
        out.hand[o][o] = HandBelief{resampled.hid.hands[o], 0};
    }
    return out;
}

}  // namespace catan
