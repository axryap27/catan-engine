// Tests for encode.h, the network's view of a state.
//
// The header is the contract. Random games check every observation byte for byte against
// reference_obs(), an independent transcription of the header's comments, and also check
// contract properties that need no reference (one-hot groups, ranges, cross-tile agreement),
// so a misreading shared by both encoders still has to get past those. Hand-built positions
// pin the semantics with literal offsets.
//
//   CATAN_ENCODE_GAMES=n          games per random-game suite (default 40)
//   CATAN_ENCODE_GOLDEN_PRINT=1   print fresh golden hashes, after a deliberate layout or
//                                 semantics change (which also bumps ENCODE_VERSION)

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "catan/action.h"
#include "catan/belief.h"
#include "catan/board.h"
#include "catan/chance.h"
#include "catan/decision.h"
#include "catan/encode.h"
#include "catan/event.h"
#include "catan/invariants.h"
#include "catan/mask.h"
#include "catan/rng.h"
#include "catan/rules.h"
#include "catan/state.h"
#include "catan/topology.h"
#include "catan/types.h"
#include "support/fixtures.h"

using namespace std;
using namespace catan;
using namespace catan::test;

namespace {

// ---- layout ---------------------------------------------------------------------------------

constexpr int NUM_TOKENS = static_cast<int>(TOKEN_VALUES.size());
constexpr int NUM_PHASES = static_cast<int>(Phase::TERMINAL) + 1;
constexpr int NUM_HEX_TYPES = static_cast<int>(HexType::DESERT) + 1;
constexpr int NUM_PORT_TYPES = static_cast<int>(PortType::GENERIC) + 1;
constexpr int NUM_SUMS = 11;  // dice sums 2..12
constexpr int SIDES = 6;
constexpr int VP = idx(DevCard::VICTORY_POINT);

struct Group {
    int offset;
    int width;
};

// True if `groups` cover [0, total) in order, with no gap and no overlap.
template <size_t N>
constexpr bool contiguous(const array<Group, N>& groups, int total) {
    int at = 0;
    for (const Group& g : groups) {
        if (g.offset != at || g.width <= 0) return false;
        at += g.width;
    }
    return at == total;
}

constexpr array<Group, 8> TILE_GROUPS = {{
    {TILE_HEX_TYPE, NUM_HEX_TYPES},
    {TILE_TOKEN, NUM_TOKENS},
    {TILE_PIPS, 1},
    {TILE_ROBBER, 1},
    {TILE_CORNERS, SIDES * NUM_PLAYERS * 2},
    {TILE_ROADS, SIDES * NUM_PLAYERS},
    {TILE_PORT_TYPE, NUM_PORT_TYPES},
    {TILE_PORT_SIDE, SIDES},
}};
constexpr array<Group, 5> SELF_GROUPS = {{
    {SELF_HAND_BUCKET, NUM_RES * HAND_BUCKETS},
    {SELF_HAND, NUM_RES},
    {SELF_DEV_HELD, NUM_DEV},
    {SELF_DEV_BOUGHT, NUM_DEV},
    {SELF_TOTAL_VP, 1},
}};
constexpr array<Group, 15> PLAYER_GROUPS = {{
    {PLAYER_HAND_MIN, NUM_RES},
    {PLAYER_HAND_MAX, NUM_RES},
    {PLAYER_HAND_SIZE, 1},
    {PLAYER_DEV_COUNT, 1},
    {PLAYER_KNIGHTS, 1},
    {PLAYER_PUBLIC_VP, 1},
    {PLAYER_ROAD_LEN, 1},
    {PLAYER_PIECES_LEFT, 3},
    {PLAYER_PENDING_DISCARD, 1},
    {PLAYER_LONGEST_ROAD, 1},
    {PLAYER_LARGEST_ARMY, 1},
    {PLAYER_IS_CURRENT, 1},
    {PLAYER_VP_AT_TURN_END, 1},
    {PLAYER_PORTS, NUM_PORT_TYPES},
    {PLAYER_PRODUCTION, NUM_RES * NUM_TOKENS},
}};
constexpr array<Group, 13> GLOBAL_GROUPS = {{
    {GLOBAL_PHASE, NUM_PHASES},
    {GLOBAL_DICE, NUM_SUMS},
    {GLOBAL_BANK, NUM_RES},
    {GLOBAL_DEV_DECK, 1},
    {GLOBAL_DEV_PLAYED, NUM_DEV},
    {GLOBAL_BOUGHT_THIS_TURN, 1},
    {GLOBAL_PLAYED_DEV, 1},
    {GLOBAL_FREE_ROADS, 1},
    {GLOBAL_YOP_REMAINING, 1},
    {GLOBAL_EXCHANGE_GIVE, NUM_RES},
    {GLOBAL_DISCARD_FLOOR, 1},
    {GLOBAL_SETUP_STEP, 1},
    {GLOBAL_TURN, 1},
}};
constexpr array<Group, 4> OBS_GROUPS = {{
    {OBS_TILES, NUM_HEXES * TILE_FEATURES},
    {OBS_SELF, SELF_FEATURES},
    {OBS_PLAYERS, NUM_PLAYERS * PLAYER_FEATURES},
    {OBS_GLOBAL, GLOBAL_FEATURES},
}};
constexpr array<Group, 8> MASK_GROUPS = {{
    {MASK_TYPE, NUM_ACTION_TYPES},
    {MASK_SETTLEMENT, NUM_VERTICES},
    {MASK_CITY, NUM_VERTICES},
    {MASK_EDGE, NUM_EDGES},
    {MASK_TILE, NUM_HEXES},
    {MASK_DEVCARD, NUM_DEV},
    {MASK_PLAYER, NUM_PLAYERS - 1},
    {MASK_RESOURCE, NUM_RES},
}};

static_assert(contiguous(TILE_GROUPS, TILE_FEATURES));
static_assert(contiguous(SELF_GROUPS, SELF_FEATURES));
static_assert(contiguous(PLAYER_GROUPS, PLAYER_FEATURES));
static_assert(contiguous(GLOBAL_GROUPS, GLOBAL_FEATURES));
static_assert(contiguous(OBS_GROUPS, OBS_SIZE));
static_assert(contiguous(MASK_GROUPS, MASK_SIZE));

// The sizes the header's comments state.
static_assert(NUM_TOKENS == 10 && HAND_BUCKETS == 8);
static_assert(NUM_ACTION_TYPES == 13 && NUM_PHASES == 11);
static_assert(NUM_HEX_TYPES == 6 && NUM_PORT_TYPES == 6);
static_assert(TILE_FEATURES == 102 && SELF_FEATURES == 56);
static_assert(PLAYER_FEATURES == 79 && GLOBAL_FEATURES == 45);
static_assert(OBS_SELF == 1938 && OBS_PLAYERS == 1994 && OBS_GLOBAL == 2310);
static_assert(OBS_SIZE == 2355);
static_assert(MASK_SETTLEMENT == 13 && MASK_CITY == 67 && MASK_EDGE == 121);
static_assert(MASK_TILE == 193 && MASK_DEVCARD == 212 && MASK_PLAYER == 217);
static_assert(MASK_RESOURCE == 220 && MASK_SIZE == 225);
static_assert(sizeof(Observation) == OBS_SIZE && sizeof(MaskBuffer) == MASK_SIZE);
// Golden hashes below are tied to this version.
static_assert(ENCODE_VERSION == 1);

// ---- reading observations -------------------------------------------------------------------

// Pre-fill for output buffers. Every byte is a small count or flag, so 0xAB never appears
// legitimately except as GLOBAL_TURN (turn 171).
constexpr uint8_t FILL = 0xAB;

const uint8_t* tile_of(const Observation& x, int h) {
    return x.data() + OBS_TILES + h * TILE_FEATURES;
}
const uint8_t* self_of(const Observation& x) { return x.data() + OBS_SELF; }
const uint8_t* row_of(const Observation& x, int k) {
    return x.data() + OBS_PLAYERS + k * PLAYER_FEATURES;
}
const uint8_t* global_of(const Observation& x) { return x.data() + OBS_GLOBAL; }

int nonzero(const uint8_t* p, int n) {
    int c = 0;
    for (int i = 0; i < n; ++i) c += p[i] != 0;
    return c;
}

int total(const uint8_t* p, int n) {
    int s = 0;
    for (int i = 0; i < n; ++i) s += p[i];
    return s;
}

bool all_flags(const uint8_t* p, int n) {
    for (int i = 0; i < n; ++i) {
        if (p[i] > 1) return false;
    }
    return true;
}

// Index of the only set byte of a group: -1 if none, -2 if several or a byte is not 0/1.
int hot(const uint8_t* p, int n) {
    int at = -1;
    for (int i = 0; i < n; ++i) {
        if (p[i] == 0) continue;
        if (p[i] != 1 || at != -1) return -2;
        at = i;
    }
    return at;
}

int token_index(int token) {
    for (int i = 0; i < NUM_TOKENS; ++i) {
        if (TOKEN_VALUES[i] == token) return i;
    }
    return -1;
}

int pips_of(int token) { return token == 0 ? 0 : 6 - abs(7 - token); }
int bucket_of(int n) { return n <= 5 ? n : n <= 7 ? 6 : 7; }

uint8_t rel(int seat, int observer) {
    return static_cast<uint8_t>((seat - observer + NUM_PLAYERS) % NUM_PLAYERS);
}
uint8_t seat_at(int observer, int k) {
    return static_cast<uint8_t>((observer + k) % NUM_PLAYERS);
}

uint8_t road_owner(const PublicState& pub, int e) {
    for (uint8_t p = 0; p < NUM_PLAYERS; ++p) {
        if (edge_set_has(pub.roads[p], e)) return p;
    }
    return NONE;
}

// Section, row and column of observation byte i, for failure messages.
string describe(int i) {
    for (const ObsSection& s : OBS_SECTIONS) {
        if (i < s.offset || i >= s.offset + s.rows * s.width) continue;
        const int r = (i - s.offset) / s.width, c = (i - s.offset) % s.width;
        return string(s.name) + " row " + to_string(r) + " col " + to_string(c);
    }
    return "byte " + to_string(i);
}

template <size_t N>
int first_diff(const array<uint8_t, N>& a, const array<uint8_t, N>& b) {
    for (size_t i = 0; i < N; ++i) {
        if (a[i] != b[i]) return static_cast<int>(i);
    }
    return -1;
}

// ---- encoding wrappers ----------------------------------------------------------------------

Observation obs_of(const Board& board, const GameState& s, const BeliefState& b, uint8_t o) {
    Observation x;
    x.fill(FILL);
    encode(board, s, b, o, x);
    return x;
}

MaskBuffer mask_of(const Decision& d) {
    MaskBuffer m;
    m.fill(FILL);
    encode_mask(d, m);
    return m;
}

// ---- the contract, transcribed --------------------------------------------------------------

// What private_view must return: the observer's own cards, its purchases only on its turn.
PrivateView reference_view(const GameState& s, uint8_t o) {
    PrivateView v{};
    v.hand = s.hid.hands[o];
    v.dev_held = s.hid.dev_held[o];
    if (s.pub.current_player == o) v.dev_bought_this_turn = s.hid.dev_bought_this_turn;
    return v;
}

bool same_view(const PrivateView& a, const PrivateView& b) {
    return a.hand == b.hand && a.dev_held == b.dev_held &&
           a.dev_bought_this_turn == b.dev_bought_this_turn;
}

// The observation encode.h describes, written from its comments alone.
Observation reference_obs(const Board& board, const PublicState& pub, const BeliefState& b,
                          const PrivateView& own, uint8_t o) {
    const Topology& topo = topology();
    Observation x{};

    for (int h = 0; h < NUM_HEXES; ++h) {
        uint8_t* t = x.data() + OBS_TILES + h * TILE_FEATURES;
        t[TILE_HEX_TYPE + static_cast<int>(board.hex_type[h])] = 1;
        if (board.hex_type[h] != HexType::DESERT) {
            t[TILE_TOKEN + token_index(board.token[h])] = 1;
            t[TILE_PIPS] = static_cast<uint8_t>(pips_of(board.token[h]));
        }
        t[TILE_ROBBER] = pub.robber_hex == h;
        for (int i = 0; i < SIDES; ++i) {
            const uint8_t v = pub.vertex[topo.hex_vertices[h][i]];
            if (vertex_occupied(v)) {
                t[TILE_CORNERS + 8 * i + 2 * rel(vertex_owner(v), o) + vertex_is_city(v)] = 1;
            }
            const uint8_t owner = road_owner(pub, topo.hex_edges[h][i]);
            if (owner != NONE) t[TILE_ROADS + 4 * i + rel(owner, o)] = 1;
        }
        for (int slot = 0; slot < NUM_PORTS; ++slot) {
            const PortSlot& ps = topo.port_slots[slot];
            if (ps.hex != h) continue;
            t[TILE_PORT_TYPE + static_cast<int>(board.port_type[slot])] = 1;
            for (int i = 0; i < SIDES; ++i) {
                if (topo.hex_edges[h][i] == ps.edge) t[TILE_PORT_SIDE + i] = 1;
            }
        }
    }

    uint8_t* me = x.data() + OBS_SELF;
    for (int r = 0; r < NUM_RES; ++r) {
        me[SELF_HAND_BUCKET + HAND_BUCKETS * r + bucket_of(own.hand[r])] = 1;
        me[SELF_HAND + r] = own.hand[r];
    }
    for (int c = 0; c < NUM_DEV; ++c) {
        me[SELF_DEV_HELD + c] = own.dev_held[c];
        me[SELF_DEV_BOUGHT + c] = own.dev_bought_this_turn[c];
    }
    me[SELF_TOTAL_VP] = static_cast<uint8_t>(pub.public_vp[o] + own.dev_held[VP]);

    for (int k = 0; k < NUM_PLAYERS; ++k) {
        const uint8_t t = seat_at(o, k);
        uint8_t* p = x.data() + OBS_PLAYERS + k * PLAYER_FEATURES;
        for (int r = 0; r < NUM_RES; ++r) {
            p[PLAYER_HAND_MIN + r] = hand_min(b, o, t, r);
            p[PLAYER_HAND_MAX + r] = hand_max(b, pub, o, t, r);
        }
        p[PLAYER_HAND_SIZE] = pub.hand_size[t];
        p[PLAYER_DEV_COUNT] = pub.dev_count[t];
        p[PLAYER_KNIGHTS] = pub.knights_played[t];
        p[PLAYER_PUBLIC_VP] = pub.public_vp[t];
        p[PLAYER_ROAD_LEN] = pub.road_len[t];
        for (int i = 0; i < 3; ++i) p[PLAYER_PIECES_LEFT + i] = pub.pieces_left[t][i];
        p[PLAYER_PENDING_DISCARD] = pub.pending_discard[t];
        p[PLAYER_LONGEST_ROAD] = pub.longest_road_holder == t;
        p[PLAYER_LARGEST_ARMY] = pub.largest_army_holder == t;
        p[PLAYER_IS_CURRENT] = pub.current_player == t;
        p[PLAYER_VP_AT_TURN_END] = b.vp_at_turn_end[t];
        for (int v = 0; v < NUM_VERTICES; ++v) {
            if (vertex_owner(pub.vertex[v]) == t && board.vertex_port[v] != NONE) {
                p[PLAYER_PORTS + board.vertex_port[v]] = 1;
            }
        }
        for (int h = 0; h < NUM_HEXES; ++h) {
            if (h == pub.robber_hex || board.hex_type[h] == HexType::DESERT) continue;
            const int at = PLAYER_PRODUCTION + NUM_TOKENS * static_cast<int>(board.hex_type[h]) +
                           token_index(board.token[h]);
            for (uint8_t v : topo.hex_vertices[h]) {
                if (vertex_owner(pub.vertex[v]) != t) continue;
                p[at] = static_cast<uint8_t>(p[at] + (vertex_is_city(pub.vertex[v]) ? 2 : 1));
            }
        }
    }

    uint8_t* g = x.data() + OBS_GLOBAL;
    g[GLOBAL_PHASE + static_cast<int>(pub.phase)] = 1;
    if (pub.dice >= 2 && pub.dice <= 12) g[GLOBAL_DICE + pub.dice - 2] = 1;
    for (int r = 0; r < NUM_RES; ++r) g[GLOBAL_BANK + r] = pub.bank[r];
    g[GLOBAL_DEV_DECK] = pub.dev_deck_remaining;
    for (int c = 0; c < NUM_DEV; ++c) g[GLOBAL_DEV_PLAYED + c] = b.dev_played[c];
    g[GLOBAL_BOUGHT_THIS_TURN] = b.bought_this_turn;
    g[GLOBAL_PLAYED_DEV] = pub.played_dev_this_turn;
    g[GLOBAL_FREE_ROADS] = pub.free_roads;
    g[GLOBAL_YOP_REMAINING] = pub.yop_remaining;
    if (pub.phase == Phase::EXCHANGE_RECV) g[GLOBAL_EXCHANGE_GIVE + pub.exchange_give] = 1;
    g[GLOBAL_DISCARD_FLOOR] = pub.discard_floor;
    g[GLOBAL_SETUP_STEP] = pub.setup_step;  // 16 after setup; see check_structure
    g[GLOBAL_TURN] = static_cast<uint8_t>(min<int>(pub.turn, 255));
    return x;
}

// ---- failure bookkeeping --------------------------------------------------------------------

// Remembers the first failure of a long loop (per-step REQUIREs would swamp Catch) and counts
// the checks made.
struct Failure {
    string msg;
    long checks = 0;
    uint64_t seed = 0;
    int ply = 0;
    int observer = -1;

    void at(uint64_t s, int p, int o = -1) {
        seed = s;
        ply = p;
        observer = o;
    }
    bool ok() const { return msg.empty(); }
    void fail(const string& what) {
        if (!msg.empty()) return;
        msg = "seed " + to_string(seed) + " ply " + to_string(ply) +
              (observer >= 0 ? " observer " + to_string(observer) : string()) + ": " + what;
    }
    void expect(bool cond, const char* what) {
        ++checks;
        if (!cond) fail(what);
    }
    void same(const Observation& got, const Observation& want, const char* what) {
        ++checks;
        const int i = first_diff(got, want);
        if (i < 0) return;
        fail(string(what) + " at " + describe(i) + " (got " + to_string(got[i]) + ", want " +
             to_string(want[i]) + ")");
    }
};

// ---- random games ---------------------------------------------------------------------------

int env_count(const char* name, int fallback) {
    const char* v = getenv(name);
    const int n = v ? atoi(v) : 0;
    return n > 0 ? n : fallback;
}

uint64_t games() { return static_cast<uint64_t>(env_count("CATAN_ENCODE_GAMES", 40)); }

Board board_for(uint64_t seed) { return seed % 2 ? standard_board() : random_board(seed); }

// Plays the uniform random game play_random plays with this seed, keeping a BeliefState in
// step, and calls at(state, belief, decision) before every decision and once at TERMINAL.
template <typename F>
void play_tracked(const Board& board, uint64_t seed, F&& at) {
    GameState s = initial_state(board);
    BeliefState b = initial_belief();
    for (;;) {
        const Decision d = next_decision(board, s);
        at(static_cast<const GameState&>(s), static_cast<const BeliefState&>(b), d);
        if (d.kind == DecisionKind::TERMINAL) return;
        EventList ev;
        if (d.kind == DecisionKind::CHANCE) {
            ev = apply_chance(board, s, sample_chance(d.chance, s, seed));
        } else {
            const auto n = random_below(seed, 1000, s.pub.ply, d.n_legal);
            ev = apply(board, s, nth_legal(d.mask, d.phase, static_cast<int>(n)));
        }
        observe(b, s.pub, ev);
    }
}

// Where each vertex (edge) appears among the tiles, as (hex, corner (side)) pairs.
struct Occurrences {
    array<vector<pair<int, int>>, NUM_VERTICES> vertex;
    array<vector<pair<int, int>>, NUM_EDGES> edge;
};

const Occurrences& occurrences() {
    static const Occurrences occ = [] {
        Occurrences o;
        const Topology& topo = topology();
        for (int h = 0; h < NUM_HEXES; ++h) {
            for (int i = 0; i < SIDES; ++i) {
                o.vertex[topo.hex_vertices[h][i]].push_back({h, i});
                o.edge[topo.hex_edges[h][i]].push_back({h, i});
            }
        }
        return o;
    }();
    return occ;
}

// Contract properties that need no reference encoder: every byte written, one-hot groups,
// flags, ranges, the observer's exact row, and corner and road bytes against the board in
// every tile they touch.
void check_structure(const Board& board, const GameState& s, const BeliefState& b, uint8_t o,
                     const Observation& x, Failure& f) {
    const PublicState& pub = s.pub;
    const Topology& topo = topology();
    // Every byte written: no pre-fill byte survives (GLOBAL_TURN alone can legitimately be
    // 0xAB = 171), and a buffer pre-filled differently encodes identically.
    bool filled = false;
    for (int i = 0; i < OBS_SIZE; ++i) filled |= x[i] == FILL && i != OBS_GLOBAL + GLOBAL_TURN;
    f.expect(!filled, "a byte was left unwritten (0xAB)");
    Observation y;
    y.fill(static_cast<uint8_t>(~FILL));
    encode(board, s, b, o, y);
    f.expect(x == y, "output depends on the buffer's previous contents");

    int robbers = 0, harbors = 0;
    for (int h = 0; h < NUM_HEXES; ++h) {
        const uint8_t* t = tile_of(x, h);
        const bool desert = board.hex_type[h] == HexType::DESERT;
        f.expect(hot(t + TILE_HEX_TYPE, NUM_HEX_TYPES) == static_cast<int>(board.hex_type[h]),
                 "hex type is not a one-hot of the board's type");
        f.expect(hot(t + TILE_TOKEN, NUM_TOKENS) >= (desert ? -1 : 0) &&
                     (nonzero(t + TILE_TOKEN, NUM_TOKENS) == 0) == desert,
                 "token is not one-hot (none on the desert)");
        f.expect(t[TILE_PIPS] <= 5 && (t[TILE_PIPS] == 0) == desert, "pips outside 1..5");
        f.expect(t[TILE_ROBBER] <= 1, "robber flag not 0/1");
        robbers += t[TILE_ROBBER];
        for (int i = 0; i < SIDES; ++i) {
            const uint8_t* c = t + TILE_CORNERS + 8 * i;
            const uint8_t v = pub.vertex[topo.hex_vertices[h][i]];
            if (vertex_occupied(v)) {
                f.expect(hot(c, 8) == 2 * rel(vertex_owner(v), o) + vertex_is_city(v),
                         "occupied corner: not the one byte of its owner and kind");
            } else {
                f.expect(nonzero(c, 8) == 0, "empty corner has a building byte");
            }
            const uint8_t* r = t + TILE_ROADS + 4 * i;
            const uint8_t owner = road_owner(pub, topo.hex_edges[h][i]);
            f.expect(hot(r, 4) == (owner == NONE ? -1 : rel(owner, o)),
                     "side bytes do not match the road on that edge");
        }
        const int type = hot(t + TILE_PORT_TYPE, NUM_PORT_TYPES);
        const int side = hot(t + TILE_PORT_SIDE, SIDES);
        f.expect(type >= -1 && side >= -1 && (type == -1) == (side == -1),
                 "harbor type and side are not both one-hot or both none");
        harbors += type >= 0;
    }
    f.expect(robbers == 1 && tile_of(x, pub.robber_hex)[TILE_ROBBER] == 1,
             "the robber is not flagged on exactly its hex");
    f.expect(harbors == NUM_PORTS, "not every harbor is on one tile");

    const Occurrences& occ = occurrences();
    for (const auto& list : occ.vertex) {
        const uint8_t* first = tile_of(x, list[0].first) + TILE_CORNERS + 8 * list[0].second;
        for (const auto& [h, i] : list) {
            const uint8_t* c = tile_of(x, h) + TILE_CORNERS + 8 * i;
            f.expect(memcmp(c, first, 8) == 0, "tiles disagree about a shared vertex");
        }
    }
    for (const auto& list : occ.edge) {
        const uint8_t* first = tile_of(x, list[0].first) + TILE_ROADS + 4 * list[0].second;
        for (const auto& [h, i] : list) {
            const uint8_t* r = tile_of(x, h) + TILE_ROADS + 4 * i;
            f.expect(memcmp(r, first, 4) == 0, "tiles disagree about a shared edge");
        }
    }

    const uint8_t* me = self_of(x);
    const auto& hand = s.hid.hands[o];
    for (int r = 0; r < NUM_RES; ++r) {
        f.expect(hot(me + SELF_HAND_BUCKET + HAND_BUCKETS * r, HAND_BUCKETS) ==
                     bucket_of(hand[r]),
                 "hand bucket is not the one-hot of the observer's count");
        f.expect(me[SELF_HAND + r] == hand[r], "SELF_HAND is not the observer's hand");
    }
    const bool own_turn = pub.current_player == o;
    for (int c = 0; c < NUM_DEV; ++c) {
        f.expect(me[SELF_DEV_HELD + c] == s.hid.dev_held[o][c], "SELF_DEV_HELD");
        f.expect(me[SELF_DEV_BOUGHT + c] == (own_turn ? s.hid.dev_bought_this_turn[c] : 0),
                 "SELF_DEV_BOUGHT is not the observer's purchases on its turn");
    }
    f.expect(me[SELF_TOTAL_VP] == pub.public_vp[o] + s.hid.dev_held[o][VP], "SELF_TOTAL_VP");

    int current = 0, longest = 0, largest = 0;
    for (int k = 0; k < NUM_PLAYERS; ++k) {
        const uint8_t* p = row_of(x, k);
        const uint8_t t = seat_at(o, k);
        int lo = 0, hi = 0;
        bool ordered = true, sound = true;
        for (int r = 0; r < NUM_RES; ++r) {
            const int mn = p[PLAYER_HAND_MIN + r], mx = p[PLAYER_HAND_MAX + r];
            lo += mn;
            hi += mx;
            ordered = ordered && mn <= mx;
            sound = sound && mn <= s.hid.hands[t][r] && s.hid.hands[t][r] <= mx;
            if (k == 0) {
                f.expect(mn == hand[r] && mx == hand[r], "row 0 bounds are not the exact hand");
            }
        }
        f.expect(ordered, "HAND_MIN above HAND_MAX");
        f.expect(sound, "a true hand count lies outside [HAND_MIN, HAND_MAX]");
        f.expect(p[PLAYER_HAND_SIZE] == pub.hand_size[t], "HAND_SIZE");
        f.expect(p[PLAYER_VP_AT_TURN_END] == b.vp_at_turn_end[t], "VP_AT_TURN_END");
        f.expect(lo <= pub.hand_size[t] && pub.hand_size[t] <= hi,
                 "hand size outside [sum HAND_MIN, sum HAND_MAX]");
        f.expect(p[PLAYER_LONGEST_ROAD] <= 1 && p[PLAYER_LARGEST_ARMY] <= 1 &&
                     p[PLAYER_IS_CURRENT] <= 1 && all_flags(p + PLAYER_PORTS, NUM_PORT_TYPES),
                 "player flag not 0/1");
        current += p[PLAYER_IS_CURRENT];
        longest += p[PLAYER_LONGEST_ROAD];
        largest += p[PLAYER_LARGEST_ARMY];
    }
    f.expect(current == 1 && row_of(x, rel(pub.current_player, o))[PLAYER_IS_CURRENT] == 1,
             "IS_CURRENT is not set on exactly the current player's row");
    f.expect(longest == (pub.longest_road_holder != NONE), "LONGEST_ROAD rows");
    f.expect(largest == (pub.largest_army_holder != NONE), "LARGEST_ARMY rows");

    const uint8_t* g = global_of(x);
    f.expect(hot(g + GLOBAL_PHASE, NUM_PHASES) == static_cast<int>(pub.phase), "phase one-hot");
    f.expect(hot(g + GLOBAL_DICE, NUM_SUMS) == (pub.dice == 0 ? -1 : pub.dice - 2),
             "dice one-hot (none before the roll)");
    f.expect(hot(g + GLOBAL_EXCHANGE_GIVE, NUM_RES) ==
                 (pub.phase == Phase::EXCHANGE_RECV ? pub.exchange_give : -1),
             "exchange give one-hot only in EXCHANGE_RECV");
    f.expect(g[GLOBAL_PLAYED_DEV] <= 1, "PLAYED_DEV flag not 0/1");
    f.expect(g[GLOBAL_BOUGHT_THIS_TURN] == b.bought_this_turn, "BOUGHT_THIS_TURN (public)");
    f.expect(g[GLOBAL_DISCARD_FLOOR] < NUM_RES, "DISCARD_FLOOR outside 0..4");
    // The header says 0..15, but PublicState::setup_step is 16 once setup is over and the
    // encoder copies it (as reference_obs does): 0..15 holds only during SETUP.
    f.expect(g[GLOBAL_SETUP_STEP] < SETUP_STEPS + (pub.phase == Phase::SETUP ? 0 : 1),
             "SETUP_STEP out of range");
    f.expect(g[GLOBAL_TURN] == min<int>(pub.turn, 255), "TURN is not min(turn, 255)");
}

// The flag a legal action sets besides its type flag; -1 for a type without an argument.
int param_flag(Phase phase, Action a) {
    switch (param_head(phase, a.type)) {
        case Head::NONE: return -1;
        case Head::VERTEX:
            return (a.type == ActionType::BUILD_SETTLEMENT ? MASK_SETTLEMENT : MASK_CITY) + a.arg;
        case Head::EDGE: return MASK_EDGE + a.arg;
        case Head::TILE: return MASK_TILE + a.arg;
        case Head::DEVCARD: return MASK_DEVCARD + a.arg;
        case Head::PLAYER: return MASK_PLAYER + a.arg;
        case Head::RESOURCE: return MASK_RESOURCE + a.arg;
    }
    return -1;
}

// The expansion of a decision's mask: field by field, and as a bijection with its legal moves.
void check_mask(const Decision& d, const MaskBuffer& m, Failure& f) {
    f.expect(all_flags(m.data(), MASK_SIZE), "mask byte not 0/1 (or left unwritten)");
    if (d.kind != DecisionKind::PLAYER) {
        f.expect(nonzero(m.data(), MASK_SIZE) == 0, "mask not all zero for a non-PLAYER decision");
        return;
    }
    const ActionMask& am = d.mask;
    bool same = true;
    for (int t = 0; t < NUM_ACTION_TYPES; ++t) same &= m[MASK_TYPE + t] == ((am.type >> t) & 1);
    for (int v = 0; v < NUM_VERTICES; ++v) {
        same &= m[MASK_SETTLEMENT + v] == ((am.settlement >> v) & 1);
        same &= m[MASK_CITY + v] == ((am.city >> v) & 1);
    }
    for (int e = 0; e < NUM_EDGES; ++e) same &= m[MASK_EDGE + e] == edge_set_has(am.edge, e);
    for (int h = 0; h < NUM_HEXES; ++h) same &= m[MASK_TILE + h] == ((am.tile >> h) & 1);
    for (int c = 0; c < NUM_DEV; ++c) same &= m[MASK_DEVCARD + c] == ((am.devcard >> c) & 1);
    for (int i = 0; i < 3; ++i) same &= m[MASK_PLAYER + i] == ((am.player >> i) & 1);
    for (int r = 0; r < NUM_RES; ++r) same &= m[MASK_RESOURCE + r] == ((am.resource >> r) & 1);
    f.expect(same, "mask flags differ from the ActionMask bits");
    // Nothing the expansion could drop: no ActionMask bit beyond its head's range.
    f.expect((am.type >> NUM_ACTION_TYPES) == 0 && (am.settlement >> NUM_VERTICES) == 0 &&
                 (am.city >> NUM_VERTICES) == 0 && (am.edge[1] >> (NUM_EDGES - 64)) == 0 &&
                 (am.tile >> NUM_HEXES) == 0 && (am.devcard >> NUM_DEV) == 0 &&
                 (am.player >> 3) == 0 && (am.resource >> NUM_RES) == 0,
             "ActionMask has a bit beyond its head");

    // The exact count: each set type without an argument is one action, and every other legal
    // action is one flag of one parameter section. That holds because in any phase each head
    // serves at most one type (the vertex head's two types have separate sections).
    int bare = 0;
    for (int t = 0; t < NUM_ACTION_TYPES; ++t) {
        bare += m[MASK_TYPE + t] && param_head(d.phase, static_cast<ActionType>(t)) == Head::NONE;
    }
    const int params = total(m.data() + MASK_SETTLEMENT, MASK_SIZE - MASK_SETTLEMENT);
    const int legal = count_legal(am, d.phase);
    f.expect(legal == d.n_legal, "n_legal differs from count_legal");
    f.expect(bare + params == legal, "argument-less types + parameter flags != count_legal");

    // Bijection: every legal action's flags are set, no two actions share a parameter flag,
    // and every set flag belongs to some legal action.
    MaskBuffer hit{};
    for (int n = 0; n < legal; ++n) {
        const Action a = nth_legal(am, d.phase, n);
        hit[MASK_TYPE + idx(a.type)] = 1;
        const int p = param_flag(d.phase, a);
        if (p < 0) continue;
        f.expect(p < MASK_SIZE && hit[p] == 0, "two legal actions share one mask flag");
        if (p < MASK_SIZE) hit[p] = 1;
    }
    f.expect(hit == m, "set flags are not exactly those of the legal actions");
}

// ---- hidden-information and rotation helpers ------------------------------------------------

// Overwrites everything `o` may not see with arbitrary bytes: other seats' hands and dev cards,
// the deck, and (unless o is to move) the current player's purchases this turn. The result is
// usually inconsistent; the encoder never reads any of it, so it must not care.
void scramble_hidden(GameState& s, uint8_t o, uint64_t key) {
    uint64_t i = 0;
    auto next = [&] { return static_cast<uint8_t>(random_u64(key, 900, i++)); };
    for (int t = 0; t < NUM_PLAYERS; ++t) {
        if (t == o) continue;
        for (uint8_t& c : s.hid.hands[t]) c = next();
        for (uint8_t& c : s.hid.dev_held[t]) c = next();
    }
    for (uint8_t& c : s.hid.dev_deck) c = next();
    if (s.pub.current_player != o) {
        for (uint8_t& c : s.hid.dev_bought_this_turn) c = next();
    }
    for (uint8_t& c : s.hid.reserved_) c = next();
}

// Overwrites every observer's belief row but o's with arbitrary bytes.
void scramble_other_rows(BeliefState& b, uint8_t o, uint64_t key) {
    uint64_t i = 0;
    auto next = [&] { return static_cast<uint8_t>(random_u64(key, 901, i++)); };
    for (int w = 0; w < NUM_PLAYERS; ++w) {
        if (w == o) continue;
        for (HandBelief& h : b.hand[w]) {
            for (uint8_t& c : h.known) c = next();
            h.unknown = next();
        }
    }
}

uint8_t turned(uint8_t seat, int k) {
    return seat == NONE ? NONE : static_cast<uint8_t>((seat + k) % NUM_PLAYERS);
}

// Relabels every absolute seat p as p + k (mod 4): per-player arrays, building owners, hands
// and dev cards, and every seat-valued field. Hexes, vertices, edges, resources and cards keep
// their ids. setup_step stays: setup order is absolute (setup_player), so a rotated SETUP
// state cannot arise in play and next_decision rejects it, but it is still a fine encoder input.
GameState rotated(const GameState& s, int k) {
    GameState r = s;
    const PublicState& a = s.pub;
    PublicState& z = r.pub;
    for (int p = 0; p < NUM_PLAYERS; ++p) {
        const int q = (p + k) % NUM_PLAYERS;
        z.roads[q] = a.roads[p];
        z.hand_size[q] = a.hand_size[p];
        z.dev_count[q] = a.dev_count[p];
        z.knights_played[q] = a.knights_played[p];
        z.public_vp[q] = a.public_vp[p];
        z.road_len[q] = a.road_len[p];
        z.pending_discard[q] = a.pending_discard[p];
        z.pieces_left[q] = a.pieces_left[p];
        r.hid.hands[q] = s.hid.hands[p];
        r.hid.dev_held[q] = s.hid.dev_held[p];
    }
    for (int v = 0; v < NUM_VERTICES; ++v) {
        const uint8_t byte = a.vertex[v];
        if (!vertex_occupied(byte)) continue;
        const uint8_t q = turned(vertex_owner(byte), k);
        z.vertex[v] = vertex_is_city(byte) ? make_city(q) : make_settlement(q);
    }
    z.current_player = turned(a.current_player, k);
    z.longest_road_holder = turned(a.longest_road_holder, k);
    z.largest_army_holder = turned(a.largest_army_holder, k);
    z.steal_victim = turned(a.steal_victim, k);
    z.trade_target = turned(a.trade_target, k);
    return r;
}

BeliefState rotated(const BeliefState& b, int k) {
    BeliefState r = b;
    for (int o = 0; o < NUM_PLAYERS; ++o) {
        for (int t = 0; t < NUM_PLAYERS; ++t) r.hand[turned(o, k)][turned(t, k)] = b.hand[o][t];
        r.vp_at_turn_end[turned(o, k)] = b.vp_at_turn_end[o];
    }
    return r;
}

template <typename T>
bool same_bytes(const T& a, const T& b) {
    return memcmp(&a, &b, sizeof a) == 0;
}

// ---- hand-built positions -------------------------------------------------------------------

const Board& std_board() {
    static const Board b = standard_board();
    return b;
}

// Player p to act in MAIN on a bare standard board: no buildings, empty hands, full deck.
GameState blank(uint8_t p = 0) {
    GameState s = random_setup(std_board(), 1);
    clear_board(s);
    clear_hands(s);
    set_main(s, p);
    return s;
}

// Belief of a crafted state in which every hand is common knowledge (as right after setup).
BeliefState open_belief(const GameState& s) {
    BeliefState b = initial_belief();
    for (int o = 0; o < NUM_PLAYERS; ++o) {
        for (int t = 0; t < NUM_PLAYERS; ++t) b.hand[o][t].known = s.hid.hands[t];
    }
    return b;
}

// Belief of a crafted state in which each observer knows only its own hand.
BeliefState opaque_belief(const GameState& s) {
    BeliefState b = initial_belief();
    for (int o = 0; o < NUM_PLAYERS; ++o) {
        for (int t = 0; t < NUM_PLAYERS; ++t) {
            if (o == t) {
                b.hand[o][t].known = s.hid.hands[t];
            } else {
                b.hand[o][t].unknown = s.pub.hand_size[t];
            }
        }
    }
    return b;
}

Observation see(const GameState& s, const BeliefState& b, uint8_t o) {
    return obs_of(std_board(), s, b, o);
}

// Every check the random games make, on one hand-built position.
void require_consistent(const GameState& s, const BeliefState& b) {
    Failure f;
    for (uint8_t o = 0; o < NUM_PLAYERS; ++o) {
        f.at(0, s.pub.ply, o);
        const Observation x = see(s, b, o);
        check_structure(std_board(), s, b, o, x, f);
        f.same(x, reference_obs(std_board(), s.pub, b, reference_view(s, o), o),
               "differs from the reference encoding");
    }
    check_mask(next_decision(std_board(), s), mask_of(next_decision(std_board(), s)), f);
    INFO(f.msg);
    REQUIRE(f.ok());
}

// Applies a legal action, or a chance outcome, and updates the belief.
void act(GameState& s, BeliefState& b, ActionType type, int arg = 0) {
    const Action a{type, static_cast<uint8_t>(arg)};
    REQUIRE(is_legal(next_decision(std_board(), s).mask, s.pub.phase, a));
    const EventList ev = apply(std_board(), s, a);
    observe(b, s.pub, ev);
}

void resolve(GameState& s, BeliefState& b, int outcome) {
    REQUIRE(next_decision(std_board(), s).kind == DecisionKind::CHANCE);
    const EventList ev = apply_chance(std_board(), s, static_cast<uint8_t>(outcome));
    observe(b, s.pub, ev);
}

// FNV-1a, 64-bit.
uint64_t fnv(uint64_t h, const uint8_t* p, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001B3ull;
    }
    return h;
}

}  // namespace

// ---- layout ---------------------------------------------------------------------------------

TEST_CASE("layout: the section table, token order and full coverage of every byte",
          "[encode][layout]") {
    // OBS_SECTIONS tiles the buffer in order.
    int at = 0;
    for (const ObsSection& s : OBS_SECTIONS) {
        CHECK(s.offset == at);
        CHECK(s.rows > 0);
        CHECK(s.width > 0);
        at = s.offset + s.rows * s.width;
    }
    CHECK(at == OBS_SIZE);
    CHECK(string(OBS_SECTIONS[0].name) == "tiles");
    CHECK(string(OBS_SECTIONS[1].name) == "self");
    CHECK(string(OBS_SECTIONS[2].name) == "players");
    CHECK(string(OBS_SECTIONS[3].name) == "global");
    CHECK(OBS_SECTIONS[0].rows == NUM_HEXES);
    CHECK(OBS_SECTIONS[2].rows == NUM_PLAYERS);

    // Tokens 2..12 without 7, ascending.
    CHECK(TOKEN_VALUES == array<uint8_t, 10>{2, 3, 4, 5, 6, 8, 9, 10, 11, 12});

    // Every byte is written, whatever the buffer held, for an observation and for each kind
    // of decision's mask.
    const Board board = random_board(3);
    GameState s = initial_state(board);
    BeliefState b = initial_belief();
    for (uint8_t fill : {uint8_t{0x00}, FILL, uint8_t{0xFF}}) {
        for (uint8_t o = 0; o < NUM_PLAYERS; ++o) {
            Observation x, y;
            x.fill(fill);
            y.fill(static_cast<uint8_t>(fill ^ 0x55));
            encode(board, s, b, o, x);
            encode(board, s, b, o, y);
            CHECK(x == y);
        }
        MaskBuffer m, n;
        m.fill(fill);
        n.fill(static_cast<uint8_t>(fill ^ 0x55));
        encode_mask(next_decision(board, s), m);
        encode_mask(next_decision(board, s), n);
        CHECK(m == n);
    }
    Decision chance{};
    chance.kind = DecisionKind::CHANCE;
    chance.mask.type = 0xFFFF;  // junk in a non-PLAYER decision's mask is ignored
    chance.mask.resource = 0xFF;
    CHECK(mask_of(chance) == MaskBuffer{});
    Decision terminal{};
    terminal.kind = DecisionKind::TERMINAL;
    terminal.phase = Phase::TERMINAL;
    CHECK(mask_of(terminal) == MaskBuffer{});
}

// ---- hidden information ---------------------------------------------------------------------

TEST_CASE("private_view holds exactly the observer's own cards", "[encode][leak]") {
    GameState s = blank(2);
    give(s, 0, Resource::BRICK, 2);
    give(s, 2, Resource::ORE, 3);
    give(s, 3, Resource::WOOL, 1);
    give_dev(s, 2, DevCard::KNIGHT, 2);
    give_dev(s, 0, DevCard::MONOPOLY, 1);
    give_dev(s, 3, DevCard::VICTORY_POINT, 1);
    s.hid.dev_bought_this_turn[idx(DevCard::KNIGHT)] = 1;  // seat 2 bought one this turn
    REQUIRE(invariant_violation(std_board(), s) == nullptr);
    BeliefState b = opaque_belief(s);
    b.bought_this_turn = 1;
    REQUIRE(belief_violation(b, s) == nullptr);

    for (uint8_t o = 0; o < NUM_PLAYERS; ++o) {
        INFO("observer " << int(o));
        const PrivateView v = private_view(s, o);
        CHECK(v.hand == s.hid.hands[o]);
        CHECK(v.dev_held == s.hid.dev_held[o]);
        if (o == 2) {
            CHECK(v.dev_bought_this_turn == s.hid.dev_bought_this_turn);
        } else {
            CHECK(v.dev_bought_this_turn == array<uint8_t, NUM_DEV>{});
        }
        // The convenience overload is the explicit one with private_view.
        Observation x;
        x.fill(FILL);
        encode(std_board(), s.pub, b, v, o, x);
        CHECK(x == see(s, b, o));
    }
    require_consistent(s, b);
}

TEST_CASE("no hidden information reaches an observation", "[encode][leak]") {
    // For every observer at sampled points of random games: resampling the hidden state as
    // that observer (the header's invariant), overwriting everything it cannot see with junk,
    // and overwriting the other observers' belief rows must all leave the encoding unchanged.
    Failure f;
    long points = 0, resamples = 0, moved = 0;
    for (uint64_t seed = 1; seed <= games() && f.ok(); ++seed) {
        const Board board = board_for(seed);
        play_tracked(board, seed, [&](const GameState& s, const BeliefState& b,
                                      const Decision& d) {
            if (!f.ok()) return;
            // Points with publicly implied hand constraints (discards, exchanges, steals,
            // purchases) are rarer and more delicate, so take them all.
            const Phase ph = s.pub.phase;
            const bool rare = ph == Phase::DISCARD || ph == Phase::EXCHANGE_RECV ||
                              ph == Phase::STEAL || d.kind == DecisionKind::CHANCE ||
                              s.pub.played_dev_this_turn || ph == Phase::TERMINAL;
            if (!rare && random_below(seed, 910, s.pub.ply, 2) != 0) return;
            ++points;
            for (uint8_t o = 0; o < NUM_PLAYERS; ++o) {
                f.at(seed, s.pub.ply, o);
                const uint64_t key = seed << 32 | uint64_t{s.pub.ply} << 4 | o;
                const Observation base = obs_of(board, s, b, o);

                const PrivateView view = private_view(s, o);
                f.expect(same_view(view, reference_view(s, o)),
                         "private_view is not exactly the observer's own cards");
                Observation pure;
                pure.fill(FILL);
                encode(board, s.pub, b, view, o, pure);
                f.same(pure, base, "explicit and convenience overloads differ");

                for (uint64_t k = 0; k < 3; ++k) {
                    const GameState r = resample_hidden(board, s, b, o, key * 3 + k);
                    ++resamples;
                    moved += !same_bytes(r.hid, s.hid);
                    f.same(obs_of(board, r, b, o), base, "a resampled state encodes differently");
                }

                GameState junk = s;
                scramble_hidden(junk, o, key);
                f.expect(same_view(private_view(junk, o), view),
                         "private_view changed with hidden data of others");
                f.same(obs_of(board, junk, b, o), base,
                       "others' hidden cards or the deck changed the observation");

                BeliefState other = b;
                scramble_other_rows(other, o, key);
                f.same(obs_of(board, s, other, o), base,
                       "another observer's belief row changed the observation");
            }
        });
    }
    INFO(f.msg);
    REQUIRE(f.ok());
    INFO("points " << points << " resamples " << resamples << " moved " << moved);
    REQUIRE(points >= static_cast<long>(games()) * 100);
    REQUIRE(moved * 2 > resamples);  // the resamples really did move hidden cards
}

// ---- fuzzed structure and reference ---------------------------------------------------------

TEST_CASE("every observation and mask of random games matches the contract",
          "[encode][fuzz]") {
    Failure f;
    long points = 0, saturated = 0;
    array<long, NUM_PHASES> phases{};
    for (uint64_t seed = 201; seed <= 200 + games() && f.ok(); ++seed) {
        const Board board = board_for(seed);
        play_tracked(board, seed, [&](const GameState& s, const BeliefState& b,
                                      const Decision& d) {
            if (!f.ok()) return;
            ++points;
            ++phases[static_cast<int>(s.pub.phase)];
            saturated += s.pub.turn > 255;
            f.at(seed, s.pub.ply);
            check_mask(d, mask_of(d), f);
            for (uint8_t o = 0; o < NUM_PLAYERS; ++o) {
                f.at(seed, s.pub.ply, o);
                const Observation x = obs_of(board, s, b, o);
                check_structure(board, s, b, o, x, f);
                f.same(x, reference_obs(board, s.pub, b, reference_view(s, o), o),
                       "differs from the reference encoding");
            }
        });
    }
    INFO(f.msg);
    REQUIRE(f.ok());
    INFO("points " << points << " checks " << f.checks);
    REQUIRE(points >= static_cast<long>(games()) * 200);
    if (games() >= 20) {
        // Enough games reach every phase and a turn count past the saturation point.
        for (int p = 0; p < NUM_PHASES; ++p) {
            INFO("phase " << p);
            CHECK(phases[p] > 0);
        }
        CHECK(saturated > 0);
    }
}

// ---- seat rotation --------------------------------------------------------------------------

TEST_CASE("rotating every seat together with the observer changes nothing",
          "[encode][rotation]") {
    // encode(rotated(s, k), o + k) == encode(s, o) for every k and o. Masks: the mask is
    // absolute in vertices, edges and tiles and relative to the decider in PLAYER, so the
    // rotated state's next decision must have decider + k and the identical mask. That is
    // checked outside SETUP only: setup order is absolute (setup_player(step)), so a rotated
    // SETUP state has the wrong seat to move and next_decision rejects it. Observations are
    // compared in SETUP too: the encoder must take the mover from current_player, not from
    // the setup order.
    Failure f;
    long points = 0, masks = 0;
    for (uint64_t seed = 401; seed <= 400 + games() && f.ok(); ++seed) {
        const Board board = board_for(seed);
        play_tracked(board, seed, [&](const GameState& s, const BeliefState& b,
                                      const Decision& d) {
            if (!f.ok()) return;
            ++points;
            const bool setup = s.pub.phase == Phase::SETUP;
            array<Observation, NUM_PLAYERS> base;
            for (uint8_t o = 0; o < NUM_PLAYERS; ++o) base[o] = obs_of(board, s, b, o);
            const MaskBuffer base_mask = mask_of(d);
            for (int k = 1; k < NUM_PLAYERS; ++k) {
                f.at(seed, s.pub.ply);
                const GameState rs = rotated(s, k);
                const BeliefState rb = rotated(b, k);
                // The helper itself: rotations compose to the identity and keep play legal.
                f.expect(same_bytes(rotated(rs, NUM_PLAYERS - k), s) &&
                             same_bytes(rotated(rb, NUM_PLAYERS - k), b),
                         "rotation helper is not invertible");
                if (!setup) {
                    f.expect(invariant_violation(board, rs) == nullptr,
                             "rotated state breaks an invariant");
                    f.expect(belief_violation(rb, rs) == nullptr, "rotated belief is unsound");
                }
                for (uint8_t o = 0; o < NUM_PLAYERS; ++o) {
                    f.at(seed, s.pub.ply, o);
                    f.same(obs_of(board, rs, rb, seat_at(o, k)), base[o],
                           "rotating seats and observer changed the observation");
                }
                if (setup) continue;
                const Decision rd = next_decision(board, rs);
                ++masks;
                f.expect(rd.kind == d.kind && rd.phase == d.phase &&
                             rd.player == turned(d.player, k),
                         "the rotated state's decider is not the decider + k");
                f.expect(mask_of(rd) == base_mask, "rotation changed the mask");
            }
        });
    }
    INFO(f.msg);
    REQUIRE(f.ok());
    INFO("points " << points << " masks " << masks);
    REQUIRE(points >= static_cast<long>(games()) * 300);
    REQUIRE(masks > points);
}

// ---- tiles ----------------------------------------------------------------------------------

TEST_CASE("tiles: hex type, token one-hot, pips and robber", "[encode][tiles]") {
    GameState s = blank(0);
    const BeliefState b = open_belief(s);
    struct Expect {
        int hex;
        HexType type;
        int token;  // index into TOKEN_VALUES, -1 for none
        int pips;
    };
    // Standard board: hex 4 is brick 6, 11 ore 8, 1 wool 2, 3 grain 12, 9 the desert.
    const array<Expect, 9> cases = {{
        {4, HexType::BRICK, 4, 5},
        {11, HexType::ORE, 5, 5},
        {1, HexType::WOOL, 0, 1},
        {3, HexType::GRAIN, 9, 1},
        {9, HexType::DESERT, -1, 0},
        {0, HexType::ORE, 7, 3},     // 10
        {13, HexType::ORE, 1, 2},    // 3
        {15, HexType::WOOL, 3, 4},   // 5
        {8, HexType::LUMBER, 8, 2},  // 11
    }};
    for (uint8_t o : {0, 3}) {
        const Observation x = see(s, b, o);
        for (const Expect& e : cases) {
            INFO("observer " << int(o) << " hex " << e.hex);
            const uint8_t* t = tile_of(x, e.hex);
            CHECK(hot(t + TILE_HEX_TYPE, NUM_HEX_TYPES) == static_cast<int>(e.type));
            CHECK(hot(t + TILE_TOKEN, NUM_TOKENS) == e.token);
            CHECK(t[TILE_PIPS] == e.pips);
        }
        // Literal offsets: hex 4's six is token byte 6 + 4, its pips byte 16.
        CHECK(x[4 * 102 + 10] == 1);
        CHECK(x[4 * 102 + 16] == 5);
        CHECK(x[9 * 102 + 5] == 1);  // desert type
        for (int h = 0; h < NUM_HEXES; ++h) {
            INFO("hex " << h);
            CHECK(tile_of(x, h)[TILE_ROBBER] == (h == 9));
        }
    }
    s.pub.robber_hex = 4;
    const Observation x = see(s, b, 1);
    for (int h = 0; h < NUM_HEXES; ++h) CHECK(tile_of(x, h)[TILE_ROBBER] == (h == 4));
    require_consistent(s, b);
}

TEST_CASE("tiles: a building shows on every corner it occupies, relative to the observer",
          "[encode][tiles]") {
    const Topology& topo = topology();
    REQUIRE(topo.hex_vertices[9][0] == 18);
    REQUIRE(topo.hex_vertices[9][3] == 35);
    GameState s = blank(0);
    place_settlement(s, 1, 18);
    place_settlement(s, 2, 35);
    place_city(s, 2, 35);
    REQUIRE(invariant_violation(std_board(), s) == nullptr);
    const BeliefState b = open_belief(s);

    // Observer 0: seat 1's settlement on corner 0 is rel 1, seat 2's city on corner 3 rel 2.
    const Observation x0 = see(s, b, 0);
    const uint8_t* t = tile_of(x0, 9);
    CHECK(t[18 + 8 * 0 + 2 * 1 + 0] == 1);  // byte 20
    CHECK(t[18 + 8 * 3 + 2 * 2 + 1] == 1);  // byte 47
    CHECK(nonzero(t + TILE_CORNERS, 48) == 2);

    // Observer 2: the settlement is rel 3 and the city rel 0.
    const Observation x2 = see(s, b, 2);
    t = tile_of(x2, 9);
    CHECK(t[18 + 2 * 3] == 1);          // byte 24
    CHECK(t[18 + 8 * 3 + 0 + 1] == 1);  // byte 43
    CHECK(nonzero(t + TILE_CORNERS, 48) == 2);

    // The same bytes on every other tile the vertices touch, at that tile's corner index.
    int touching = 0;
    for (int v : {18, 35}) {
        for (uint8_t h : topo.vertex_hexes[v]) {
            if (h == NONE) continue;
            ++touching;
            int corner = -1;
            for (int i = 0; i < SIDES; ++i) {
                if (topo.hex_vertices[h][i] == v) corner = i;
            }
            INFO("vertex " << v << " hex " << int(h) << " corner " << corner);
            const int byte = v == 18 ? 2 * 1 : 2 * 2 + 1;  // from observer 0
            CHECK(tile_of(x0, h)[TILE_CORNERS + 8 * corner + byte] == 1);
        }
    }
    CHECK(touching == 6);  // both are inland vertices
    int set = 0;
    for (int h = 0; h < NUM_HEXES; ++h) set += nonzero(tile_of(x0, h) + TILE_CORNERS, 48);
    CHECK(set == touching);
    require_consistent(s, b);
}

TEST_CASE("tiles: a road shows on both tiles of its edge, relative to the observer",
          "[encode][tiles]") {
    const Topology& topo = topology();
    REQUIRE(topo.hex_edges[9][1] == 36);   // centre, east side
    REQUIRE(topo.hex_edges[10][4] == 36);  // its east neighbour, west side
    REQUIRE(topo.hex_edges[0][5] == 0);    // a coastal edge: one tile only
    GameState s = blank(0);
    place_road(s, 3, 36);
    place_road(s, 0, 0);
    const BeliefState b = open_belief(s);

    const Observation x0 = see(s, b, 0);
    CHECK(tile_of(x0, 9)[66 + 4 * 1 + 3] == 1);   // byte 73
    CHECK(tile_of(x0, 10)[66 + 4 * 4 + 3] == 1);  // byte 85
    CHECK(tile_of(x0, 0)[66 + 4 * 5 + 0] == 1);   // byte 86
    const Observation x3 = see(s, b, 3);
    CHECK(tile_of(x3, 9)[66 + 4 * 1 + 0] == 1);
    CHECK(tile_of(x3, 10)[66 + 4 * 4 + 0] == 1);
    CHECK(tile_of(x3, 0)[66 + 4 * 5 + 1] == 1);
    for (const Observation* x : {&x0, &x3}) {
        int set = 0;
        for (int h = 0; h < NUM_HEXES; ++h) set += nonzero(tile_of(*x, h) + TILE_ROADS, 24);
        CHECK(set == 3);
    }
    require_consistent(s, b);
}

TEST_CASE("tiles: each harbor shows its type and side on the tile it borders",
          "[encode][tiles]") {
    const Topology& topo = topology();
    struct Harbor {
        int hex, side;
        PortType type;
    };
    // topology.h's slot table with the beginner map's harbors (board.cpp).
    const array<Harbor, NUM_PORTS> standard = {{
        {0, 5, PortType::GENERIC},
        {1, 0, PortType::GRAIN},
        {6, 0, PortType::ORE},
        {11, 1, PortType::GENERIC},
        {15, 2, PortType::WOOL},
        {17, 2, PortType::GENERIC},
        {16, 3, PortType::GENERIC},
        {12, 4, PortType::BRICK},
        {3, 4, PortType::LUMBER},
    }};
    for (int slot = 0; slot < NUM_PORTS; ++slot) {
        const Harbor& hb = standard[slot];
        REQUIRE(topo.port_slots[slot].hex == hb.hex);
        REQUIRE(topo.hex_edges[hb.hex][hb.side] == topo.port_slots[slot].edge);
    }
    GameState s = blank(0);
    const BeliefState b = open_belief(s);
    for (uint8_t o : {0, 1}) {
        const Observation x = see(s, b, o);
        // Slot 0: hex 0, generic, on side 5 (NW).
        CHECK(x[0 * 102 + 90 + 5] == 1);
        CHECK(x[0 * 102 + 96 + 5] == 1);
        for (int h = 0; h < NUM_HEXES; ++h) {
            INFO("observer " << int(o) << " hex " << h);
            const uint8_t* t = tile_of(x, h);
            int type = -1, side = -1;
            for (const Harbor& hb : standard) {
                if (hb.hex != h) continue;
                type = static_cast<int>(hb.type);
                side = hb.side;
            }
            CHECK(hot(t + TILE_PORT_TYPE, NUM_PORT_TYPES) == type);
            CHECK(hot(t + TILE_PORT_SIDE, SIDES) == side);
        }
    }

    // On a shuffled board the types follow board.port_type while the sides stay.
    const Board board = random_board(11);
    const GameState r = initial_state(board);
    const Observation x = obs_of(board, r, initial_belief(), 2);
    for (int slot = 0; slot < NUM_PORTS; ++slot) {
        const uint8_t* t = tile_of(x, standard[slot].hex);
        CHECK(hot(t + TILE_PORT_TYPE, NUM_PORT_TYPES) == static_cast<int>(board.port_type[slot]));
        CHECK(hot(t + TILE_PORT_SIDE, SIDES) == standard[slot].side);
    }
}

// ---- player rows ----------------------------------------------------------------------------

TEST_CASE("players: production counts settlements once and cities twice, without the robber "
          "hex or the desert",
          "[encode][players]") {
    const Topology& topo = topology();
    // Hex 4 is brick 6; its corners 0, 2, 4 are vertices 8, 18, 17.
    REQUIRE(topo.hex_vertices[4][0] == 8);
    REQUIRE(topo.hex_vertices[4][2] == 18);
    REQUIRE(topo.hex_vertices[4][4] == 17);
    REQUIRE(topo.vertex_hexes[8] == array<uint8_t, 3>{0, 1, 4});    // ore 10, wool 2, brick 6
    REQUIRE(topo.vertex_hexes[18] == array<uint8_t, 3>{4, 5, 9});   // brick 6, wool 4, desert
    REQUIRE(topo.vertex_hexes[17] == array<uint8_t, 3>{3, 4, 8});   // grain 12, brick 6, lumber 11
    GameState s = blank(0);
    place_settlement(s, 1, 8);
    place_settlement(s, 1, 18);
    place_city(s, 1, 18);
    place_settlement(s, 2, 17);
    REQUIRE(invariant_violation(std_board(), s) == nullptr);
    const BeliefState b = open_belief(s);

    // Production index: 10 * resource + token index (2 3 4 5 6 8 9 10 11 12).
    constexpr int BRICK_6 = 10 * 0 + 4, ORE_10 = 10 * 4 + 7, WOOL_2 = 10 * 2 + 0,
                  WOOL_4 = 10 * 2 + 2, GRAIN_12 = 10 * 3 + 9, LUMBER_11 = 10 * 1 + 8;
    array<uint8_t, 50> one{}, two{};
    one[BRICK_6] = 3;  // settlement 1 + city 2 on the same hex
    one[ORE_10] = 1;
    one[WOOL_2] = 1;
    one[WOOL_4] = 2;  // the city's third hex is the desert: nothing
    two[BRICK_6] = 1;
    two[GRAIN_12] = 1;
    two[LUMBER_11] = 1;

    auto check = [&](const GameState& st) {
        for (uint8_t o = 0; o < NUM_PLAYERS; ++o) {
            const Observation x = see(st, b, o);
            for (int k = 0; k < NUM_PLAYERS; ++k) {
                INFO("observer " << int(o) << " row " << k << " robber " << int(st.pub.robber_hex));
                const uint8_t t = seat_at(o, k);
                array<uint8_t, 50> got;
                copy_n(row_of(x, k) + PLAYER_PRODUCTION, 50, got.begin());
                const array<uint8_t, 50> want = t == 1 ? one : t == 2 ? two : array<uint8_t, 50>{};
                CHECK(got == want);
            }
        }
    };
    check(s);
    CHECK(see(s, b, 0)[OBS_PLAYERS + 1 * 79 + 29 + BRICK_6] == 3);  // literal offset, row 1

    s.pub.robber_hex = 4;  // blocks brick 6 for both
    one[BRICK_6] = 0;
    two[BRICK_6] = 0;
    check(s);

    s.pub.robber_hex = 5;  // blocks wool 4 instead
    one[BRICK_6] = 3;
    two[BRICK_6] = 1;
    one[WOOL_4] = 0;
    check(s);
    require_consistent(s, b);
}

TEST_CASE("players: harbor flags come from buildings on harbor vertices, not roads",
          "[encode][players]") {
    REQUIRE(std_board().vertex_port[1] == static_cast<uint8_t>(PortType::GRAIN));
    REQUIRE(std_board().vertex_port[52] == static_cast<uint8_t>(PortType::GENERIC));
    REQUIRE(topology().edge_vertices[0] == array<uint8_t, 2>{0, 3});  // generic harbor edge
    GameState s = blank(0);
    place_settlement(s, 3, 1);
    place_settlement(s, 3, 52);
    place_city(s, 3, 52);
    place_road(s, 0, 0);
    const BeliefState b = open_belief(s);
    for (uint8_t o = 0; o < NUM_PLAYERS; ++o) {
        const Observation x = see(s, b, o);
        for (int k = 0; k < NUM_PLAYERS; ++k) {
            INFO("observer " << int(o) << " row " << k);
            const uint8_t* ports = row_of(x, k) + PLAYER_PORTS;
            array<uint8_t, NUM_PORT_TYPES> got;
            copy_n(ports, NUM_PORT_TYPES, got.begin());
            const array<uint8_t, NUM_PORT_TYPES> want =
                seat_at(o, k) == 3 ? array<uint8_t, NUM_PORT_TYPES>{0, 0, 0, 1, 0, 1}
                                   : array<uint8_t, NUM_PORT_TYPES>{};
            CHECK(got == want);
        }
    }
    require_consistent(s, b);
}

TEST_CASE("players: each row holds its seat's public facts and the observer's belief",
          "[encode][players]") {
    GameState s = blank(2);
    for (uint8_t p = 0; p < NUM_PLAYERS; ++p) give(s, p, Resource::GRAIN, p + 1);
    give(s, 3, Resource::ORE, 2);
    give_dev(s, 1, DevCard::KNIGHT, 2);
    give_dev(s, 3, DevCard::MONOPOLY, 1);
    place_settlement(s, 0, 8);
    place_settlement(s, 3, 18);
    place_city(s, 3, 18);
    for (auto [a, z] : {pair{1, 5}, {5, 9}, {9, 14}, {14, 19}, {19, 25}}) {
        place_road(s, 1, edge_between(a, z));
    }
    refresh_awards(s);
    REQUIRE(s.pub.longest_road_holder == 1);
    // Seat 2 played three knights from the deck and holds Largest Army.
    s.hid.dev_deck[idx(DevCard::KNIGHT)] = static_cast<uint8_t>(
        s.hid.dev_deck[idx(DevCard::KNIGHT)] - 3);
    s.pub.dev_deck_remaining = static_cast<uint8_t>(s.pub.dev_deck_remaining - 3);
    s.pub.knights_played[2] = 3;
    s.pub.largest_army_holder = 2;
    s.pub.public_vp[2] = static_cast<uint8_t>(s.pub.public_vp[2] + 2);
    REQUIRE(invariant_violation(std_board(), s) == nullptr);
    BeliefState b = open_belief(s);
    b.dev_played[idx(DevCard::KNIGHT)] = 3;
    b.vp_at_turn_end = {1, 3, 0, 2};
    REQUIRE(belief_violation(b, s) == nullptr);

    for (uint8_t o = 0; o < NUM_PLAYERS; ++o) {
        const Observation x = see(s, b, o);
        for (int k = 0; k < NUM_PLAYERS; ++k) {
            const uint8_t t = seat_at(o, k);
            INFO("observer " << int(o) << " row " << k << " seat " << int(t));
            const uint8_t* p = row_of(x, k);
            for (int r = 0; r < NUM_RES; ++r) {
                CHECK(p[PLAYER_HAND_MIN + r] == s.hid.hands[t][r]);  // open hands: exact
                CHECK(p[PLAYER_HAND_MAX + r] == s.hid.hands[t][r]);
            }
            CHECK(p[PLAYER_HAND_SIZE] == s.pub.hand_size[t]);
            CHECK(p[PLAYER_DEV_COUNT] == s.pub.dev_count[t]);
            CHECK(p[PLAYER_KNIGHTS] == s.pub.knights_played[t]);
            CHECK(p[PLAYER_PUBLIC_VP] == s.pub.public_vp[t]);
            CHECK(p[PLAYER_ROAD_LEN] == s.pub.road_len[t]);
            for (int i = 0; i < 3; ++i) CHECK(p[PLAYER_PIECES_LEFT + i] == s.pub.pieces_left[t][i]);
            CHECK(p[PLAYER_PENDING_DISCARD] == 0);
            CHECK(p[PLAYER_LONGEST_ROAD] == (t == 1));
            CHECK(p[PLAYER_LARGEST_ARMY] == (t == 2));
            CHECK(p[PLAYER_IS_CURRENT] == (t == 2));
            CHECK(p[PLAYER_VP_AT_TURN_END] == b.vp_at_turn_end[t]);
        }
        CHECK(global_of(x)[GLOBAL_DEV_PLAYED + idx(DevCard::KNIGHT)] == 3);
        CHECK(global_of(x)[GLOBAL_DEV_DECK] == DEV_DECK_SIZE - 6);
    }
    // Literal offsets from observer 0: seat 1's road length and award, seat 2's army.
    const Observation x0 = see(s, b, 0);
    CHECK(x0[1994 + 79 * 1 + 14] == 5);
    CHECK(x0[1994 + 79 * 1 + 19] == 1);
    CHECK(x0[1994 + 79 * 2 + 12] == 3);
    CHECK(x0[1994 + 79 * 2 + 20] == 1);
    CHECK(x0[1994 + 79 * 2 + 21] == 1);
    CHECK(x0[1994 + 79 * 3 + 15 + PIECE_CITY] == CITIES_PER_PLAYER - 1);
    require_consistent(s, b);

    // Bounds follow hand_min / hand_max: observer 0 sees seat 3's 4 grain and 2 ore as six
    // unknown cards, capped per type by what 0 has not seen (grain 9, ore 2, others 0).
    const BeliefState hidden = opaque_belief(s);
    const Observation xh = see(s, hidden, 0);
    const uint8_t* p = row_of(xh, 3);
    CHECK(array<uint8_t, 5>{p[0], p[1], p[2], p[3], p[4]} == array<uint8_t, 5>{});
    CHECK(array<uint8_t, 5>{p[5], p[6], p[7], p[8], p[9]} == array<uint8_t, 5>{0, 0, 0, 6, 2});
    require_consistent(s, hidden);

    // A pending discard is shown on the owing seat's row.
    s.pub.phase = Phase::DISCARD;
    s.pub.dice = 7;
    give(s, 1, Resource::LUMBER, 8);
    s.pub.pending_discard[1] = static_cast<uint8_t>(s.pub.hand_size[1] / 2);
    const BeliefState b2 = open_belief(s);
    REQUIRE(invariant_violation(std_board(), s) == nullptr);
    for (uint8_t o = 0; o < NUM_PLAYERS; ++o) {
        CHECK(row_of(see(s, b2, o), rel(1, o))[PLAYER_PENDING_DISCARD] == 5);
    }
    require_consistent(s, b2);
}

// ---- self block -----------------------------------------------------------------------------

TEST_CASE("self: hand buckets at their edges and exact counts", "[encode][self]") {
    // Buckets 0, 1, 2, 3, 4, 5, 6-7, 8+.
    constexpr array<int, 20> bucket = {0, 1, 2, 3, 4, 5, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7};
    for (int n = 0; n <= RESOURCE_PER_TYPE; ++n) {
        INFO("ore " << n);
        GameState s = blank(1);
        give(s, 1, Resource::ORE, n);
        give(s, 1, Resource::WOOL, 1);
        const BeliefState b = open_belief(s);
        const Observation x = see(s, b, 1);
        const uint8_t* me = self_of(x);
        const int ore = idx(Resource::ORE), wool = idx(Resource::WOOL);
        CHECK(hot(me + SELF_HAND_BUCKET + 8 * ore, 8) == bucket[n]);
        CHECK(x[OBS_SELF + 8 * ore + bucket[n]] == 1);
        CHECK(hot(me + SELF_HAND_BUCKET + 8 * wool, 8) == 1);
        CHECK(hot(me + SELF_HAND_BUCKET + 0, 8) == 0);  // no brick: bucket 0
        CHECK(me[SELF_HAND + ore] == n);
        CHECK(row_of(x, 0)[PLAYER_HAND_MIN + ore] == n);
        CHECK(row_of(x, 0)[PLAYER_HAND_MAX + ore] == n);
        // An opponent that knows nothing: min 0, max = what it has not seen.
        const Observation y = see(s, opaque_belief(s), 0);
        CHECK(row_of(y, 1)[PLAYER_HAND_MIN + ore] == 0);
        CHECK(row_of(y, 1)[PLAYER_HAND_MAX + ore] == n);
        CHECK(row_of(y, 1)[PLAYER_HAND_MAX + wool] == 1);
        CHECK(row_of(y, 1)[PLAYER_HAND_SIZE] == n + 1);
    }
}

TEST_CASE("self: dev cards bought this turn are shown to the buyer only", "[encode][self]") {
    GameState s = blank(2);
    give(s, 2, Resource::WOOL, 1);
    give(s, 2, Resource::GRAIN, 1);
    give(s, 2, Resource::ORE, 1);
    BeliefState b = open_belief(s);
    const int knight = idx(DevCard::KNIGHT);
    act(s, b, ActionType::BUY_DEV);
    resolve(s, b, knight);
    REQUIRE(s.hid.dev_bought_this_turn[knight] == 1);

    const Observation x2 = see(s, b, 2);
    CHECK(self_of(x2)[SELF_DEV_HELD + knight] == 1);
    CHECK(self_of(x2)[SELF_DEV_BOUGHT + knight] == 1);
    CHECK(global_of(x2)[GLOBAL_BOUGHT_THIS_TURN] == 1);
    for (uint8_t o : {0, 1, 3}) {
        INFO("observer " << int(o));
        CHECK(private_view(s, o).dev_bought_this_turn == array<uint8_t, NUM_DEV>{});
        const Observation x = see(s, b, o);
        CHECK(nonzero(self_of(x) + SELF_DEV_HELD, NUM_DEV) == 0);
        CHECK(nonzero(self_of(x) + SELF_DEV_BOUGHT, NUM_DEV) == 0);
        CHECK(row_of(x, rel(2, o))[PLAYER_DEV_COUNT] == 1);
        CHECK(global_of(x)[GLOBAL_BOUGHT_THIS_TURN] == 1);  // public: that a card was bought
    }
    require_consistent(s, b);

    act(s, b, ActionType::END_TURN);
    const Observation after = see(s, b, 2);
    CHECK(self_of(after)[SELF_DEV_HELD + knight] == 1);
    CHECK(self_of(after)[SELF_DEV_BOUGHT + knight] == 0);
    CHECK(global_of(after)[GLOBAL_BOUGHT_THIS_TURN] == 0);
}

TEST_CASE("self: total VP adds hidden VP cards to public VP", "[encode][self]") {
    GameState s = blank(1);
    place_settlement(s, 1, 8);
    place_settlement(s, 1, 18);
    place_city(s, 1, 18);
    give_dev(s, 1, DevCard::VICTORY_POINT, 2);
    REQUIRE(s.pub.public_vp[1] == 3);
    const BeliefState b = open_belief(s);
    const Observation x1 = see(s, b, 1);
    CHECK(self_of(x1)[SELF_TOTAL_VP] == 5);
    CHECK(x1[OBS_SELF + 55] == 5);
    CHECK(self_of(x1)[SELF_DEV_HELD + VP] == 2);
    CHECK(row_of(x1, 0)[PLAYER_PUBLIC_VP] == 3);
    const Observation x0 = see(s, b, 0);
    CHECK(self_of(x0)[SELF_TOTAL_VP] == 0);
    CHECK(row_of(x0, 1)[PLAYER_PUBLIC_VP] == 3);
    CHECK(row_of(x0, 1)[PLAYER_DEV_COUNT] == 2);
    require_consistent(s, b);
}

// ---- global block ---------------------------------------------------------------------------

TEST_CASE("global: the turn saturates at 255", "[encode][global]") {
    GameState s = blank(0);
    const BeliefState b = open_belief(s);
    for (auto [turn, want] : {pair{0, 0}, {1, 1}, {254, 254}, {255, 255}, {256, 255},
                              {999, 255}, {65535, 255}}) {
        s.pub.turn = static_cast<uint16_t>(turn);
        CHECK(global_of(see(s, b, 0))[GLOBAL_TURN] == want);
    }
}

TEST_CASE("global: the dice show this turn's roll only", "[encode][global]") {
    GameState s = random_setup(std_board(), 3);
    BeliefState b = open_belief(s);  // after setup every hand is public
    REQUIRE(s.pub.phase == Phase::ROLL);
    auto dice = [&](uint8_t o) { return hot(global_of(see(s, b, o)) + GLOBAL_DICE, NUM_SUMS); };
    auto phase = [&] { return hot(global_of(see(s, b, 0)) + GLOBAL_PHASE, NUM_PHASES); };
    CHECK(dice(0) == -1);
    CHECK(phase() == static_cast<int>(Phase::ROLL));
    act(s, b, ActionType::ROLL);
    CHECK(dice(1) == -1);  // the roll is owed, not yet made
    resolve(s, b, 6);      // an 8
    REQUIRE(s.pub.dice == 8);
    for (uint8_t o = 0; o < NUM_PLAYERS; ++o) CHECK(dice(o) == 6);
    CHECK(phase() == static_cast<int>(Phase::MAIN));
    act(s, b, ActionType::END_TURN);
    CHECK(dice(2) == -1);
    act(s, b, ActionType::ROLL);
    resolve(s, b, 5);  // a 7
    REQUIRE(s.pub.dice == 7);
    CHECK(dice(3) == 5);
    CHECK(phase() != static_cast<int>(Phase::MAIN));
    require_consistent(s, b);
}

TEST_CASE("global: the bank exchange's give resource shows only in EXCHANGE_RECV",
          "[encode][global]") {
    GameState s = blank(0);
    give(s, 0, Resource::WOOL, 4);
    BeliefState b = open_belief(s);
    const int wool = idx(Resource::WOOL), ore = idx(Resource::ORE);
    auto give_byte = [&](uint8_t o) {
        return hot(global_of(see(s, b, o)) + GLOBAL_EXCHANGE_GIVE, NUM_RES);
    };
    CHECK(give_byte(0) == -1);
    MaskBuffer m = mask_of(next_decision(std_board(), s));
    CHECK(m[MASK_TYPE + idx(ActionType::BANK_EXCHANGE)] == 1);
    CHECK(m[MASK_RESOURCE + wool] == 1);
    CHECK(nonzero(m.data() + MASK_RESOURCE, NUM_RES) == 1);

    act(s, b, ActionType::BANK_EXCHANGE, wool);
    REQUIRE(s.pub.phase == Phase::EXCHANGE_RECV);
    for (uint8_t o = 0; o < NUM_PLAYERS; ++o) CHECK(give_byte(o) == wool);
    CHECK(global_of(see(s, b, 0))[GLOBAL_EXCHANGE_GIVE + wool] == 1);
    m = mask_of(next_decision(std_board(), s));
    CHECK(m[MASK_TYPE + idx(ActionType::BANK_EXCHANGE)] == 1);
    CHECK(m[MASK_RESOURCE + wool] == 0);  // the receive pick: anything but wool
    CHECK(nonzero(m.data() + MASK_RESOURCE, NUM_RES) == NUM_RES - 1);
    require_consistent(s, b);

    act(s, b, ActionType::BANK_EXCHANGE, ore);
    REQUIRE(s.pub.phase == Phase::MAIN);
    CHECK(give_byte(0) == -1);
    require_consistent(s, b);
}

// ---- masks ----------------------------------------------------------------------------------

TEST_CASE("mask: each head has its own section; PLAYER flags are relative to the decider",
          "[encode][mask]") {
    SECTION("settlement and city vertices are separate") {
        GameState s = blank(0);
        place_settlement(s, 0, 8);
        place_road(s, 0, edge_between(8, 13));
        place_road(s, 0, edge_between(13, 18));
        refresh_awards(s);
        give(s, 0, Resource::BRICK, 1);
        give(s, 0, Resource::LUMBER, 1);
        give(s, 0, Resource::WOOL, 1);
        give(s, 0, Resource::GRAIN, 3);
        give(s, 0, Resource::ORE, 3);
        REQUIRE(invariant_violation(std_board(), s) == nullptr);
        const Decision d = next_decision(std_board(), s);
        const MaskBuffer m = mask_of(d);
        // Settle only at 18 (13 is next to 8); upgrade only 8.
        CHECK(nonzero(m.data() + MASK_SETTLEMENT, NUM_VERTICES) == 1);
        CHECK(m[13 + 18] == 1);
        CHECK(nonzero(m.data() + MASK_CITY, NUM_VERTICES) == 1);
        CHECK(m[67 + 8] == 1);
        for (ActionType t : {ActionType::BUILD_ROAD, ActionType::BUILD_SETTLEMENT,
                             ActionType::BUILD_CITY, ActionType::BUY_DEV, ActionType::END_TURN}) {
            CHECK(m[MASK_TYPE + idx(t)] == 1);
        }
        CHECK(nonzero(m.data() + MASK_TYPE, NUM_ACTION_TYPES) == 5);
        Failure f;
        check_mask(d, m, f);
        INFO(f.msg);
        CHECK(f.ok());
    }

    SECTION("steal targets are seats after the decider") {
        GameState s = blank(1);
        set_main(s, 1, 7);
        place_settlement(s, 3, 8);   // hex 4 corner 0
        place_settlement(s, 0, 18);  // corner 2
        place_settlement(s, 2, 17);  // corner 4, but an empty hand
        give(s, 3, Resource::ORE, 1);
        give(s, 0, Resource::BRICK, 2);
        s.pub.robber_hex = 4;
        s.pub.phase = Phase::STEAL;
        REQUIRE(invariant_violation(std_board(), s) == nullptr);
        const Decision d = next_decision(std_board(), s);
        REQUIRE(d.player == 1);
        const MaskBuffer m = mask_of(d);
        CHECK(m[MASK_TYPE + idx(ActionType::STEAL)] == 1);
        CHECK(m[217 + 0] == 0);  // seat 2, next after 1: nothing to steal
        CHECK(m[217 + 1] == 1);  // seat 3
        CHECK(m[217 + 2] == 1);  // seat 0
        Failure f;
        check_mask(d, m, f);
        INFO(f.msg);
        CHECK(f.ok());
    }

    SECTION("Year of Plenty picks use the resource section under PLAY_DEV") {
        GameState s = blank(0);
        s.pub.phase = Phase::YOP_PICK;
        s.pub.yop_remaining = 2;
        s.pub.played_dev_this_turn = 1;
        REQUIRE(invariant_violation(std_board(), s) == nullptr);
        const Decision d = next_decision(std_board(), s);
        const MaskBuffer m = mask_of(d);
        CHECK(m[MASK_TYPE + idx(ActionType::PLAY_DEV)] == 1);
        CHECK(nonzero(m.data() + MASK_TYPE, NUM_ACTION_TYPES) == 1);
        CHECK(nonzero(m.data() + MASK_RESOURCE, NUM_RES) == NUM_RES);
        CHECK(nonzero(m.data() + MASK_DEVCARD, NUM_DEV) == 0);
        Failure f;
        check_mask(d, m, f);
        INFO(f.msg);
        CHECK(f.ok());
    }

    SECTION("chance and terminal decisions mask nothing") {
        GameState s = random_setup(std_board(), 5);
        apply(std_board(), s, Action{ActionType::ROLL, 0});
        const Decision d = next_decision(std_board(), s);
        REQUIRE(d.kind == DecisionKind::CHANCE);
        CHECK(mask_of(d) == MaskBuffer{});
        s.pub.pending_chance = 0;
        s.pub.phase = Phase::TERMINAL;
        const Decision t = next_decision(std_board(), s);
        REQUIRE(t.kind == DecisionKind::TERMINAL);
        CHECK(mask_of(t) == MaskBuffer{});
    }
}

// ---- golden ---------------------------------------------------------------------------------

namespace {

// Golden streams pin the whole encoding: FNV-1a over every mask and every observer's
// observation at every decision of a game, in order. They change when the layout or the
// semantics of any byte change (bump ENCODE_VERSION) or when play or belief tracking changes
// (then test_golden or test_belief moves too). Regenerate with CATAN_ENCODE_GOLDEN_PRINT=1.
struct EncodeGolden {
    uint64_t seed;
    bool standard;  // standard_board() instead of random_board(seed)
    uint64_t hash;
    uint16_t ply;  // final ply; the same games as test_golden.cpp
};

// Generated from the first implementation of encode.cpp once every other test passed.
constexpr array<EncodeGolden, 3> ENCODE_GOLDENS = {{
    {1, false, 0x6f466b1faeef9092ull, 1733},
    {7, true, 0xcaef9e94fa7d2507ull, 1127},
    {42, false, 0x05b20efdea3e9cc0ull, 1379},
}};

}  // namespace

TEST_CASE("golden: observation and mask streams of whole games are pinned", "[encode][golden]") {
    const bool print = getenv("CATAN_ENCODE_GOLDEN_PRINT") != nullptr;
    for (const EncodeGolden& g : ENCODE_GOLDENS) {
        const Board board = g.standard ? standard_board() : random_board(g.seed);
        uint64_t h = 0xCBF29CE484222325ull;
        uint16_t ply = 0;
        play_tracked(board, g.seed, [&](const GameState& s, const BeliefState& b,
                                        const Decision& d) {
            ply = s.pub.ply;
            const MaskBuffer m = mask_of(d);
            h = fnv(h, m.data(), m.size());
            for (uint8_t o = 0; o < NUM_PLAYERS; ++o) {
                const Observation x = obs_of(board, s, b, o);
                h = fnv(h, x.data(), x.size());
            }
        });
        if (print) {
            printf("    {%llu, %s, 0x%016llxull, %u},\n", static_cast<unsigned long long>(g.seed),
                   g.standard ? "true" : "false", static_cast<unsigned long long>(h), ply);
        }
        INFO("seed " << g.seed << (g.standard ? " standard" : " random"));
        CHECK(ply == g.ply);
        CHECK(h == g.hash);
    }
}

TEST_CASE("mask: bits beyond a head's size are ignored", "[encode][mask]") {
    Decision d{};
    d.kind = DecisionKind::PLAYER;
    d.player = 0;
    d.phase = Phase::MAIN;
    d.mask.type = 0xFFFF;
    d.mask.settlement = ~uint64_t{0};
    d.mask.city = ~uint64_t{0};
    d.mask.edge = {~uint64_t{0}, ~uint64_t{0}};
    d.mask.tile = 0xFFFFFFFFu;
    d.mask.devcard = 0xFF;
    d.mask.player = 0xFF;
    d.mask.resource = 0xFF;

    MaskBuffer out;
    encode_mask(d, out);
    // Every head is full and nothing spilled past MASK_SIZE (ASan) or into a neighbor.
    for (int i = 0; i < MASK_SIZE; ++i) {
        INFO("byte " << i);
        CHECK(out[i] == 1);
    }
}
