// Tests for text.h, the human-readable views of a game.
//
// The header is the contract. Most checks are properties that need no knowledge of the layout:
// equal line widths, determinism, every piece visible and gone again when removed, ids found by
// counting the numbers a render gains, and no hidden information in render_state or
// describe_event, checked over many states of random games and every observer. A few checks
// read the marker format of the implementation (counted against the board, so a layout change
// that keeps the markers keeps them passing), and a few pin exact strings: the header's own
// examples and two golden boards.
//
//   CATAN_TEXT_GAMES=n          games per random-game suite (default 12)
//   CATAN_TEXT_GOLDEN_PRINT=1   print fresh golden boards after a deliberate layout change

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <string>
#include <vector>

#include "catan/action.h"
#include "catan/board.h"
#include "catan/decision.h"
#include "catan/event.h"
#include "catan/mask.h"
#include "catan/rng.h"
#include "catan/rules.h"
#include "catan/state.h"
#include "catan/text.h"
#include "catan/topology.h"
#include "catan/types.h"
#include "support/fixtures.h"

using namespace std;
using namespace catan;
using namespace catan::test;

namespace {

constexpr int NUM_PHASES = static_cast<int>(Phase::TERMINAL) + 1;
constexpr int NUM_HEX_TYPES = static_cast<int>(HexType::DESERT) + 1;
constexpr int NUM_PORT_TYPES = static_cast<int>(PortType::GENERIC) + 1;
constexpr int NUM_EVENT_TYPES = static_cast<int>(EventType::GAME_OVER) + 1;
constexpr array<uint8_t, 5> OBSERVERS = {0, 1, 2, 3, NONE};

const RenderOptions PLAIN{};
const RenderOptions VERTEX_IDS{true, false};
const RenderOptions EDGE_IDS{false, true};
const RenderOptions BOTH_IDS{true, true};
const array<RenderOptions, 4> ALL_OPTIONS = {PLAIN, VERTEX_IDS, EDGE_IDS, BOTH_IDS};

uint64_t games() {
    const char* env = getenv("CATAN_TEXT_GAMES");
    return env ? strtoull(env, nullptr, 10) : 12;
}

Board board_for(uint64_t seed) { return seed % 4 == 0 ? standard_board() : random_board(seed); }

string seat(int p) { return "P" + to_string(p); }

// ---- text helpers ---------------------------------------------------------------------------

// Lines of `text`; a final '\n' ends the last line rather than starting an empty one.
vector<string> lines_of(const string& text) {
    vector<string> out;
    size_t at = 0;
    while (at < text.size()) {
        const size_t nl = text.find('\n', at);
        if (nl == string::npos) {
            out.push_back(text.substr(at));
            break;
        }
        out.push_back(text.substr(at, nl - at));
        at = nl + 1;
    }
    return out;
}

// Every maximal run of decimal digits, as numbers.
multiset<int> numbers_in(const string& text) {
    multiset<int> out;
    for (size_t i = 0; i < text.size();) {
        if (!isdigit(static_cast<unsigned char>(text[i]))) {
            ++i;
            continue;
        }
        int n = 0;
        while (i < text.size() && isdigit(static_cast<unsigned char>(text[i]))) {
            n = n * 10 + (text[i] - '0');
            ++i;
        }
        out.insert(n);
    }
    return out;
}

// with - without, as multisets; `extra` is set if `without` holds a number `with` lacks.
multiset<int> gained(const multiset<int>& with, const multiset<int>& without, bool& extra) {
    multiset<int> out;
    set_difference(with.begin(), with.end(), without.begin(), without.end(),
                   inserter(out, out.begin()));
    multiset<int> lost;
    set_difference(without.begin(), without.end(), with.begin(), with.end(),
                   inserter(lost, lost.begin()));
    extra = !lost.empty();
    return out;
}

int count_of(const string& text, const string& needle) {
    int n = 0;
    for (size_t at = text.find(needle); at != string::npos; at = text.find(needle, at + 1)) ++n;
    return n;
}

bool contains(const string& text, const string& needle) {
    return text.find(needle) != string::npos;
}

bool starts_with(const string& text, const string& prefix) {
    return text.compare(0, prefix.size(), prefix) == 0;
}

bool printable_ascii(const string& text, bool allow_newline) {
    for (char c : text) {
        if (c == '\n' && allow_newline) continue;
        if (c < 0x20 || c > 0x7E) return false;
    }
    return true;
}

bool has_upper(const string& text) {
    return any_of(text.begin(), text.end(),
                  [](char c) { return isupper(static_cast<unsigned char>(c)) != 0; });
}

// Empty if `text` is a well-formed render: printable ASCII lines of one common width.
string shape_error(const string& text) {
    if (text.empty()) return "empty render";
    if (!printable_ascii(text, true)) return "not printable ASCII";
    const vector<string> lines = lines_of(text);
    if (lines.size() < 10) return "fewer than 10 lines";
    const size_t width = lines.front().size();
    if (width == 0) return "empty first line";
    for (size_t i = 0; i < lines.size(); ++i) {
        if (lines[i].size() != width) {
            return "line " + to_string(i) + " has width " + to_string(lines[i].size()) +
                   ", line 0 has " + to_string(width);
        }
    }
    return {};
}

// ---- states ---------------------------------------------------------------------------------

// The state before every decision of a random game, plus the final state.
vector<GameState> game_states(const Board& board, uint64_t seed) {
    vector<GameState> out;
    GameState s = initial_state(board);
    play_random(board, s, seed, 1 << 30,
                [&](const GameState& before, const Decision&, const GameState&,
                    const EventList&) { out.push_back(before); });
    out.push_back(s);
    return out;
}

// Every `stride`-th state of game_states, plus every state in a rarer phase and the last.
vector<GameState> sampled_states(const Board& board, uint64_t seed, int stride) {
    const vector<GameState> all = game_states(board, seed);
    vector<GameState> out;
    for (size_t i = 0; i < all.size(); ++i) {
        const Phase ph = all[i].pub.phase;
        const bool rare = ph != Phase::MAIN && ph != Phase::ROLL && ph != Phase::SETUP;
        if (i % stride == 0 || rare || i + 1 == all.size()) out.push_back(all[i]);
    }
    return out;
}

GameState after_decisions(const Board& board, uint64_t seed, int n) {
    GameState s = initial_state(board);
    play_random(board, s, seed, n);
    return s;
}

// Overwrites everything `o` cannot see with arbitrary bytes: other seats' hands and dev cards,
// the deck, and the current player's purchases this turn unless o is that player.
void scramble_hidden(GameState& s, uint8_t o, uint64_t key) {
    uint64_t i = 0;
    auto next = [&] { return static_cast<uint8_t>(random_u64(key, 700, i++)); };
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

// ---- events ---------------------------------------------------------------------------------

Event make_event(EventType t, uint8_t player, uint8_t target = NONE, uint8_t a = 0,
                 uint8_t b = 0) {
    Event e;
    e.type = t;
    e.player = player;
    e.target = target;
    e.a = a;
    e.b = b;
    return e;
}

// A plausible event of type t for player p (other seats and args derived from p and k).
Event sample_event(EventType t, uint8_t p, int k) {
    const auto other = static_cast<uint8_t>((p + 1 + k % 3) % NUM_PLAYERS);
    const auto res = static_cast<uint8_t>(k % NUM_RES);
    switch (t) {
        case EventType::SETTLEMENT_BUILT:
        case EventType::CITY_BUILT: return make_event(t, p, NONE, static_cast<uint8_t>(k % 54));
        case EventType::ROAD_BUILT: return make_event(t, p, NONE, static_cast<uint8_t>(k % 72));
        case EventType::GAIN:
        case EventType::SPEND:
        case EventType::DISCARD:
            return make_event(t, p, NONE, res, static_cast<uint8_t>(1 + k % 4));
        case EventType::DICE_ROLLED:
            return make_event(t, p, NONE, static_cast<uint8_t>(2 + k % 11));
        case EventType::ROBBER_MOVED: return make_event(t, p, NONE, static_cast<uint8_t>(k % 19));
        case EventType::STEAL: return make_event(t, p, other, res);
        case EventType::DEV_BOUGHT:
        case EventType::DEV_PLAYED: return make_event(t, p, NONE, static_cast<uint8_t>(k % 5));
        case EventType::MONOPOLY:
            return make_event(t, p, other, res, static_cast<uint8_t>(k % 6));
        case EventType::LONGEST_ROAD:
        case EventType::LARGEST_ARMY:
        case EventType::TURN_ENDED:
        case EventType::GAME_OVER: return make_event(t, p);
    }
    return make_event(t, p);
}

// The checks every description of `e` must pass; returns an error or empty.
string event_error(const Event& e) {
    for (uint8_t o : OBSERVERS) {
        const string d = describe_event(e, o);
        const string where = "event type " + to_string(int(e.type)) + " observer " +
                             to_string(int(o)) + " \"" + d + "\": ";
        if (d.empty()) return where + "empty";
        if (!printable_ascii(d, false)) return where + "not one printable line";
        if (contains(d, "255")) return where + "prints the NONE sentinel";
        // DICE_ROLLED's player is not part of its contract (event.h: a = sum).
        const bool named = e.type != EventType::DICE_ROLLED;
        if (named && e.player != NONE && !contains(d, seat(e.player))) {
            return where + "lacks the player";
        }
        const bool with_target = e.type == EventType::STEAL || e.type == EventType::MONOPOLY;
        if (with_target && !contains(d, seat(e.target))) return where + "lacks the target";
    }

    const bool steal = e.type == EventType::STEAL;
    const bool bought = e.type == EventType::DEV_BOUGHT;
    if (!steal && !bought) {
        // Public: every observer reads the same line.
        for (uint8_t o : OBSERVERS) {
            if (describe_event(e, o) != describe_event(e)) {
                return "public event type " + to_string(int(e.type)) +
                       " reads differently for observer " + to_string(int(o));
            }
        }
        return {};
    }

    // Hidden payload: the event with each possible value of `a`, as each observer reads it.
    const int values = steal ? NUM_RES : NUM_DEV;
    for (uint8_t o : OBSERVERS) {
        const bool sees = o == NONE || o == e.player || (steal && o == e.target);
        set<string> distinct;
        for (int v = 0; v < values; ++v) {
            Event x = e;
            x.a = static_cast<uint8_t>(v);
            const string d = describe_event(x, o);
            distinct.insert(d);
            if (steal && sees && !contains(d, resource_name(v))) {
                return "steal seen by observer " + to_string(int(o)) + " lacks the resource: " +
                       d;
            }
            if (steal && !sees && contains(d, resource_name(v))) {
                return "steal hidden from observer " + to_string(int(o)) + " names a resource: " +
                       d;
            }
        }
        const size_t want = sees ? static_cast<size_t>(values) : 1;
        if (distinct.size() != want) {
            return string(steal ? "STEAL" : "DEV_BOUGHT") + " by P" + to_string(int(e.player)) +
                   (steal ? " from P" + to_string(int(e.target)) : string()) + ": observer " +
                   to_string(int(o)) + " reads " + to_string(distinct.size()) +
                   " distinct lines over the hidden value, want " + to_string(want);
        }
    }
    return {};
}

// ---- board markers (the implementation's format, see text.cpp) ------------------------------
//
// Corners: "s<p>" settlement, "C<p>" city, "*" empty harbor corner. Roads: "/<p>/" or "\<p>\"
// slanted, a digit between two '#' above and below for a vertical side. Robber: "[R]".
// Harbors: port_type_name(type) beside their corners.

struct Markers {
    array<int, NUM_PLAYERS> settlements{}, cities{}, roads{};
    int robbers = 0;
    int harbor_corners = 0;
    array<int, NUM_PORT_TYPES> harbors{};
};

Markers markers_of(const string& text) {
    Markers m;
    const vector<string> lines = lines_of(text);
    for (size_t r = 0; r < lines.size(); ++r) {
        const string& line = lines[r];
        for (size_t c = 0; c < line.size(); ++c) {
            const char ch = line[c];
            m.harbor_corners += ch == '*';
            if (!(ch >= '0' && ch <= '3')) continue;
            const int p = ch - '0';
            const char before = c > 0 ? line[c - 1] : ' ';
            const char after = c + 1 < line.size() ? line[c + 1] : ' ';
            if (before == 's' && !isdigit(static_cast<unsigned char>(after))) ++m.settlements[p];
            if (before == 'C' && !isdigit(static_cast<unsigned char>(after))) ++m.cities[p];
            if ((before == '/' && after == '/') || (before == '\\' && after == '\\')) ++m.roads[p];
            const bool vertical = r > 0 && r + 1 < lines.size() && lines[r - 1].size() > c &&
                                  lines[r + 1].size() > c && lines[r - 1][c] == '#' &&
                                  lines[r + 1][c] == '#';
            if (vertical) ++m.roads[p];
        }
    }
    m.robbers = count_of(text, "[R]");
    for (int t = 0; t < NUM_PORT_TYPES; ++t) {
        m.harbors[t] = count_of(text, port_type_name(static_cast<PortType>(t)));
    }
    return m;
}

// What markers_of should find for this board and public state.
Markers expected_markers(const Board& board, const PublicState& pub, bool vertex_ids) {
    Markers m;
    for (int v = 0; v < NUM_VERTICES; ++v) {
        const uint8_t b = pub.vertex[v];
        if (vertex_occupied(b)) {
            ++(vertex_is_city(b) ? m.cities : m.settlements)[vertex_owner(b)];
        } else if (!vertex_ids && board.vertex_port[v] != NONE) {
            ++m.harbor_corners;
        }
    }
    for (int p = 0; p < NUM_PLAYERS; ++p) {
        for (int e = 0; e < NUM_EDGES; ++e) m.roads[p] += edge_set_has(pub.roads[p], e);
    }
    m.robbers = 1;
    for (PortType t : board.port_type) ++m.harbors[static_cast<int>(t)];
    return m;
}

string markers_text(const Markers& m) {
    string out;
    auto list = [&](const char* what, const auto& counts) {
        out += what;
        for (int n : counts) out += " " + to_string(n);
        out += ";";
    };
    list(" settlements", m.settlements);
    list(" cities", m.cities);
    list(" roads", m.roads);
    list(" harbors", m.harbors);
    out += " robbers " + to_string(m.robbers) + "; harbor corners " +
           to_string(m.harbor_corners);
    return out;
}

// ---- golden boards --------------------------------------------------------------------------

bool golden_printing() { return getenv("CATAN_TEXT_GOLDEN_PRINT") != nullptr; }

// Prints `text` as C++ source for a string constant: one or more literals per render line,
// escaped, each source line at most 100 columns.
void print_literal(const char* name, const string& text) {
    printf("const char* const %s =\n", name);
    const vector<string> lines = lines_of(text);
    for (size_t l = 0; l < lines.size(); ++l) {
        string escaped;
        for (char c : lines[l]) {
            if (c == '\\' || c == '"') escaped += '\\';
            escaped += c;
        }
        if (l + 1 < lines.size() || text.back() == '\n') escaped += "\\n";
        // Chunks of at most 90 characters, never splitting an escape.
        size_t at = 0;
        while (at < escaped.size()) {
            size_t len = min<size_t>(90, escaped.size() - at);
            size_t slashes = 0;
            for (size_t i = at + len; i > at && escaped[i - 1] == '\\'; --i) ++slashes;
            if (slashes % 2 == 1) --len;
            printf("    \"%s\"\n", escaped.substr(at, len).c_str());
            at += len;
        }
    }
    printf("    ;\n");
}

}  // namespace

// ---- names ----------------------------------------------------------------------------------

TEST_CASE("names: every enum value has a distinct, non-empty, lower-case name", "[text][names]") {
    auto check_table = [](const char* what, int n, auto&& name_of) {
        set<string> seen;
        for (int i = 0; i < n; ++i) {
            INFO(what << " " << i);
            const char* raw = name_of(i);
            REQUIRE(raw != nullptr);
            const string name = raw;
            CHECK(!name.empty());
            CHECK(printable_ascii(name, false));
            CHECK(!has_upper(name));
            CHECK(seen.insert(name).second);
        }
    };
    check_table("resource", NUM_RES, [](int i) { return resource_name(i); });
    check_table("dev card", NUM_DEV, [](int i) { return dev_card_name(i); });
    check_table("phase", NUM_PHASES,
                [](int i) { return phase_name(static_cast<Phase>(i)); });
    check_table("action type", NUM_ACTION_TYPES,
                [](int i) { return action_type_name(static_cast<ActionType>(i)); });
    check_table("hex type", NUM_HEX_TYPES,
                [](int i) { return hex_type_name(static_cast<HexType>(i)); });
    check_table("port type", NUM_PORT_TYPES,
                [](int i) { return port_type_name(static_cast<PortType>(i)); });
}

TEST_CASE("names: the header's examples", "[text][names]") {
    const array<string, NUM_RES> resources = {"brick", "lumber", "wool", "grain", "ore"};
    for (int r = 0; r < NUM_RES; ++r) {
        INFO("resource " << r);
        CHECK(resource_name(r) == resources[r]);
        CHECK(port_type_name(static_cast<PortType>(r)) == "2:1 " + resources[r]);
    }
    CHECK(string(port_type_name(PortType::GENERIC)) == "3:1");
    CHECK(string(dev_card_name(idx(DevCard::KNIGHT))) == "knight");
    CHECK(string(phase_name(Phase::MAIN)) == "main");
    CHECK(string(action_type_name(ActionType::BUILD_SETTLEMENT)) == "build_settlement");
    CHECK(string(hex_type_name(HexType::DESERT)) == "desert");
}

// ---- render_board ---------------------------------------------------------------------------

TEST_CASE("render_board: equal-width ASCII lines, deterministic, for every option",
          "[text][board]") {
    vector<pair<Board, GameState>> cases;
    const Board standard = standard_board();
    cases.emplace_back(standard, initial_state(standard));
    for (uint64_t seed = 1; seed <= 6; ++seed) {
        const Board board = board_for(seed);
        for (const GameState& s : sampled_states(board, seed, 97)) cases.emplace_back(board, s);
    }
    for (const auto& [board, s] : cases) {
        for (const RenderOptions& opts : ALL_OPTIONS) {
            INFO("ply " << s.pub.ply << " ids " << opts.vertex_ids << opts.edge_ids);
            const string text = render_board(board, s.pub, opts);
            const string err = shape_error(text);
            INFO(err << "\n" << text);
            REQUIRE(err.empty());
            CHECK(render_board(board, s.pub, opts) == text);
        }
    }
}

TEST_CASE("render_board: every building and road shows with its owner and goes again",
          "[text][board]") {
    const Board board = standard_board();
    const GameState empty = initial_state(board);
    const string blank = render_board(board, empty.pub);

    // Every (vertex, owner, settlement or city) and every (edge, owner) renders differently
    // from every other, and removing the piece restores the blank board exactly.
    set<string> vertex_renders;
    for (int v = 0; v < NUM_VERTICES; ++v) {
        for (uint8_t p = 0; p < NUM_PLAYERS; ++p) {
            for (bool city : {false, true}) {
                INFO("vertex " << v << " P" << int(p) << (city ? " city" : " settlement"));
                PublicState pub = empty.pub;
                pub.vertex[v] = city ? make_city(p) : make_settlement(p);
                const string text = render_board(board, pub);
                CHECK(shape_error(text).empty());
                CHECK(text != blank);
                CHECK(vertex_renders.insert(text).second);
                pub.vertex[v] = 0;
                CHECK(render_board(board, pub) == blank);
            }
        }
    }
    set<string> edge_renders;
    for (int e = 0; e < NUM_EDGES; ++e) {
        for (uint8_t p = 0; p < NUM_PLAYERS; ++p) {
            INFO("edge " << e << " P" << int(p));
            PublicState pub = empty.pub;
            edge_set_add(pub.roads[p], e);
            const string text = render_board(board, pub);
            CHECK(shape_error(text).empty());
            CHECK(text != blank);
            CHECK(edge_renders.insert(text).second);
            pub.roads[p] = {};
            CHECK(render_board(board, pub) == blank);
        }
    }
    CHECK(vertex_renders.size() == size_t{NUM_VERTICES} * NUM_PLAYERS * 2);
    CHECK(edge_renders.size() == size_t{NUM_EDGES} * NUM_PLAYERS);

    // In a real game, taking away any one piece changes the render.
    const Board rb = random_board(3);
    const GameState s = after_decisions(rb, 3, 600);
    const string full = render_board(rb, s.pub);
    int pieces = 0;
    for (int v = 0; v < NUM_VERTICES; ++v) {
        if (!vertex_occupied(s.pub.vertex[v])) continue;
        ++pieces;
        PublicState pub = s.pub;
        pub.vertex[v] = 0;
        INFO("vertex " << v);
        CHECK(render_board(rb, pub) != full);
    }
    for (uint8_t p = 0; p < NUM_PLAYERS; ++p) {
        for (int e = 0; e < NUM_EDGES; ++e) {
            if (!edge_set_has(s.pub.roads[p], e)) continue;
            ++pieces;
            PublicState pub = s.pub;
            pub.roads[p][e / 64] &= ~(uint64_t{1} << (e % 64));
            INFO("edge " << e);
            CHECK(render_board(rb, pub) != full);
        }
    }
    CHECK(pieces > 20);
}

TEST_CASE("render_board: each hex shows its resource, number and the robber", "[text][board]") {
    const Board board = standard_board();
    const GameState s = initial_state(board);
    const string base = render_board(board, s.pub);

    // Every dice token appears as a number.
    multiset<int> tokens;
    for (uint8_t t : board.token) {
        if (t != 0) tokens.insert(t);
    }
    const multiset<int> numbers = numbers_in(base);
    CHECK(includes(numbers.begin(), numbers.end(), tokens.begin(), tokens.end()));

    // Changing one hex's resource or number, or where the robber stands, changes the render;
    // the robber on each of the 19 hexes gives 19 different renders.
    set<string> robber_renders;
    for (int h = 0; h < NUM_HEXES; ++h) {
        INFO("hex " << h);
        Board other = board;
        other.hex_type[h] =
            static_cast<HexType>((static_cast<int>(board.hex_type[h]) + 1) % NUM_HEX_TYPES);
        CHECK(render_board(other, s.pub) != base);
        if (board.token[h] != 0) {
            other = board;
            other.token[h] = static_cast<uint8_t>(board.token[h] == 12 ? 2 : board.token[h] + 1);
            CHECK(render_board(other, s.pub) != base);
        }
        PublicState pub = s.pub;
        pub.robber_hex = static_cast<uint8_t>(h);
        const string text = render_board(board, pub);
        CHECK(shape_error(text).empty());
        CHECK(robber_renders.insert(text).second);
        CHECK((text == base) == (h == board.desert_hex));
    }
    // Each harbor's type shows: changing any one slot's type changes the render.
    for (int p = 0; p < NUM_PORTS; ++p) {
        INFO("port slot " << p);
        Board other = board;
        other.port_type[p] = static_cast<PortType>(
            (static_cast<int>(board.port_type[p]) + 1) % NUM_PORT_TYPES);
        CHECK(render_board(other, s.pub) != base);
    }
}

TEST_CASE("render_board: building, road, robber and harbor markers match the board",
          "[text][board][markers]") {
    long states = 0;
    auto check = [&](const Board& board, const PublicState& pub) {
        ++states;
        for (const RenderOptions& opts : ALL_OPTIONS) {
            INFO("ply " << pub.ply << " ids " << opts.vertex_ids << opts.edge_ids);
            const string text = render_board(board, pub, opts);
            const string got = markers_text(markers_of(text));
            const string want = markers_text(expected_markers(board, pub, opts.vertex_ids));
            if (got != want) {
                INFO(text);
                CHECK(got == want);
            }
        }
    };
    const Board standard = standard_board();
    const GameState empty = initial_state(standard);
    check(standard, empty.pub);
    // The empty standard board: 18 harbor corners, four 3:1 harbors and one of each 2:1.
    const Markers m = markers_of(render_board(standard, empty.pub));
    CHECK(m.harbor_corners == 18);
    CHECK(m.harbors == array<int, NUM_PORT_TYPES>{1, 1, 1, 1, 1, 4});
    CHECK(m.robbers == 1);
    for (uint64_t seed = 1; seed <= games(); ++seed) {
        const Board board = board_for(seed);
        for (const GameState& s : sampled_states(board, seed, 53)) check(board, s.pub);
    }
    CHECK(states > static_cast<long>(games()) * 20);
}

TEST_CASE("render_board: vertex_ids and edge_ids label exactly the empty spots",
          "[text][board][ids]") {
    // A render with ids holds exactly the ids of the empty corners / road spots on top of the
    // numbers of the plain render (dice numbers, harbor rates, owners).
    auto check_ids = [](const Board& board, const PublicState& pub) {
        multiset<int> empty_vertices, empty_edges;
        for (int v = 0; v < NUM_VERTICES; ++v) {
            if (!vertex_occupied(pub.vertex[v])) empty_vertices.insert(v);
        }
        for (int e = 0; e < NUM_EDGES; ++e) {
            bool taken = false;
            for (const EdgeSet& roads : pub.roads) taken |= edge_set_has(roads, e);
            if (!taken) empty_edges.insert(e);
        }
        multiset<int> both = empty_vertices;
        both.insert(empty_edges.begin(), empty_edges.end());

        const multiset<int> plain = numbers_in(render_board(board, pub, PLAIN));
        bool lost = false;
        INFO("ply " << pub.ply);
        CHECK(gained(numbers_in(render_board(board, pub, VERTEX_IDS)), plain, lost) ==
              empty_vertices);
        CHECK(!lost);
        CHECK(gained(numbers_in(render_board(board, pub, EDGE_IDS)), plain, lost) == empty_edges);
        CHECK(!lost);
        CHECK(gained(numbers_in(render_board(board, pub, BOTH_IDS)), plain, lost) == both);
        CHECK(!lost);
    };

    const Board standard = standard_board();
    const GameState empty = initial_state(standard);
    check_ids(standard, empty.pub);
    // On the empty board, that is every vertex id 0..53 and every edge id 0..71.
    bool lost = false;
    const multiset<int> plain = numbers_in(render_board(standard, empty.pub));
    const multiset<int> v_ids = gained(numbers_in(render_board(standard, empty.pub, VERTEX_IDS)),
                                       plain, lost);
    const multiset<int> e_ids = gained(numbers_in(render_board(standard, empty.pub, EDGE_IDS)),
                                       plain, lost);
    REQUIRE(v_ids.size() == NUM_VERTICES);
    REQUIRE(e_ids.size() == NUM_EDGES);
    CHECK(*v_ids.begin() == 0);
    CHECK(*v_ids.rbegin() == NUM_VERTICES - 1);
    CHECK(set<int>(v_ids.begin(), v_ids.end()).size() == NUM_VERTICES);
    CHECK(*e_ids.rbegin() == NUM_EDGES - 1);
    CHECK(set<int>(e_ids.begin(), e_ids.end()).size() == NUM_EDGES);

    for (uint64_t seed = 1; seed <= 4; ++seed) {
        const Board board = board_for(seed);
        for (const GameState& s : sampled_states(board, seed, 211)) check_ids(board, s.pub);
    }
}

// ---- render_state ---------------------------------------------------------------------------

TEST_CASE("render_state: starts with render_board and is well formed", "[text][state]") {
    for (uint64_t seed = 1; seed <= 3; ++seed) {
        const Board board = board_for(seed);
        for (const GameState& s : sampled_states(board, seed, 151)) {
            for (uint8_t o : OBSERVERS) {
                for (const RenderOptions& opts : {PLAIN, BOTH_IDS}) {
                    INFO("seed " << seed << " ply " << s.pub.ply << " observer " << int(o));
                    const string text = render_state(board, s, o, opts);
                    CHECK(printable_ascii(text, true));
                    CHECK(starts_with(text, render_board(board, s.pub, opts)));
                    CHECK(text.size() > render_board(board, s.pub, opts).size());
                    CHECK(render_state(board, s, o, opts) == text);
                }
            }
        }
    }
}

TEST_CASE("render_state: no hidden information reaches an observer's view",
          "[text][state][leak]") {
    // For every observer at sampled points of random games, overwriting everything it cannot
    // see with junk must leave its view byte for byte unchanged. The god view sees the junk.
    long points = 0, god_changed = 0;
    for (uint64_t seed = 1; seed <= games(); ++seed) {
        const Board board = board_for(seed);
        for (const GameState& s : sampled_states(board, seed, 13)) {
            ++points;
            for (uint8_t o = 0; o < NUM_PLAYERS; ++o) {
                const uint64_t key = seed << 32 | uint64_t{s.pub.ply} << 4 | o;
                GameState junk = s;
                scramble_hidden(junk, o, key);
                INFO("seed " << seed << " ply " << s.pub.ply << " observer " << int(o));
                const string view = render_state(board, s, o);
                const string junk_view = render_state(board, junk, o);
                if (junk_view != view) {
                    INFO("true view:\n" << view << "\nwith junk:\n" << junk_view);
                    REQUIRE(junk_view == view);
                }
                if (key % 5 == 0) CHECK(render_state(board, junk, o, BOTH_IDS) ==
                                        render_state(board, s, o, BOTH_IDS));
                if (o == 0) god_changed += render_state(board, junk) != render_state(board, s);
            }
        }
    }
    INFO("points " << points << " god view changed " << god_changed);
    REQUIRE(points >= static_cast<long>(games()) * 50);
    CHECK(god_changed == points);
}

TEST_CASE("render_state: the observer sees its own cards, everyone sees the public panels",
          "[text][state]") {
    const Board board = random_board(11);
    GameState s = after_decisions(board, 11, 500);
    set_main(s, s.pub.current_player, 8);
    // Make sure there is something to see: every seat holds a mix of cards.
    for (uint8_t p = 0; p < NUM_PLAYERS; ++p) {
        give(s, p, Resource::BRICK, 1);
        give(s, p, Resource::ORE, 1);
        give_dev(s, p, DevCard::KNIGHT);
    }
    array<string, 5> base;
    for (size_t i = 0; i < OBSERVERS.size(); ++i) base[i] = render_state(board, s, OBSERVERS[i]);

    // Hidden edits to seat p's hand or dev cards (same counts) show only to p and the god view.
    for (uint8_t p = 0; p < NUM_PLAYERS; ++p) {
        GameState hand = s;
        --hand.hid.hands[p][idx(Resource::BRICK)];
        ++hand.hid.hands[p][idx(Resource::GRAIN)];
        GameState dev = s;
        --dev.hid.dev_held[p][idx(DevCard::KNIGHT)];
        ++dev.hid.dev_held[p][idx(DevCard::MONOPOLY)];
        for (size_t i = 0; i < OBSERVERS.size(); ++i) {
            const uint8_t o = OBSERVERS[i];
            const bool sees = o == NONE || o == p;
            INFO("seat " << int(p) << " observer " << int(o));
            CHECK((render_state(board, hand, o) != base[i]) == sees);
            CHECK((render_state(board, dev, o) != base[i]) == sees);
        }
    }

    // Public facts of every seat change every observer's view.
    using Edit = void (*)(PublicState&, uint8_t);
    const vector<pair<const char*, Edit>> public_edits = {
        {"public_vp", [](PublicState& pub, uint8_t p) { ++pub.public_vp[p]; }},
        {"hand_size", [](PublicState& pub, uint8_t p) { ++pub.hand_size[p]; }},
        {"dev_count", [](PublicState& pub, uint8_t p) { ++pub.dev_count[p]; }},
        {"knights_played", [](PublicState& pub, uint8_t p) { ++pub.knights_played[p]; }},
        {"road_len", [](PublicState& pub, uint8_t p) { ++pub.road_len[p]; }},
        {"roads left",
         [](PublicState& pub, uint8_t p) { --pub.pieces_left[p][PIECE_ROAD]; }},
        {"settlements left",
         [](PublicState& pub, uint8_t p) { --pub.pieces_left[p][PIECE_SETTLEMENT]; }},
        {"cities left", [](PublicState& pub, uint8_t p) { --pub.pieces_left[p][PIECE_CITY]; }},
        {"longest road",
         [](PublicState& pub, uint8_t p) {
             pub.longest_road_holder = pub.longest_road_holder == p ? NONE : p;
         }},
        {"largest army",
         [](PublicState& pub, uint8_t p) {
             pub.largest_army_holder = pub.largest_army_holder == p ? NONE : p;
         }},
    };
    for (const auto& [what, edit] : public_edits) {
        for (uint8_t p = 0; p < NUM_PLAYERS; ++p) {
            GameState t = s;
            edit(t.pub, p);
            for (size_t i = 0; i < OBSERVERS.size(); ++i) {
                INFO(what << " of seat " << int(p) << " observer " << int(OBSERVERS[i]));
                CHECK(render_state(board, t, OBSERVERS[i]) != base[i]);
            }
        }
    }
    using TurnEdit = void (*)(PublicState&);
    const vector<pair<const char*, TurnEdit>> turn_edits = {
        {"turn", [](PublicState& pub) { ++pub.turn; }},
        {"phase",
         [](PublicState& pub) {
             pub.phase = pub.phase == Phase::MAIN ? Phase::ROLL : Phase::MAIN;
         }},
        {"current player",
         [](PublicState& pub) {
             pub.current_player = static_cast<uint8_t>((pub.current_player + 1) % NUM_PLAYERS);
         }},
        {"last roll", [](PublicState& pub) { pub.dice = 5; }},
        {"bank", [](PublicState& pub) { --pub.bank[idx(Resource::WOOL)]; }},
        {"dev deck", [](PublicState& pub) { --pub.dev_deck_remaining; }},
    };
    for (const auto& [what, edit] : turn_edits) {
        GameState t = s;
        edit(t.pub);
        for (size_t i = 0; i < OBSERVERS.size(); ++i) {
            INFO(what << " observer " << int(OBSERVERS[i]));
            CHECK(render_state(board, t, OBSERVERS[i]) != base[i]);
        }
    }
    // Pending discards, while discarding.
    GameState discarding = s;
    discarding.pub.phase = Phase::DISCARD;
    discarding.pub.pending_discard = {0, 2, 0, 0};
    GameState more = discarding;
    more.pub.pending_discard[3] = 4;
    for (uint8_t o : OBSERVERS) {
        INFO("pending discards observer " << int(o));
        CHECK(render_state(board, more, o) != render_state(board, discarding, o));
    }
}

// ---- describe_action ------------------------------------------------------------------------

TEST_CASE("describe_action: the header's examples", "[text][action]") {
    using AT = ActionType;
    CHECK(describe_action({AT::BUILD_SETTLEMENT, 17}, Phase::MAIN, 0) ==
          "build settlement at v17");
    CHECK(describe_action({AT::STEAL, 0}, Phase::STEAL, 1) == "steal from P2");
    CHECK(describe_action({AT::DISCARD, idx(Resource::GRAIN)}, Phase::DISCARD, 3) ==
          "discard grain");
    CHECK(describe_action({AT::BANK_EXCHANGE, idx(Resource::GRAIN)}, Phase::MAIN, 0) ==
          "trade grain to the bank");
    CHECK(describe_action({AT::BANK_EXCHANGE, idx(Resource::ORE)}, Phase::EXCHANGE_RECV, 0) ==
          "receive ore");
    CHECK(describe_action({AT::PLAY_DEV, idx(Resource::GRAIN)}, Phase::YOP_PICK, 2) ==
          "take grain");
    CHECK(describe_action({AT::END_TURN, 0}, Phase::MAIN, 1) == "end turn");
}

TEST_CASE("describe_action: STEAL names the absolute seat of the relative victim",
          "[text][action]") {
    for (uint8_t decider = 0; decider < NUM_PLAYERS; ++decider) {
        set<string> seen;
        for (uint8_t rel = 0; rel < 3; ++rel) {
            const uint8_t victim = seat_from_relative(decider, rel);
            INFO("decider " << int(decider) << " rel " << int(rel));
            const string d = describe_action({ActionType::STEAL, rel}, Phase::STEAL, decider);
            CHECK(d == "steal from " + seat(victim));
            CHECK(seen.insert(d).second);
        }
    }
}

TEST_CASE("describe_action: the resource head names resources in YOP and MONOPOLY picks",
          "[text][action]") {
    for (Phase ph : {Phase::YOP_PICK, Phase::MONOPOLY_PICK, Phase::DISCARD,
                     Phase::EXCHANGE_RECV}) {
        const ActionType t = ph == Phase::DISCARD         ? ActionType::DISCARD
                             : ph == Phase::EXCHANGE_RECV ? ActionType::BANK_EXCHANGE
                                                          : ActionType::PLAY_DEV;
        set<string> seen;
        for (int r = 0; r < NUM_RES; ++r) {
            INFO("phase " << phase_name(ph) << " resource " << r);
            const string d = describe_action({t, static_cast<uint8_t>(r)}, ph, 0);
            CHECK(contains(d, resource_name(r)));
            CHECK(seen.insert(d).second);
        }
    }
    // In MAIN and ROLL, PLAY_DEV names a card.
    for (Phase ph : {Phase::MAIN, Phase::ROLL}) {
        set<string> seen;
        for (int c = 0; c < NUM_DEV; ++c) {
            const Action play{ActionType::PLAY_DEV, static_cast<uint8_t>(c)};
            const string d = describe_action(play, ph, 0);
            CHECK(seen.insert(d).second);
        }
    }
}

TEST_CASE("describe_action: every legal action of random games reads distinctly",
          "[text][action]") {
    long decisions = 0, actions = 0;
    array<long, NUM_PHASES> phases{};
    for (uint64_t seed = 1; seed <= games(); ++seed) {
        const Board board = board_for(seed);
        GameState s = initial_state(board);
        bool ok = true;
        play_random(board, s, seed, 1 << 30,
                    [&](const GameState& before, const Decision& d, const GameState&,
                        const EventList&) {
                        if (!ok || d.kind != DecisionKind::PLAYER) return;
                        ++decisions;
                        ++phases[static_cast<int>(d.phase)];
                        const int n = count_legal(d.mask, d.phase);
                        set<string> seen;
                        for (int i = 0; i < n; ++i) {
                            const Action a = nth_legal(d.mask, d.phase, i);
                            const string text = describe_action(a, d.phase, d.player);
                            ++actions;
                            const bool good =
                                !text.empty() && printable_ascii(text, false) &&
                                !contains(text, "255") && seen.insert(text).second &&
                                (a.type != ActionType::STEAL ||
                                 text == "steal from " +
                                             seat(seat_from_relative(d.player, a.arg)));
                            if (!good) {
                                ok = false;
                                INFO("seed " << seed << " ply " << before.pub.ply << " phase "
                                             << phase_name(d.phase) << " action "
                                             << int(a.type) << "/" << int(a.arg) << ": \""
                                             << text << "\"");
                                CHECK(good);
                                return;
                            }
                        }
                    });
        REQUIRE(ok);
    }
    INFO("decisions " << decisions << " actions " << actions);
    CHECK(decisions > static_cast<long>(games()) * 300);
    if (games() >= 8) {
        // Every phase with a player decision was described at least once.
        for (int p = 0; p < NUM_PHASES; ++p) {
            if (static_cast<Phase>(p) == Phase::TERMINAL) continue;
            INFO("phase " << p);
            CHECK(phases[p] > 0);
        }
    }
}

// ---- describe_event -------------------------------------------------------------------------

TEST_CASE("describe_event: the header's example", "[text][event]") {
    const Event e = make_event(EventType::SETTLEMENT_BUILT, 0, NONE, 12);
    for (uint8_t o : OBSERVERS) CHECK(describe_event(e, o) == "P0 builds a settlement at v12");
    CHECK(describe_event(e) == "P0 builds a settlement at v12");
}

TEST_CASE("describe_event: every type reads as one line, hiding only steals and purchases",
          "[text][event]") {
    for (int t = 0; t < NUM_EVENT_TYPES; ++t) {
        for (uint8_t p = 0; p < NUM_PLAYERS; ++p) {
            for (int k = 0; k < 6; ++k) {
                const Event e = sample_event(static_cast<EventType>(t), p, k);
                const string err = event_error(e);
                INFO(err);
                CHECK(err.empty());
            }
        }
    }
    // The events that may carry no player.
    for (EventType t : {EventType::LONGEST_ROAD, EventType::GAME_OVER}) {
        const string err = event_error(make_event(t, NONE));
        INFO(err);
        CHECK(err.empty());
        CHECK(describe_event(make_event(t, NONE)) != describe_event(make_event(t, 0)));
    }
}

TEST_CASE("describe_event: STEAL and DEV_BOUGHT show the hidden card to the right seats",
          "[text][event][leak]") {
    for (uint8_t thief = 0; thief < NUM_PLAYERS; ++thief) {
        for (uint8_t victim = 0; victim < NUM_PLAYERS; ++victim) {
            if (victim == thief) continue;
            for (int r = 0; r < NUM_RES; ++r) {
                const Event e = make_event(EventType::STEAL, thief, victim,
                                           static_cast<uint8_t>(r));
                for (uint8_t o : OBSERVERS) {
                    const string d = describe_event(e, o);
                    const bool sees = o == NONE || o == thief || o == victim;
                    INFO("thief " << int(thief) << " victim " << int(victim) << " observer "
                                  << int(o) << ": " << d);
                    CHECK(contains(d, resource_name(r)) == sees);
                }
            }
        }
        for (int c = 0; c < NUM_DEV; ++c) {
            const Event e = make_event(EventType::DEV_BOUGHT, thief, NONE,
                                       static_cast<uint8_t>(c));
            // The buyer and the god view read the card; everyone else one line for all cards.
            const Event other = make_event(EventType::DEV_BOUGHT, thief, NONE,
                                           static_cast<uint8_t>((c + 1) % NUM_DEV));
            for (uint8_t o : OBSERVERS) {
                const bool sees = o == NONE || o == thief;
                INFO("buyer " << int(thief) << " card " << c << " observer " << int(o));
                CHECK((describe_event(e, o) != describe_event(other, o)) == sees);
            }
        }
    }
}

TEST_CASE("describe_event: every event of random games", "[text][event]") {
    long events = 0, steals = 0, bought = 0;
    for (uint64_t seed = 1; seed <= games(); ++seed) {
        const Board board = board_for(seed);
        GameState s = initial_state(board);
        string err;
        play_random(board, s, seed, 1 << 30,
                    [&](const GameState& before, const Decision&, const GameState&,
                        const EventList& list) {
                        for (const Event& e : list) {
                            if (!err.empty()) return;
                            ++events;
                            steals += e.type == EventType::STEAL;
                            bought += e.type == EventType::DEV_BOUGHT;
                            err = event_error(e);
                            if (!err.empty()) {
                                err = "seed " + to_string(seed) + " ply " +
                                      to_string(before.pub.ply) + ": " + err;
                            }
                        }
                    });
        INFO(err);
        REQUIRE(err.empty());
    }
    INFO("events " << events << " steals " << steals << " bought " << bought);
    CHECK(steals > 0);
    CHECK(bought > 0);
}

// ---- golden boards --------------------------------------------------------------------------
//
// Pinned from the implementation once every property above passed, so that a layout change is
// a deliberate one. Regenerate with CATAN_TEXT_GOLDEN_PRINT=1.

namespace {

constexpr int GOLDEN_DECISIONS = 900;  // seed 1, random_board(1), play_random decisions

const char* const GOLDEN_STANDARD_EMPTY =
    "                                        2:1 grain                                \n"
    "                    3:1  *           *           o                               \n"
    "                     .-'   '-.   .-'   '-.   .-'   '-.                           \n"
    "                   *           o           *           o                         \n"
    "                   |  ore 10   |  wool 2   | lumber 9  |                         \n"
    "                   |   hex 0   |   hex 1   |   hex 2   |                         \n"
    "                   |           |           |           |                         \n"
    "                   o           o           o           *  2:1 ore                \n"
    "               .-'   '-.   .-'   '-.   .-'   '-.   .-'   '-.                     \n"
    "             *           o           o           o           *                   \n"
    "             | grain 12  |  brick 6  |  wool 4   | brick 10  |                   \n"
    " 2:1 lumber  |   hex 3   |   hex 4   |   hex 5   |   hex 6   |                   \n"
    "             |           |           |           |           |                   \n"
    "             *           o           o           o           o                   \n"
    "         .-'   '-.   .-'   '-.   .-'   '-.   .-'   '-.   .-'   '-.               \n"
    "       o           o           o           o           o           *             \n"
    "       |  grain 9  | lumber 11 |  desert   | lumber 3  |   ore 8   |             \n"
    "       |   hex 7   |   hex 8   |   hex 9   |  hex 10   |  hex 11   |  3:1        \n"
    "       |           |           |    [R]    |           |           |             \n"
    "       o           o           o           o           o           *             \n"
    "         '-.   .-'   '-.   .-'   '-.   .-'   '-.   .-'   '-.   .-'               \n"
    "             *           o           o           o           o                   \n"
    "             | lumber 8  |   ore 3   |  grain 4  |  wool 5   |                   \n"
    "  2:1 brick  |  hex 12   |  hex 13   |  hex 14   |  hex 15   |                   \n"
    "             |           |           |           |           |                   \n"
    "             *           o           o           o           *                   \n"
    "               '-.   .-'   '-.   .-'   '-.   .-'   '-.   .-'                     \n"
    "                   o           o           o           *  2:1 wool               \n"
    "                   |  brick 5  |  grain 6  |  wool 11  |                         \n"
    "                   |  hex 16   |  hex 17   |  hex 18   |                         \n"
    "                   |           |           |           |                         \n"
    "                   *           o           *           o                         \n"
    "                     '-.   .-'   '-.   .-'   '-.   .-'                           \n"
    "                    3:1  *           *  3:1      o                               \n"
    "                                                                                 ";
const char* const GOLDEN_SEED1_MIDGAME =
    "                                                                                 \n"
    "                    3:1 s1           *  2:1 wool o                               \n"
    "                     /1/   '-.   .-'   '-.   .-'   '-.                           \n"
    "                   *           o           *           o                         \n"
    "                   #   ore 2   #  wool 4   |  grain 8  |                         \n"
    "                   2   hex 0   2   hex 1   |   hex 2   |                         \n"
    "                   #           #           |           |                         \n"
    "                   o          C2           o           *  2:1 grain              \n"
    "               .-'   \\2\\   /2/   \\2\\   .-'   '-.   .-'   '-.                     \n"
    "            s2           o           o          C0           *                   \n"
    "             #  brick 6  |  desert   |   ore 4   #  wool 3   #                   \n"
    "        3:1  2   hex 3   |   hex 4   |   hex 5   0   hex 6   3                   \n"
    "             #           |           |           #           #                   \n"
    "             *           o           o           o          s3                   \n"
    "         .-'   '-.   .-'   '-.   .-'   '-.   .-'   \\3\\   /3/   \\3\\               \n"
    "       o           o           o           o           o           *             \n"
    "       | lumber 11 | lumber 9  |  grain 5  |  wool 3   | grain 12  #             \n"
    "       |   hex 7   |   hex 8   |   hex 9   |  hex 10   |  hex 11   3  3:1        \n"
    "       |           |           |           |           |           #             \n"
    "       o           o           o           o           o           *             \n"
    "         '-.   .-'   '-.   .-'   '-.   .-'   '-.   .-'   '-.   .-'               \n"
    "             *           o           o           o           o                   \n"
    "             |   ore 5   | lumber 9  #  wool 11  |  brick 8  #                   \n"
    "    2:1 ore  |  hex 12   |  hex 13   1  hex 14   |  hex 15   0                   \n"
    "             |           |           #    [R]    |           #                   \n"
    "             *           o           o           o          C0                   \n"
    "               '-.   .-'   '-.   /1/   \\1\\   /1/   \\1\\   .-'                     \n"
    "                   o           o          C1           *  2:1 lumber             \n"
    "                   | lumber 6  # brick 10  | grain 10  |                         \n"
    "                   |  hex 16   1  hex 17   |  hex 18   |                         \n"
    "                   |           #           |           |                         \n"
    "                   *           o           *           o                         \n"
    "                     '-.   /3/   '-.   .-'   '-.   .-'                           \n"
    "              2:1 brick s3           *  3:1      o                               \n"
    "                                                                                 ";

}  // namespace

TEST_CASE("golden: the empty standard board and a mid-game board", "[text][golden]") {
    const Board standard = standard_board();
    const string empty = render_board(standard, initial_state(standard).pub);
    const Board board = random_board(1);
    const GameState s = after_decisions(board, 1, GOLDEN_DECISIONS);
    const string mid = render_board(board, s.pub);
    if (golden_printing()) {
        print_literal("GOLDEN_STANDARD_EMPTY", empty);
        print_literal("GOLDEN_SEED1_MIDGAME", mid);
    }
    CHECK(empty == GOLDEN_STANDARD_EMPTY);
    CHECK(mid == GOLDEN_SEED1_MIDGAME);
}
