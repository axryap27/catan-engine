#include "catan/text.h"

#include <algorithm>
#include <array>
#include <string>
#include <vector>

#include "catan/rules.h"
#include "catan/topology.h"

using namespace std;

namespace catan {
namespace {

// ---- Names ----------------------------------------------------------------------------------

constexpr array<const char*, NUM_RES> RESOURCE_NAMES = {"brick", "lumber", "wool", "grain",
                                                        "ore"};
constexpr array<const char*, NUM_DEV> DEV_CARD_NAMES = {
    "knight", "victory_point", "road_building", "year_of_plenty", "monopoly"};

string num(int n) { return to_string(n); }

// "P0".."P3", "P?" for anything else.
string seat(uint8_t p) { return p < NUM_PLAYERS ? "P" + num(p) : "P?"; }

// "road_building" -> "road building".
string words(const char* snake) {
    string s = snake;
    replace(s.begin(), s.end(), '_', ' ');
    return s;
}

// ---- Board geometry -------------------------------------------------------------------------

// Topology coordinates (see topology.h) map to the character grid as
//   col = MARGIN_L + (x + 5) * CX,   row = MARGIN_T + (y + 8) * CY.
// A hex is 2 * CX = 12 columns by 4 * CY = 8 lines. A terminal cell is about twice as tall as
// wide, so that is a pointy-top hex a little taller than true (1.33 against 1.15), with the
// slanted sides running three columns per line.
constexpr int CX = 6;
constexpr int CY = 2;
static_assert(CX == 6 && CY == 2, "the glyph layout below is drawn for a 6 x 2 cell");

// Room for the longest harbor label ("2:1 lumber") beside the westmost harbor edge (x = -4) and
// the eastmost (x = 5), and for a label that has to clear a neighbouring hex above or below.
constexpr int MARGIN_L = 7;
constexpr int MARGIN_R = 13;
constexpr int MARGIN_T = 1;
constexpr int BOARD_COLS = 10 * CX + 1;
constexpr int BOARD_ROWS = 16 * CY + 1;
constexpr int WIDTH = MARGIN_L + BOARD_COLS + MARGIN_R;
constexpr int HEIGHT = MARGIN_T + BOARD_ROWS + MARGIN_T;  // the same margin below

struct Point {
    int x, y;
};

constexpr array<Point, 6> CORNER_OFFSETS = {{{0, -2}, {1, -1}, {1, 1}, {0, 2}, {-1, 1}, {-1, -1}}};

int col_of(int x) { return MARGIN_L + (x + 5) * CX; }
int row_of(int y) { return MARGIN_T + (y + 8) * CY; }
// Midpoint of two coordinates, exact because CX and CY are even.
int mid_col(int xa, int xb) { return MARGIN_L + (xa + xb + 10) * CX / 2; }
int mid_row(int ya, int yb) { return MARGIN_T + (ya + yb + 16) * CY / 2; }

// Fixed-size character grid; writes outside it are clipped.
class Canvas {
public:
    Canvas(int rows, int cols) : lines_(rows, string(cols, ' ')) {}

    void put(int row, int col, const string& text) {
        if (row < 0 || row >= static_cast<int>(lines_.size())) return;
        for (size_t i = 0; i < text.size(); ++i) {
            const int c = col + static_cast<int>(i);
            if (c >= 0 && c < static_cast<int>(lines_[row].size())) lines_[row][c] = text[i];
        }
    }

    void put_centered(int row, int center, const string& text) {
        put(row, center - static_cast<int>(text.size()) / 2, text);
    }

    // True if columns [col, col + len) of `row` hold only spaces (or lie outside the grid).
    bool blank(int row, int col, int len) const {
        if (row < 0 || row >= static_cast<int>(lines_.size())) return true;
        for (int c = max(col, 0); c < min(col + len, static_cast<int>(lines_[row].size())); ++c) {
            if (lines_[row][c] != ' ') return false;
        }
        return true;
    }

    string str() const {
        string out;
        for (size_t i = 0; i < lines_.size(); ++i) {
            if (i > 0) out += '\n';
            out += lines_[i];
        }
        return out;
    }

private:
    vector<string> lines_;
};

// Position of every vertex in topology units.
array<Point, NUM_VERTICES> vertex_positions() {
    const Topology& topo = topology();
    array<Point, NUM_VERTICES> pos{};
    for (int h = 0; h < NUM_HEXES; ++h) {
        const int cx = 2 * topo.hex_coord[h].q + topo.hex_coord[h].r;
        const int cy = 3 * topo.hex_coord[h].r;
        for (int i = 0; i < 6; ++i) {
            pos[topo.hex_vertices[h][i]] = {cx + CORNER_OFFSETS[i].x, cy + CORNER_OFFSETS[i].y};
        }
    }
    return pos;
}

// Hex interior: three centred lines, "<resource> <token>", "hex <id>", and "[R]" for the robber.
void draw_hexes(Canvas& canvas, const Board& board, const PublicState& pub) {
    const Topology& topo = topology();
    for (int h = 0; h < NUM_HEXES; ++h) {
        const int cx = 2 * topo.hex_coord[h].q + topo.hex_coord[h].r;
        const int cy = 3 * topo.hex_coord[h].r;
        const int col = col_of(cx), row = row_of(cy);
        string label = hex_type_name(board.hex_type[h]);
        if (board.hex_type[h] != HexType::DESERT) label += " " + num(board.token[h]);
        canvas.put_centered(row - 1, col, label);
        canvas.put_centered(row, col, "hex " + num(h));
        if (pub.robber_hex == h) canvas.put_centered(row + 1, col, "[R]");
    }
}

// Roads, and the empty edges they could take. A slanted edge is three characters at its
// midpoint (".-'" rising, "'-." falling, "e12" with ids on, "/1/" or "\1\" for a road of P1); a
// vertical edge is three lines ("|", "|", "|" empty, "#", "1", "#" for a road, "e12" in the
// middle with ids on).
void draw_edges(Canvas& canvas, const PublicState& pub, const array<Point, NUM_VERTICES>& pos,
                bool ids) {
    const Topology& topo = topology();
    for (int e = 0; e < NUM_EDGES; ++e) {
        const Point a = pos[topo.edge_vertices[e][0]], b = pos[topo.edge_vertices[e][1]];
        const int col = mid_col(a.x, b.x), row = mid_row(a.y, b.y);
        char owner = 0;
        for (int p = 0; p < NUM_PLAYERS; ++p) {
            if (edge_set_has(pub.roads[p], e)) owner = static_cast<char>('0' + p);
        }
        const string id = "e" + num(e);

        if (a.x == b.x) {
            const char bar = owner ? '#' : '|';
            canvas.put(row - 1, col, string(1, bar));
            canvas.put(row + 1, col, string(1, bar));
            if (owner) canvas.put(row, col, string(1, owner));
            else if (ids) canvas.put_centered(row, col, id);
            else canvas.put(row, col, "|");
            continue;
        }
        const bool falling = (b.x - a.x) * (b.y - a.y) > 0;
        const char slash = falling ? '\\' : '/';
        string glyph;
        if (owner) glyph = string(1, slash) + owner + slash;
        else if (ids) glyph = id;
        else glyph = falling ? "'-." : ".-'";
        canvas.put_centered(row, col, glyph);
    }
}

// Corners: "s1" settlement, "C1" city, "o" empty, "*" empty harbor corner, "v17" empty with ids.
void draw_vertices(Canvas& canvas, const Board& board, const PublicState& pub,
                   const array<Point, NUM_VERTICES>& pos, bool ids) {
    for (int v = 0; v < NUM_VERTICES; ++v) {
        const int col = col_of(pos[v].x), row = row_of(pos[v].y);
        const uint8_t b = pub.vertex[v];
        if (vertex_occupied(b)) {
            canvas.put(row, col - 1, string(vertex_is_city(b) ? "C" : "s") + num(vertex_owner(b)));
        } else if (ids) {
            canvas.put(row, col - 1, "v" + num(v));
        } else {
            canvas.put(row, col, board.vertex_port[v] != NONE ? "*" : "o");
        }
    }
}

// Harbor labels sit outside the coast, beside the outer corner of their edge (the higher corner
// for a north side, the lower for a south side; the edge midpoint for E and W). East-facing
// sides (NE, E, SE) write rightwards, west-facing sides leftwards. A label that would touch
// something already drawn (the two harbors next to a notch between hexes) moves outwards one
// line at a time.
void draw_ports(Canvas& canvas, const Board& board, const array<Point, NUM_VERTICES>& pos) {
    const Topology& topo = topology();
    for (int p = 0; p < NUM_PORTS; ++p) {
        const PortSlot& slot = topo.port_slots[p];
        int side = 0;
        for (int i = 0; i < 6; ++i) {
            if (topo.hex_edges[slot.hex][i] == slot.edge) side = i;
        }
        const bool east = side <= 2;           // NE, E, SE
        const bool vertical = side == 1 || side == 4;
        const bool north = side == 0 || side == 5;
        const Point a = pos[slot.v0], b = pos[slot.v1];

        int col, row;
        if (vertical) {
            col = mid_col(a.x, b.x);
            row = mid_row(a.y, b.y);
        } else {
            const Point& outer = (north == (a.y < b.y)) ? a : b;
            col = col_of(outer.x);
            row = row_of(outer.y);
        }
        const string label = port_type_name(board.port_type[p]);
        const int len = static_cast<int>(label.size());
        const int start = east ? col + 3 : col - 2 - len;
        // Try the natural line, then one step outwards; the second try is taken as it stands.
        const int step = vertical ? 0 : (north ? -1 : 1);
        if (step != 0 && !canvas.blank(row, start - 1, len + 2)) row += step;
        canvas.put(row, start, label);
    }
}

// ---- Player panels --------------------------------------------------------------------------

// "brick 2, lumber 1": the non-zero entries (all of them with `show_zero`), "" if none.
template <size_t N>
string counts_list(const array<uint8_t, N>& counts, const array<const char*, N>& names,
                   bool show_zero = false) {
    string out;
    for (size_t i = 0; i < N; ++i) {
        if (counts[i] == 0 && !show_zero) continue;
        if (!out.empty()) out += ", ";
        out += string(names[i]) + " " + num(counts[i]);
    }
    return out;
}

// " (...)" around a non-empty text, "" otherwise.
string parenthesized(const string& text) { return text.empty() ? "" : " (" + text + ")"; }

string phase_detail(const PublicState& pub) {
    switch (pub.phase) {
        case Phase::SETUP:
            return " (step " + num(pub.setup_step + 1) + " of " + num(SETUP_STEPS) + ")";
        case Phase::ROAD_BUILDING: return " (" + num(pub.free_roads) + " free roads left)";
        case Phase::YOP_PICK: return " (" + num(pub.yop_remaining) + " picks left)";
        case Phase::EXCHANGE_RECV:
            return string(" (giving ") + resource_name(pub.exchange_give) + ")";
        default: return "";
    }
}

string chance_line(const PublicState& pub) {
    switch (static_cast<ChanceKind>(pub.pending_chance)) {
        case ChanceKind::DICE: return "awaiting: dice roll";
        case ChanceKind::DEV_DRAW: return "awaiting: development card draw";
        case ChanceKind::STEAL: return "awaiting: steal from " + seat(pub.steal_victim);
        default: return "";
    }
}

}  // namespace

// ---- Public API -----------------------------------------------------------------------------

// Draws the board on an 81 x 35 character grid, every line padded to the same width and no
// trailing newline. Hexes are 12 columns by 8 lines, so the map keeps the proportions of the
// real thing in a terminal font. Excerpt, the top of the beginner layout with P0 on v4 and e1,
// P1 on v8 and e10..e12, P2 on v12 and e6 (the first line holds the harbor that cannot sit
// beside its edge without touching the next hex):
//
//                                             2:1 grain
//                         3:1  *           *           o
//                          .-'   \0\   .-'   '-.   .-'   '-.
//                        *          s0           *           o
//                        #  ore 10   |  wool 2   | lumber 9  |
//                        2   hex 0   |   hex 1   |   hex 2   |
//                        #           |           |           |
//                        o          C1           o           *  2:1 ore
//                    /1/   \1\   /1/   '-.   .-'   '-.   .-'   '-.
//                  *          s2           o           o           *
//                  | grain 12  |  brick 6  |  wool 4   | brick 10  |
//      2:1 lumber  |   hex 3   |   hex 4   |   hex 5   |   hex 6   |
//
// Legend
//   corners  o empty, * empty corner of a harbor, s1 settlement and C1 city of P1; with
//            opts.vertex_ids an empty corner reads v17 instead
//   edges    '-. and .-' and | empty; /1/ and \1\ (slanted) and a # 1 # column (vertical) are
//            roads of P1; with opts.edge_ids an empty edge reads e12 instead
//   hexes    resource and dice number, "hex N", and [R] where the robber stands
//   harbors  "3:1" or "2:1 ore" beside the outer corner of the coastal edge (the edge midpoint
//            for the vertical east and west sides), on the sea side
// Both id options can be on together without collisions: vertex ids sit on corner lines, edge
// ids on edge midpoint lines, and the hex label lines leave room for the vertical edges' ids.
string render_board(const Board& board, const PublicState& pub, RenderOptions opts) {
    const array<Point, NUM_VERTICES> pos = vertex_positions();
    Canvas canvas(HEIGHT, WIDTH);
    draw_hexes(canvas, board, pub);
    draw_edges(canvas, pub, pos, opts.edge_ids);
    draw_vertices(canvas, board, pub, pos, opts.vertex_ids);
    draw_ports(canvas, board, pos);
    return canvas.str();
}

string render_state(const Board& board, const GameState& state, uint8_t observer,
                    RenderOptions opts) {
    const PublicState& pub = state.pub;
    const HiddenState& hid = state.hid;
    const bool god = observer == NONE;

    string out = render_board(board, pub, opts);
    out += "\n\n";

    out += "turn " + num(pub.turn) + "  phase " + phase_name(pub.phase) + phase_detail(pub);
    out += "  player " + seat(pub.current_player);
    if (pub.dice != 0) out += "  last roll " + num(pub.dice);
    out += '\n';
    const string chance = chance_line(pub);
    if (!chance.empty()) out += chance + '\n';

    string owed;
    for (int p = 0; p < NUM_PLAYERS; ++p) {
        if (pub.pending_discard[p] == 0) continue;
        owed += (owed.empty() ? "" : ", ") + seat(static_cast<uint8_t>(p)) + " " +
                num(pub.pending_discard[p]);
    }
    if (!owed.empty()) out += "discards owed: " + owed + '\n';

    out += "bank: " + counts_list(pub.bank, RESOURCE_NAMES, true) + "   dev deck: " +
           num(pub.dev_deck_remaining);
    if (god) out += parenthesized(counts_list(hid.dev_deck, DEV_CARD_NAMES));
    out += '\n';

    for (int p = 0; p < NUM_PLAYERS; ++p) {
        // Hands and dev cards are shown to their owner and in the god view only; everyone
        // else gets the public counts.
        const bool visible = god || observer == p;
        const bool current = pub.current_player == p;

        out += '\n';
        out += string(current ? "> " : "  ") + seat(static_cast<uint8_t>(p)) + "  VP " +
               num(pub.public_vp[p]);
        if (visible) {
            const int vp_cards = hid.dev_held[p][idx(DevCard::VICTORY_POINT)];
            out += " (true " + num(pub.public_vp[p] + vp_cards) + ")";
        }
        out += "  knights " + num(pub.knights_played[p]) + "  road " + num(pub.road_len[p]);
        if (pub.longest_road_holder == p) out += "  [Longest Road]";
        if (pub.largest_army_holder == p) out += "  [Largest Army]";
        out += '\n';

        out += "     hand " + num(pub.hand_size[p]);
        if (visible) {
            const string hand = counts_list(hid.hands[p], RESOURCE_NAMES);
            if (!hand.empty()) out += ": " + hand;
        }
        out += '\n';

        out += "     dev " + num(pub.dev_count[p]);
        if (visible) {
            const string held = counts_list(hid.dev_held[p], DEV_CARD_NAMES);
            if (!held.empty()) out += ": " + held;
            // dev_bought_this_turn is the current player's fresh, still locked cards: shown to
            // that player and the god view (the same rule as encode's private_view).
            if (current) {
                const string fresh = counts_list(hid.dev_bought_this_turn, DEV_CARD_NAMES);
                if (!fresh.empty()) out += " (bought this turn: " + fresh + ")";
            }
        }
        out += '\n';

        out += "     left: " + num(pub.pieces_left[p][PIECE_ROAD]) + " roads, " +
               num(pub.pieces_left[p][PIECE_SETTLEMENT]) + " settlements, " +
               num(pub.pieces_left[p][PIECE_CITY]) + " cities";
    }
    return out;
}

string describe_action(Action action, Phase phase, uint8_t decider) {
    const string arg = num(action.arg);
    switch (action.type) {
        case ActionType::ROLL: return "roll";
        case ActionType::BUILD_ROAD:
            if (phase == Phase::ROAD_BUILDING) return "place free road at e" + arg;
            if (phase == Phase::SETUP) return "place road at e" + arg;
            return "build road at e" + arg;
        case ActionType::BUILD_SETTLEMENT:
            return string(phase == Phase::SETUP ? "place" : "build") + " settlement at v" + arg;
        case ActionType::BUILD_CITY: return "build city at v" + arg;
        case ActionType::BUY_DEV: return "buy dev card";
        case ActionType::PLAY_DEV:
            // YOP_PICK and MONOPOLY_PICK reuse PLAY_DEV with a resource instead of a card.
            if (param_head(phase, action.type) == Head::RESOURCE) {
                const string res = resource_name(action.arg);
                return phase == Phase::YOP_PICK ? "take " + res : "monopolize " + res;
            }
            return "play " + words(dev_card_name(action.arg));
        case ActionType::BANK_EXCHANGE:
            if (phase == Phase::EXCHANGE_RECV) {
                return string("receive ") + resource_name(action.arg);
            }
            return string("trade ") + resource_name(action.arg) + " to the bank";
        case ActionType::MOVE_ROBBER: return "move robber to hex " + arg;
        case ActionType::STEAL:
            return "steal from " + seat(seat_from_relative(decider, action.arg));
        case ActionType::DISCARD: return string("discard ") + resource_name(action.arg);
        case ActionType::END_TURN: return "end turn";
        case ActionType::PROPOSE_TRADE: return "propose trade";
        case ActionType::RESPOND_TRADE: return "respond to trade";
    }
    return "unknown action";
}

string describe_event(const Event& e, uint8_t observer) {
    const string p = seat(e.player);
    const string target = seat(e.target);
    const string res = resource_name(e.a);
    switch (e.type) {
        case EventType::SETTLEMENT_BUILT: return p + " builds a settlement at v" + num(e.a);
        case EventType::CITY_BUILT: return p + " builds a city at v" + num(e.a);
        case EventType::ROAD_BUILT: return p + " builds a road at e" + num(e.a);
        case EventType::GAIN: return p + " +" + num(e.b) + " " + res;
        case EventType::SPEND: return p + " -" + num(e.b) + " " + res;
        case EventType::DISCARD:
            return p + " discards " + (e.b > 1 ? num(e.b) + " " : string()) + res;
        case EventType::DICE_ROLLED: return p + " rolls " + num(e.a);
        case EventType::ROBBER_MOVED: return p + " moves the robber to hex " + num(e.a);
        case EventType::STEAL: {
            const bool sees = observer == NONE || observer == e.player || observer == e.target;
            return p + " steals " + (sees ? res : string("a card")) + " from " + target;
        }
        case EventType::DEV_BOUGHT: {
            const bool sees = observer == NONE || observer == e.player;
            return p + " buys a " + (sees ? words(dev_card_name(e.a)) : string("dev")) + " card";
        }
        case EventType::DEV_PLAYED: return p + " plays " + words(dev_card_name(e.a));
        case EventType::MONOPOLY:
            if (e.b == 0) return p + " takes no " + res + " from " + target;
            return p + " takes " + num(e.b) + " " + res + " from " + target;
        case EventType::LONGEST_ROAD:
            return e.player == NONE ? "Longest Road is now unclaimed" : p + " takes Longest Road";
        case EventType::LARGEST_ARMY: return p + " takes Largest Army";
        case EventType::TURN_ENDED: return p + " ends the turn";
        case EventType::GAME_OVER:
            return e.player == NONE ? "Game over: turn cap reached" : "Game over: " + p + " wins";
    }
    return "unknown event";
}

const char* resource_name(int resource) {
    return resource >= 0 && resource < NUM_RES ? RESOURCE_NAMES[resource] : "unknown";
}

const char* dev_card_name(int card) {
    return card >= 0 && card < NUM_DEV ? DEV_CARD_NAMES[card] : "unknown";
}

const char* phase_name(Phase phase) {
    switch (phase) {
        case Phase::SETUP: return "setup";
        case Phase::ROLL: return "roll";
        case Phase::DISCARD: return "discard";
        case Phase::MOVE_ROBBER: return "move_robber";
        case Phase::STEAL: return "steal";
        case Phase::MAIN: return "main";
        case Phase::ROAD_BUILDING: return "road_building";
        case Phase::YOP_PICK: return "yop_pick";
        case Phase::MONOPOLY_PICK: return "monopoly_pick";
        case Phase::EXCHANGE_RECV: return "exchange_recv";
        case Phase::TERMINAL: return "terminal";
    }
    return "unknown";
}

const char* action_type_name(ActionType type) {
    switch (type) {
        case ActionType::ROLL: return "roll";
        case ActionType::BUILD_ROAD: return "build_road";
        case ActionType::BUILD_SETTLEMENT: return "build_settlement";
        case ActionType::BUILD_CITY: return "build_city";
        case ActionType::BUY_DEV: return "buy_dev";
        case ActionType::PLAY_DEV: return "play_dev";
        case ActionType::BANK_EXCHANGE: return "bank_exchange";
        case ActionType::MOVE_ROBBER: return "move_robber";
        case ActionType::STEAL: return "steal";
        case ActionType::DISCARD: return "discard";
        case ActionType::END_TURN: return "end_turn";
        case ActionType::PROPOSE_TRADE: return "propose_trade";
        case ActionType::RESPOND_TRADE: return "respond_trade";
    }
    return "unknown";
}

const char* hex_type_name(HexType type) {
    return type == HexType::DESERT ? "desert" : resource_name(static_cast<int>(type));
}

const char* port_type_name(PortType type) {
    switch (type) {
        case PortType::BRICK: return "2:1 brick";
        case PortType::LUMBER: return "2:1 lumber";
        case PortType::WOOL: return "2:1 wool";
        case PortType::GRAIN: return "2:1 grain";
        case PortType::ORE: return "2:1 ore";
        case PortType::GENERIC: return "3:1";
    }
    return "unknown";
}

}  // namespace catan
