// The `catan_engine` Python extension module (pybind11). The API is fixed by catan_engine.pyi.
//
// Game wraps one game: board, full state, every observer's belief and the seed that resolves
// chance. VecGames is a vector of the same struct with batched stepping and encoding. All
// rules, masks, belief tracking, encoding, text and the greedy baseline are the C++ libraries;
// this file converts types and validates arguments (ValueError for anything bad).
#include <pybind11/native_enum.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "catan/belief.h"
#include "catan/board.h"
#include "catan/chance.h"
#include "catan/encode.h"
#include "catan/policy.h"
#include "catan/rng.h"
#include "catan/rules.h"
#include "catan/text.h"
#include "catan/topology.h"

namespace py = pybind11;
using namespace std;
using namespace catan;

namespace {

using Events = vector<Event>;

// ---- argument checks -------------------------------------------------------------------------

uint8_t seat_arg(int64_t p, const char* what) {
    if (p < 0 || p >= NUM_PLAYERS) {
        throw py::value_error(string(what) + " must be a seat 0.." + to_string(NUM_PLAYERS - 1) +
                              ", got " + to_string(p));
    }
    return static_cast<uint8_t>(p);
}

// An observer argument: None is NONE (the god view / the default observer), else a seat.
uint8_t observer_arg(optional<int64_t> observer) {
    return observer ? seat_arg(*observer, "observer") : NONE;
}

optional<int> seat_or_none(uint8_t p) {
    if (p == NONE) return nullopt;
    return p;
}

// (type, arg) as an Action if both fit the engine's types; legality is a separate question.
optional<Action> to_action(int64_t type, int64_t arg) {
    if (type < 0 || type >= NUM_ACTION_TYPES || arg < 0 || arg > 255) return nullopt;
    return Action{static_cast<ActionType>(type), static_cast<uint8_t>(arg)};
}

string action_label(int64_t type, int64_t arg) {
    string name = "?";
    if (type >= 0 && type < NUM_ACTION_TYPES) {
        name = action_type_name(static_cast<ActionType>(type));
    }
    return "(type=" + to_string(type) + " " + name + ", arg=" + to_string(arg) + ")";
}

// Number of valid args of a parameter head (NONE: just 0).
int head_size(Head h) {
    switch (h) {
        case Head::NONE: return 1;
        case Head::VERTEX: return NUM_VERTICES;
        case Head::EDGE: return NUM_EDGES;
        case Head::TILE: return NUM_HEXES;
        case Head::DEVCARD: return NUM_DEV;
        case Head::PLAYER: return 3;
        case Head::RESOURCE: return NUM_RES;
    }
    return 0;
}

int final_vp(const GameState& s, int p) {
    return s.pub.public_vp[p] + s.hid.dev_held[p][idx(DevCard::VICTORY_POINT)];
}

// The greedy baseline keeps no state between calls; one instance per thread keeps it that way
// even if its implementation ever does.
Policy& greedy_policy() {
    thread_local const unique_ptr<Policy> policy = make_greedy_policy();
    return *policy;
}

// ---- Game ------------------------------------------------------------------------------------

struct Game {
    uint64_t seed;
    bool standard;
    bool auto_chance;
    Board board;
    GameState state;
    BeliefState belief;
    Decision decision;      // next_decision(board, state), refreshed after every change
    uint8_t winner = NONE;  // from the GAME_OVER event: NONE while playing or if the cap ended it

    Game(uint64_t seed_, bool standard_, bool auto_chance_)
        : seed(seed_),
          standard(standard_),
          auto_chance(auto_chance_),
          board(standard_ ? standard_board() : random_board(seed_)),
          state(initial_state(board)),
          belief(initial_belief()),
          decision(next_decision(board, state)) {
        if (auto_chance) resolve_chance(nullptr);
    }

    // A new game on `new_seed` with the same options.
    void restart(uint64_t new_seed) { *this = Game(new_seed, standard, auto_chance); }

    bool player_to_act() const { return decision.kind == DecisionKind::PLAYER; }

    bool legal(int64_t type, int64_t arg) const {
        const optional<Action> a = to_action(type, arg);
        return a && player_to_act() && is_legal(decision.mask, decision.phase, *a);
    }

    // Feeds the events to every observer's belief and, if wanted, collects them.
    void absorb(const EventList& list, Events* out) {
        observe(belief, state.pub, list);
        for (const Event& e : list) {
            if (e.type == EventType::GAME_OVER) winner = e.player;
            if (out) out->push_back(e);
        }
    }

    void apply_outcome(uint8_t outcome, Events* out) {
        absorb(apply_chance(board, state, outcome), out);
        decision = next_decision(board, state);
    }

    // Resolves chance from the seed (as play_random and match.cpp do) until a player must act
    // or the game is over.
    void resolve_chance(Events* out) {
        while (decision.kind == DecisionKind::CHANCE) {
            apply_outcome(sample_chance(decision.chance, state, seed), out);
        }
    }

    // Precondition: legal(). Applies the action, then resolves chance if auto_chance.
    void act(Action a, Events* out) {
        absorb(apply(board, state, a), out);
        decision = next_decision(board, state);
        if (auto_chance) resolve_chance(out);
    }

    // The greedy baseline's choice, with the rng value match.cpp hands its policies.
    Action greedy() const {
        const uint64_t rng = random_u64(seed, 100 + decision.player, state.pub.ply);
        return greedy_policy().choose(board, state, decision, rng);
    }

    void encode_obs(uint8_t observer, uint8_t* out) const {
        Observation buf;
        encode(board, state, belief, observer, buf);
        memcpy(out, buf.data(), buf.size());
    }

    void encode_legal_mask(uint8_t* out) const {
        MaskBuffer buf;
        encode_mask(decision, buf);
        memcpy(out, buf.data(), buf.size());
    }
};

const char* kind_text(DecisionKind k) {
    return k == DecisionKind::CHANCE ? "a chance decision is pending (use apply_chance)"
                                     : "the game is over";
}

template <class Container>
vector<int> ints(const Container& values) {
    vector<int> out;
    out.reserve(values.size());
    for (const auto x : values) out.push_back(static_cast<int>(x));
    return out;
}

// ---- Game: the Python-facing operations --------------------------------------------------------

optional<int> game_decider(const Game& g) {
    return g.player_to_act() ? seat_or_none(g.decision.player) : nullopt;
}

vector<pair<int, int>> game_legal_actions(const Game& g) {
    vector<pair<int, int>> out;
    if (!g.player_to_act()) return out;
    const int n = count_legal(g.decision.mask, g.decision.phase);
    out.reserve(n);
    for (int i = 0; i < n; ++i) {
        const Action a = nth_legal(g.decision.mask, g.decision.phase, i);
        out.emplace_back(idx(a.type), a.arg);
    }
    return out;
}

Events game_step(Game& g, int64_t type, int64_t arg) {
    if (!g.player_to_act()) {
        throw py::value_error(string("step: no player to act, ") + kind_text(g.decision.kind));
    }
    if (!g.legal(type, arg)) {
        throw py::value_error("illegal action " + action_label(type, arg) + " in phase " +
                              phase_name(g.decision.phase));
    }
    Events events;
    g.act(*to_action(type, arg), &events);
    return events;
}

vector<int> game_chance_weights(const Game& g) {
    vector<int> out;
    if (g.decision.kind != DecisionKind::CHANCE) return out;
    const ChanceDist& c = g.decision.chance;
    out.assign(c.weight.begin(), c.weight.begin() + c.n_outcomes);
    return out;
}

Events game_apply_chance(Game& g, int64_t outcome) {
    if (g.decision.kind != DecisionKind::CHANCE) {
        throw py::value_error("apply_chance: no chance decision is pending");
    }
    const ChanceDist& c = g.decision.chance;
    if (outcome < 0 || outcome >= c.n_outcomes || c.weight[outcome] == 0) {
        throw py::value_error("apply_chance: outcome " + to_string(outcome) +
                              " is impossible (this decision has " + to_string(c.n_outcomes) +
                              " outcomes)");
    }
    Events events;
    g.apply_outcome(static_cast<uint8_t>(outcome), &events);
    return events;
}

pair<int, int> game_greedy_action(const Game& g) {
    if (!g.player_to_act()) {
        throw py::value_error(string("greedy_action: no player to act, ") +
                              kind_text(g.decision.kind));
    }
    const Action a = g.greedy();
    return {idx(a.type), a.arg};
}

py::array_t<uint8_t> game_observation(const Game& g, optional<int64_t> observer) {
    uint8_t who = observer_arg(observer);
    if (who == NONE) {
        if (!g.player_to_act()) {
            throw py::value_error(string("observation: no player to act, ") +
                                  kind_text(g.decision.kind) + "; pass an observer");
        }
        who = g.decision.player;
    }
    py::array_t<uint8_t> out(OBS_SIZE);
    g.encode_obs(who, out.mutable_data());
    return out;
}

py::array_t<uint8_t> game_mask(const Game& g) {
    py::array_t<uint8_t> out(MASK_SIZE);
    g.encode_legal_mask(out.mutable_data());
    return out;
}

string game_describe(const Game& g, int64_t type, int64_t arg) {
    if (!g.player_to_act()) {
        throw py::value_error(string("describe: no player to act, ") + kind_text(g.decision.kind));
    }
    const optional<Action> a = to_action(type, arg);
    if (!a || arg >= head_size(param_head(g.decision.phase, a->type))) {
        throw py::value_error("describe: no such action " + action_label(type, arg));
    }
    return describe_action(*a, g.decision.phase, g.decision.player);
}

string game_render(const Game& g, optional<int64_t> observer, bool vertex_ids, bool edge_ids) {
    return render_state(g.board, g.state, observer_arg(observer), {vertex_ids, edge_ids});
}

py::dict game_board(const Game& g) {
    py::dict d;
    d["hex_type"] = ints(g.board.hex_type);
    d["token"] = ints(g.board.token);
    d["port_type"] = ints(g.board.port_type);
    d["robber"] = int{g.state.pub.robber_hex};
    return d;
}

vector<int> game_vp(const Game& g) {
    vector<int> out;
    for (int p = 0; p < NUM_PLAYERS; ++p) out.push_back(final_vp(g.state, p));
    return out;
}

// ---- Events ----------------------------------------------------------------------------------

string event_describe(const Event& e, optional<int64_t> observer) {
    return describe_event(e, observer_arg(observer));
}

string event_repr(const Event& e) {
    const auto seat = [](uint8_t p) { return p == NONE ? string("None") : to_string(p); };
    const string type = py::str(py::cast(e.type).attr("name"));
    return "Event(type=EventType." + type + ", player=" + seat(e.player) +
           ", target=" + seat(e.target) + ", a=" + to_string(e.a) + ", b=" + to_string(e.b) + ")";
}

// ---- Static data -----------------------------------------------------------------------------

// Fixed board geometry (topology.h) as nested lists; NONE padding is dropped.
py::dict topology_dict() {
    const Topology& t = topology();
    const auto unpadded = [](const auto& row) {
        vector<int> out;
        for (const uint8_t x : row) {
            if (x != NONE) out.push_back(x);
        }
        return out;
    };
    const auto rows = [&](const auto& table) {
        vector<vector<int>> out;
        for (const auto& row : table) out.push_back(unpadded(row));
        return out;
    };

    vector<pair<int, int>> hex_coord;
    for (const HexCoord& c : t.hex_coord) hex_coord.emplace_back(c.q, c.r);

    vector<vector<optional<int>>> hex_neighbors;
    for (const auto& row : t.hex_neighbors) {
        vector<optional<int>> r;
        for (const uint8_t x : row) r.push_back(seat_or_none(x));
        hex_neighbors.push_back(r);
    }

    vector<vector<int>> hex_vertices, hex_edges;
    for (int h = 0; h < NUM_HEXES; ++h) {
        hex_vertices.push_back(ints(t.hex_vertices[h]));
        hex_edges.push_back(ints(t.hex_edges[h]));
    }
    vector<pair<int, int>> edge_vertices;
    for (const auto& e : t.edge_vertices) edge_vertices.emplace_back(e[0], e[1]);
    vector<tuple<int, int, int, int>> port_slots;
    for (const PortSlot& p : t.port_slots) port_slots.emplace_back(p.edge, p.v0, p.v1, p.hex);

    py::dict d;
    d["hex_coord"] = hex_coord;
    d["hex_vertices"] = hex_vertices;
    d["hex_edges"] = hex_edges;
    d["hex_neighbors"] = hex_neighbors;
    d["vertex_hexes"] = rows(t.vertex_hexes);
    d["vertex_vertices"] = rows(t.vertex_vertices);
    d["vertex_edges"] = rows(t.vertex_edges);
    d["edge_vertices"] = edge_vertices;
    d["port_slots"] = port_slots;
    return d;
}

py::dict obs_sections() {
    py::dict d;
    for (const ObsSection& s : OBS_SECTIONS) d[s.name] = py::make_tuple(s.offset, s.rows, s.width);
    return d;
}

py::dict mask_sections() {
    py::dict d;
    const auto add = [&](const char* name, int offset, int size) {
        d[name] = py::make_tuple(offset, size);
    };
    add("type", MASK_TYPE, NUM_ACTION_TYPES);
    add("settlement", MASK_SETTLEMENT, NUM_VERTICES);
    add("city", MASK_CITY, NUM_VERTICES);
    add("edge", MASK_EDGE, NUM_EDGES);
    add("tile", MASK_TILE, NUM_HEXES);
    add("devcard", MASK_DEVCARD, NUM_DEV);
    add("player", MASK_PLAYER, 3);
    add("resource", MASK_RESOURCE, NUM_RES);
    return d;
}

uint32_t random_below_arg(uint64_t seed, uint64_t stream, uint64_t index, int64_t bound) {
    if (bound < 1 || bound > UINT32_MAX) {
        throw py::value_error("random_below: bound must be in 1..2^32-1, got " + to_string(bound));
    }
    return random_below(seed, stream, index, static_cast<uint32_t>(bound));
}

// ---- VecGames --------------------------------------------------------------------------------

using IntArray = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;

// A length-n integer vector from any integer ndarray or sequence of ints; ValueError otherwise
// (floats and bools are rejected rather than truncated).
IntArray int_vector(py::handle h, const char* name, py::ssize_t n) {
    const py::array a = py::array::ensure(h);
    const string want = string(name) + " must be an integer array of shape (" + to_string(n) + ",)";
    if (!a || a.ndim() != 1 || a.shape(0) != n) throw py::value_error(want);
    const char kind = a.dtype().kind();
    if (kind != 'i' && kind != 'u') throw py::value_error(want);
    return IntArray::ensure(a);
}

// n games that always wait on a PLAYER decision. A game that ends is reported and restarted at
// once on the next unused seed.
class VecGames {
public:
    VecGames(py::ssize_t n, uint64_t first_seed, bool standard) : next_seed_(first_seed + n) {
        if (n < 1) throw py::value_error("VecGames: n must be at least 1, got " + to_string(n));
        games_.reserve(n);
        for (py::ssize_t i = 0; i < n; ++i) games_.emplace_back(first_seed + i, standard, true);
    }

    py::ssize_t n() const { return static_cast<py::ssize_t>(games_.size()); }

    py::array_t<uint64_t> seeds() const {
        py::array_t<uint64_t> out(n());
        lock_guard<mutex> lock(mu_);
        for (py::ssize_t i = 0; i < n(); ++i) out.mutable_data()[i] = games_[i].seed;
        return out;
    }

    py::tuple observe() const {
        py::array_t<uint8_t> obs(vector<py::ssize_t>{n(), OBS_SIZE});
        py::array_t<uint8_t> mask(vector<py::ssize_t>{n(), MASK_SIZE});
        py::array_t<int8_t> decider(n());
        uint8_t* const op = obs.mutable_data();
        uint8_t* const mp = mask.mutable_data();
        int8_t* const dp = decider.mutable_data();
        {
            py::gil_scoped_release nogil;
            lock_guard<mutex> lock(mu_);
            for (size_t i = 0; i < games_.size(); ++i) {
                const Game& g = games_[i];
                if (g.player_to_act()) {
                    dp[i] = static_cast<int8_t>(g.decision.player);
                    g.encode_obs(g.decision.player, op + i * OBS_SIZE);
                } else {  // unreachable: slots always wait on a player
                    dp[i] = -1;
                    memset(op + i * OBS_SIZE, 0, OBS_SIZE);
                }
                g.encode_legal_mask(mp + i * MASK_SIZE);
            }
        }
        return py::make_tuple(obs, mask, decider);
    }

    py::tuple step(py::handle types, py::handle args) {
        const IntArray t = int_vector(types, "types", n());
        const IntArray a = int_vector(args, "args", n());
        py::array_t<bool> done(n());
        py::array_t<int8_t> winner(n());
        py::array_t<uint8_t> vp(vector<py::ssize_t>{n(), NUM_PLAYERS});
        py::array_t<uint16_t> turns(n());
        const Out out{done.mutable_data(), winner.mutable_data(), vp.mutable_data(),
                      turns.mutable_data()};

        int64_t bad = -1;
        {
            py::gil_scoped_release nogil;
            lock_guard<mutex> lock(mu_);
            bad = step_all(t.data(), a.data(), out);
        }
        if (bad >= 0) {
            throw py::value_error("step: illegal action " +
                                  action_label(t.data()[bad], a.data()[bad]) + " in game " +
                                  to_string(bad));
        }
        return py::make_tuple(done, winner, vp, turns);
    }

    py::tuple greedy_actions() const {
        py::array_t<uint8_t> types(n());
        py::array_t<uint8_t> args(n());
        uint8_t* const tp = types.mutable_data();
        uint8_t* const ap = args.mutable_data();
        {
            py::gil_scoped_release nogil;
            lock_guard<mutex> lock(mu_);
            for (size_t i = 0; i < games_.size(); ++i) {
                const Action act = games_[i].greedy();
                tp[i] = static_cast<uint8_t>(act.type);
                ap[i] = act.arg;
            }
        }
        return py::make_tuple(types, args);
    }

    Game game(int64_t i) const {
        if (i < 0 || i >= n()) {
            throw py::value_error("game: index must be in 0.." + to_string(n() - 1) + ", got " +
                                  to_string(i));
        }
        lock_guard<mutex> lock(mu_);
        return games_[i];
    }

private:
    // Raw outputs of one step, one entry (or four VP bytes) per game.
    struct Out {
        bool* done;
        int8_t* winner;
        uint8_t* vp;
        uint16_t* turns;
    };

    // Checks every action, then applies them; returns the first illegal game's index (nothing
    // changed) or -1. Games that end are recorded in `out` and restarted on the next seed, in
    // game order.
    int64_t step_all(const int64_t* types, const int64_t* args, const Out& out) {
        for (size_t i = 0; i < games_.size(); ++i) {
            if (!games_[i].legal(types[i], args[i])) return static_cast<int64_t>(i);
        }
        memset(out.done, 0, games_.size() * sizeof(bool));
        memset(out.winner, 0xFF, games_.size());  // -1: no winner yet
        memset(out.vp, 0, games_.size() * NUM_PLAYERS);
        memset(out.turns, 0, games_.size() * sizeof(uint16_t));

        for (size_t i = 0; i < games_.size(); ++i) {
            Game& g = games_[i];
            g.act(*to_action(types[i], args[i]), nullptr);
            if (g.decision.kind != DecisionKind::TERMINAL) continue;
            out.done[i] = true;
            out.winner[i] = g.winner == NONE ? int8_t{-1} : static_cast<int8_t>(g.winner);
            for (int p = 0; p < NUM_PLAYERS; ++p) {
                out.vp[i * NUM_PLAYERS + p] = static_cast<uint8_t>(final_vp(g.state, p));
            }
            out.turns[i] = g.state.pub.turn;
            g.restart(next_seed_++);
        }
        return -1;
    }

    vector<Game> games_;
    uint64_t next_seed_;
    // The GIL is released while stepping and encoding, so two Python threads could otherwise
    // work on the same VecGames at once. Never held while waiting for the GIL.
    mutable mutex mu_;
};

// ---- Module ----------------------------------------------------------------------------------

template <class E>
void bind_enum(py::module_& m, const char* name, const vector<pair<const char*, E>>& values) {
    py::native_enum<E> e(m, name, "enum.IntEnum");
    for (const auto& [label, value] : values) e.value(label, value);
    e.finalize();
}

void bind_enums(py::module_& m) {
    using A = ActionType;
    bind_enum<A>(m, "ActionType",
                 {{"ROLL", A::ROLL}, {"BUILD_ROAD", A::BUILD_ROAD},
                  {"BUILD_SETTLEMENT", A::BUILD_SETTLEMENT}, {"BUILD_CITY", A::BUILD_CITY},
                  {"BUY_DEV", A::BUY_DEV}, {"PLAY_DEV", A::PLAY_DEV},
                  {"BANK_EXCHANGE", A::BANK_EXCHANGE}, {"MOVE_ROBBER", A::MOVE_ROBBER},
                  {"STEAL", A::STEAL}, {"DISCARD", A::DISCARD}, {"END_TURN", A::END_TURN},
                  {"PROPOSE_TRADE", A::PROPOSE_TRADE}, {"RESPOND_TRADE", A::RESPOND_TRADE}});
    using P = Phase;
    bind_enum<P>(m, "Phase",
                 {{"SETUP", P::SETUP}, {"ROLL", P::ROLL}, {"DISCARD", P::DISCARD},
                  {"MOVE_ROBBER", P::MOVE_ROBBER}, {"STEAL", P::STEAL}, {"MAIN", P::MAIN},
                  {"ROAD_BUILDING", P::ROAD_BUILDING}, {"YOP_PICK", P::YOP_PICK},
                  {"MONOPOLY_PICK", P::MONOPOLY_PICK}, {"EXCHANGE_RECV", P::EXCHANGE_RECV},
                  {"TERMINAL", P::TERMINAL}});
    using R = Resource;
    bind_enum<R>(m, "Resource",
                 {{"BRICK", R::BRICK}, {"LUMBER", R::LUMBER}, {"WOOL", R::WOOL},
                  {"GRAIN", R::GRAIN}, {"ORE", R::ORE}});
    using D = DevCard;
    bind_enum<D>(m, "DevCard",
                 {{"KNIGHT", D::KNIGHT}, {"VICTORY_POINT", D::VICTORY_POINT},
                  {"ROAD_BUILDING", D::ROAD_BUILDING}, {"YEAR_OF_PLENTY", D::YEAR_OF_PLENTY},
                  {"MONOPOLY", D::MONOPOLY}});
    using K = DecisionKind;
    bind_enum<K>(m, "DecisionKind",
                 {{"PLAYER", K::PLAYER}, {"CHANCE", K::CHANCE}, {"TERMINAL", K::TERMINAL}});
    using T = EventType;
    bind_enum<T>(m, "EventType",
                 {{"SETTLEMENT_BUILT", T::SETTLEMENT_BUILT}, {"CITY_BUILT", T::CITY_BUILT},
                  {"ROAD_BUILT", T::ROAD_BUILT}, {"GAIN", T::GAIN}, {"SPEND", T::SPEND},
                  {"DISCARD", T::DISCARD}, {"DICE_ROLLED", T::DICE_ROLLED},
                  {"ROBBER_MOVED", T::ROBBER_MOVED}, {"STEAL", T::STEAL},
                  {"DEV_BOUGHT", T::DEV_BOUGHT}, {"DEV_PLAYED", T::DEV_PLAYED},
                  {"MONOPOLY", T::MONOPOLY}, {"LONGEST_ROAD", T::LONGEST_ROAD},
                  {"LARGEST_ARMY", T::LARGEST_ARMY}, {"TURN_ENDED", T::TURN_ENDED},
                  {"GAME_OVER", T::GAME_OVER}});
}

void bind_event(py::module_& m) {
    py::class_<Event>(m, "Event", "One thing that happened, carrying the full truth.")
        .def_property_readonly("type", [](const Event& e) { return e.type; })
        .def_property_readonly("player", [](const Event& e) { return seat_or_none(e.player); })
        .def_property_readonly("target", [](const Event& e) { return seat_or_none(e.target); })
        .def_property_readonly("a", [](const Event& e) { return int{e.a}; })
        .def_property_readonly("b", [](const Event& e) { return int{e.b}; })
        .def("describe", event_describe, py::arg("observer") = py::none(),
             "describe_event: hides a steal's resource / a bought card from other observers.")
        .def("__str__", [](const Event& e) { return event_describe(e, nullopt); })
        .def("__repr__", event_repr);
}

void bind_game(py::module_& m) {
    py::class_<Game>(m, "Game",
                     "One game: board, full state, every observer's belief, and the seed for "
                     "chance.")
        .def(py::init<uint64_t, bool, bool>(), py::arg("seed"),
             py::arg("standard_board") = false, py::arg("auto_chance") = true)
        .def_property_readonly("seed", [](const Game& g) { return g.seed; })
        // state
        .def_property_readonly("kind", [](const Game& g) { return g.decision.kind; })
        .def_property_readonly("phase", [](const Game& g) { return g.state.pub.phase; })
        .def_property_readonly("decider", game_decider)
        .def_property_readonly("current_player",
                               [](const Game& g) { return int{g.state.pub.current_player}; })
        .def_property_readonly("turn", [](const Game& g) { return int{g.state.pub.turn}; })
        .def_property_readonly("ply", [](const Game& g) { return int{g.state.pub.ply}; })
        .def_property_readonly(
            "done", [](const Game& g) { return g.decision.kind == DecisionKind::TERMINAL; })
        .def_property_readonly("winner", [](const Game& g) { return seat_or_none(g.winner); })
        .def_property_readonly("public_vp",
                               [](const Game& g) { return ints(g.state.pub.public_vp); })
        .def_property_readonly("vp", game_vp)
        .def("hand",
             [](const Game& g, int64_t p) {
                 return ints(g.state.hid.hands[seat_arg(p, "player")]);
             },
             py::arg("player"))
        .def("dev_cards",
             [](const Game& g, int64_t p) {
                 return ints(g.state.hid.dev_held[seat_arg(p, "player")]);
             },
             py::arg("player"))
        .def("board", game_board)
        // acting
        .def("legal_actions", game_legal_actions)
        .def("is_legal", &Game::legal, py::arg("type"), py::arg("arg") = 0)
        .def("step", game_step, py::arg("type"), py::arg("arg") = 0)
        .def("chance_weights", game_chance_weights)
        .def("apply_chance", game_apply_chance, py::arg("outcome"))
        .def("greedy_action", game_greedy_action)
        // network input
        .def("observation", game_observation, py::arg("observer") = py::none())
        .def("mask", game_mask)
        // text
        .def("describe", game_describe, py::arg("type"), py::arg("arg") = 0)
        .def("render", game_render, py::arg("observer") = py::none(),
             py::arg("vertex_ids") = false, py::arg("edge_ids") = false)
        .def("__str__", [](const Game& g) { return game_render(g, nullopt, false, false); })
        // misc
        .def("copy", [](const Game& g) { return g; })
        .def("state_hash", [](const Game& g) { return state_hash(g.state); });
}

void bind_vec_games(py::module_& m) {
    py::class_<VecGames>(m, "VecGames", "n independent games for batched play.")
        .def(py::init<py::ssize_t, uint64_t, bool>(), py::arg("n"), py::arg("first_seed") = 1,
             py::arg("standard_board") = false)
        .def_property_readonly("n", &VecGames::n)
        .def_property_readonly("seeds", &VecGames::seeds)
        .def("observe", &VecGames::observe,
             "(obs (n, OBS_SIZE), mask (n, MASK_SIZE), decider (n,)) of every game's decider.")
        .def("step", &VecGames::step, py::arg("types"), py::arg("args"),
             "Apply one action per game; returns (done, winner, vp, turns). Finished games "
             "restart at once.")
        .def("greedy_actions", &VecGames::greedy_actions)
        .def("game", &VecGames::game, py::arg("i"));
}

}  // namespace

PYBIND11_MODULE(catan_engine, m) {
    m.doc() = "Python bindings for the C++ Catan engine (see catan_engine.pyi).";

    m.attr("ENCODE_VERSION") = ENCODE_VERSION;
    m.attr("OBS_SIZE") = OBS_SIZE;
    m.attr("MASK_SIZE") = MASK_SIZE;
    m.attr("NUM_PLAYERS") = NUM_PLAYERS;
    m.attr("NUM_ACTION_TYPES") = NUM_ACTION_TYPES;
    m.attr("VICTORY_POINTS_TO_WIN") = VICTORY_POINTS_TO_WIN;
    m.attr("TURN_CAP") = int{TURN_CAP};
    m.attr("OBS_SECTIONS") = obs_sections();
    m.attr("MASK_SECTIONS") = mask_sections();

    bind_enums(m);
    bind_event(m);
    bind_game(m);
    bind_vec_games(m);

    m.def("random_below", random_below_arg, py::arg("seed"), py::arg("stream"), py::arg("index"),
          py::arg("bound"), "The engine's counter-based RNG: uniform in [0, bound).");
    m.def("topology", topology_dict, "Fixed board geometry as nested lists.");
}
