"""Tests for the catan_engine Python module against bind/python/catan_engine.pyi (the contract).

Beyond the module surface, these tests prove that Python drives the engine exactly as the C++
drivers do: they replay the C++ golden games (core/tests/test_golden.cpp) and greedy-mirror games
(eval's play_game) to the same final states, resolving chance both automatically and by hand with
a Python transcription of sample_chance (core/include/catan/chance.h). Every VecGames slot is
shadowed by a standalone Game fed the same actions. Hidden information is checked by forking a
game at a steal or a dev-card draw: seats that may not see the outcome must see no difference.

Run: PYTHONPATH=<build>/bind CATAN_PY_PLAY=bind/python/play.py python -m pytest bind/tests -q
"""

import enum
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

import catan_engine as ce

M64 = (1 << 64) - 1

# ---- layout constants from encode/include/catan/encode.h ------------------------------------

OBS_SELF = 1938
SELF_HAND = 40
SELF_DEV_HELD = 45
SELF_TOTAL_VP = 55
OBS_PLAYERS = 1994
PLAYER_FEATURES = 79
PLAYER_HAND_MIN = 0
PLAYER_HAND_MAX = 5
PLAYER_HAND_SIZE = 10
PLAYER_DEV_COUNT = 11
PLAYER_PUBLIC_VP = 13
PLAYER_IS_CURRENT = 21
OBS_GLOBAL = 2310
GLOBAL_PHASE = 0
GLOBAL_SETUP_STEP = 43
GLOBAL_TURN = 44

DICE_WEIGHTS = [1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]
DEV_DECK = [14, 5, 2, 2, 2]  # DEV_DECK_COUNTS, by DevCard
DEV_DECK_SIZE = 25
VP_CARD = 1  # DevCard.VICTORY_POINT

# Pinned in core/tests/test_golden.cpp: (seed, standard board, state_hash, ply, turn, winner).
GOLDENS = [
    (1, False, 0x4988F73E02852A53, 1733, 352, 0),
    (2, False, 0x1CCC3D1E296214FB, 2524, 549, 1),
    (3, False, 0x944F636B0047E8C9, 997, 213, 1),
    (42, False, 0x7E94D47EE545CAEF, 1379, 289, 1),
    (1000, False, 0xEF814B2573A0D7C2, 1758, 395, 3),
    (123456789, False, 0x8AE23C5D86CB2F2E, 897, 170, 2),
    (7, True, 0x970D831133CFA2C0, 1127, 232, 0),
]
SETUP_HASH_SEED_1 = 0x684BA6B6E849A706  # random_setup(random_board(1), 1)
ROLLING_50_SEED_1 = 0x1E2837654CE4C242  # fold of the first 50 per-ply state hashes, seed 1

# Four greedy seats, from eval's play_game (C++), not from the binding:
# (seed, standard board, winner, true final VP, turns, decisions).
GREEDY_GAMES = [
    (1, False, 1, (6, 10, 8, 7), 77, 473),
    (2, False, 1, (8, 10, 5, 6), 61, 346),
    (3, False, 1, (6, 10, 4, 5), 61, 347),
    (4, False, 0, (10, 7, 6, 6), 88, 518),
    (5, False, 1, (8, 10, 3, 7), 77, 471),
    (6, False, 0, (10, 6, 6, 4), 80, 448),
    (7, False, 3, (5, 6, 7, 10), 91, 500),
    (8, False, 0, (10, 6, 7, 5), 72, 399),
    (7, True, 2, (9, 6, 10, 6), 90, 528),
]

# random_board(seed) from C++ (hex types, tokens, port types, desert).
RANDOM_BOARD_1 = (
    [4, 2, 3, 0, 5, 4, 2, 1, 1, 3, 2, 3, 4, 1, 2, 0, 1, 0, 3],
    [2, 4, 8, 6, 0, 4, 3, 11, 9, 5, 3, 12, 5, 9, 11, 8, 6, 10, 10],
    [5, 2, 3, 5, 1, 5, 0, 4, 5],
    4,
)
# standard_board() (core/src/board.cpp).
STANDARD_BOARD = (
    [4, 2, 1, 3, 0, 2, 0, 3, 1, 5, 1, 4, 1, 4, 3, 2, 0, 3, 2],
    [10, 2, 9, 12, 6, 4, 10, 9, 11, 0, 3, 8, 8, 3, 4, 5, 5, 6, 11],
    [5, 3, 4, 5, 2, 5, 5, 0, 1],
    9,
)

ENUMS = {
    "ActionType": [
        "ROLL", "BUILD_ROAD", "BUILD_SETTLEMENT", "BUILD_CITY", "BUY_DEV", "PLAY_DEV",
        "BANK_EXCHANGE", "MOVE_ROBBER", "STEAL", "DISCARD", "END_TURN", "PROPOSE_TRADE",
        "RESPOND_TRADE",
    ],
    "Phase": [
        "SETUP", "ROLL", "DISCARD", "MOVE_ROBBER", "STEAL", "MAIN", "ROAD_BUILDING",
        "YOP_PICK", "MONOPOLY_PICK", "EXCHANGE_RECV", "TERMINAL",
    ],
    "Resource": ["BRICK", "LUMBER", "WOOL", "GRAIN", "ORE"],
    "DevCard": ["KNIGHT", "VICTORY_POINT", "ROAD_BUILDING", "YEAR_OF_PLENTY", "MONOPOLY"],
    "DecisionKind": ["PLAYER", "CHANCE", "TERMINAL"],
    "EventType": [
        "SETTLEMENT_BUILT", "CITY_BUILT", "ROAD_BUILT", "GAIN", "SPEND", "DISCARD",
        "DICE_ROLLED", "ROBBER_MOVED", "STEAL", "DEV_BOUGHT", "DEV_PLAYED", "MONOPOLY",
        "LONGEST_ROAD", "LARGEST_ARMY", "TURN_ENDED", "GAME_OVER",
    ],
}

AT = ce.ActionType
PH = ce.Phase
PLAYER = ce.DecisionKind.PLAYER
CHANCE = ce.DecisionKind.CHANCE

# Parameter head of each type (action.h param_head), as a MASK_SECTIONS name.
HEADS = {
    AT.BUILD_ROAD: "edge",
    AT.BUILD_SETTLEMENT: "settlement",
    AT.BUILD_CITY: "city",
    AT.PLAY_DEV: "devcard",
    AT.MOVE_ROBBER: "tile",
    AT.STEAL: "player",
    AT.BANK_EXCHANGE: "resource",
    AT.DISCARD: "resource",
}


# ---- reference helpers ----------------------------------------------------------------------


def splitmix64(x):
    x = (x + 0x9E3779B97F4A7C15) & M64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & M64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & M64
    return x ^ (x >> 31)


def random_u64(seed, stream, index):
    """rng.h, transcribed."""
    return splitmix64(splitmix64(splitmix64(seed) ^ stream) ^ index)


def fold(acc, h):
    """The order-sensitive hash fold of test_golden.cpp."""
    return ((acc ^ h) * 0x100000001B3 + 0x9E3779B97F4A7C15) & M64


def head_of(phase, type_):
    if phase in (PH.YOP_PICK, PH.MONOPOLY_PICK):
        return "resource"
    return HEADS.get(type_)


def actions_from_mask(mask, phase):
    """Every (type, arg) the mask flags imply, in canonical order (type, then arg)."""
    off, size = ce.MASK_SECTIONS["type"]
    out = []
    for t in range(size):
        if not mask[off + t]:
            continue
        head = head_of(phase, t)
        if head is None:
            out.append((t, 0))
            continue
        hoff, hsize = ce.MASK_SECTIONS[head]
        out.extend((t, a) for a in range(hsize) if mask[hoff + a])
    return out


def mask_from_actions(actions, phase):
    mask = np.zeros(ce.MASK_SIZE, dtype=np.uint8)
    toff = ce.MASK_SECTIONS["type"][0]
    for t, a in actions:
        mask[toff + t] = 1
        head = head_of(phase, t)
        if head is not None:
            mask[ce.MASK_SECTIONS[head][0] + a] = 1
    return mask


def chance_kind(game):
    """Which ChanceKind is pending: dice follow ROLL, a steal keeps the STEAL phase, and a dev
    draw follows BUY_DEV in MAIN."""
    if game.phase == PH.ROLL:
        return "dice"
    if game.phase == PH.STEAL:
        return "steal"
    assert game.phase == PH.MAIN, game.phase
    return "dev"


def sample_chance(game, seed):
    """sample_chance (chance.h) from Python: stream and index by kind, then pick_outcome."""
    weights = game.chance_weights()
    total = sum(weights)
    kind = chance_kind(game)
    if kind == "dice":
        stream, index = 2, game.turn
    elif kind == "steal":
        stream, index = 4, game.ply
    else:
        stream, index = 3, DEV_DECK_SIZE - total  # the weights are the deck
    x = ce.random_below(seed, stream, index, total)
    for outcome, w in enumerate(weights):
        if x < w:
            return outcome
        x -= w
    raise AssertionError("weights do not sum to their total")


def random_index(seed, ply, n):
    """play_random's choice among n legal actions (fixtures.cpp)."""
    return ce.random_below(seed, 1000, ply, n)


def play_to_end(game, seed, choose=None):
    """Plays a game (auto_chance on) to the end with play_random's choices, or choose(game).
    Returns the events of the last step."""
    events = []
    while not game.done:
        if choose is None:
            legal = game.legal_actions()
            action = legal[random_index(seed, game.ply, len(legal))]
        else:
            action = choose(game)
        events = game.step(*action)
    return events


def self_section(obs):
    return obs[OBS_SELF:OBS_SELF + 56]


def player_row(obs, rel):
    start = OBS_PLAYERS + rel * PLAYER_FEATURES
    return obs[start:start + PLAYER_FEATURES]


def game_with_history(seed, history, standard=False):
    """A fresh Game (auto_chance off) replaying ("step", type, arg) / ("chance", outcome)."""
    g = ce.Game(seed, standard_board=standard, auto_chance=False)
    for item in history:
        if item[0] == "step":
            g.step(item[1], item[2])
        else:
            g.apply_chance(item[1])
    return g


def snapshot(vg):
    obs, mask, dec = vg.observe()
    hashes = [vg.game(i).state_hash() for i in range(vg.n)]
    return obs.copy(), mask.copy(), dec.copy(), vg.seeds.copy(), hashes


def assert_same_snapshot(a, b):
    for x, y in zip(a[:4], b[:4]):
        np.testing.assert_array_equal(x, y)
    assert a[4] == b[4]


# ---- module surface -------------------------------------------------------------------------


def test_constants_match_the_engine():
    assert ce.ENCODE_VERSION == 1
    assert ce.OBS_SIZE == 2355
    assert ce.MASK_SIZE == 225
    assert ce.NUM_PLAYERS == 4
    assert ce.NUM_ACTION_TYPES == 13
    assert ce.VICTORY_POINTS_TO_WIN == 10
    assert ce.TURN_CAP == 1000
    for name in ("ENCODE_VERSION", "OBS_SIZE", "MASK_SIZE", "NUM_PLAYERS", "NUM_ACTION_TYPES",
                 "VICTORY_POINTS_TO_WIN", "TURN_CAP"):
        assert type(getattr(ce, name)) is int, name


def test_obs_sections_are_the_encode_table_and_cover_the_observation():
    assert ce.OBS_SECTIONS == {
        "tiles": (0, 19, 102),
        "self": (1938, 1, 56),
        "players": (1994, 4, 79),
        "global": (2310, 1, 45),
    }
    at = 0
    for offset, rows, width in sorted(ce.OBS_SECTIONS.values()):
        assert offset == at
        at += rows * width
    assert at == ce.OBS_SIZE


def test_mask_sections_are_contiguous_and_sized_by_their_heads():
    assert ce.MASK_SECTIONS == {
        "type": (0, 13),
        "settlement": (13, 54),
        "city": (67, 54),
        "edge": (121, 72),
        "tile": (193, 19),
        "devcard": (212, 5),
        "player": (217, 3),
        "resource": (220, 5),
    }
    at = 0
    for offset, size in sorted(ce.MASK_SECTIONS.values()):
        assert offset == at
        at += size
    assert at == ce.MASK_SIZE
    assert ce.MASK_SECTIONS["type"][1] == ce.NUM_ACTION_TYPES


@pytest.mark.parametrize("name", sorted(ENUMS))
def test_enums_are_int_enums_with_the_listed_values(name):
    cls = getattr(ce, name)
    assert isinstance(cls, type) and issubclass(cls, enum.IntEnum)
    assert [(m.name, int(m.value)) for m in cls] == [(n, i) for i, n in enumerate(ENUMS[name])]
    for i, member_name in enumerate(ENUMS[name]):
        assert cls(i) is getattr(cls, member_name)
        assert cls[member_name] == i


def test_random_below_is_the_engines_counter_rng():
    cases = [(0, 0, 0), (1, 1000, 5), (42, 2, 17), (M64, 5, M64), (1 << 63, 99, 123456789)]
    for seed, stream, index in cases:
        for bound in (1, 2, 6, 36, 1000, (1 << 32) - 1):
            want = random_u64(seed, stream, index) % bound
            assert ce.random_below(seed, stream, index, bound) == want
    # Pure in its inputs.
    assert ce.random_below(7, 3, 9, 100) == ce.random_below(7, 3, 9, 100)


def test_topology_sizes_and_symmetry():
    topo = ce.topology()
    assert set(topo) == {
        "hex_coord", "hex_vertices", "hex_edges", "hex_neighbors", "vertex_hexes",
        "vertex_vertices", "vertex_edges", "edge_vertices", "port_slots",
    }
    assert len(topo["hex_coord"]) == 19
    assert tuple(topo["hex_coord"][9]) == (0, 0)
    assert len(set(map(tuple, topo["hex_coord"]))) == 19
    for key in ("hex_vertices", "hex_edges", "hex_neighbors"):
        assert len(topo[key]) == 19 and all(len(row) == 6 for row in topo[key]), key
    for key in ("vertex_hexes", "vertex_vertices", "vertex_edges"):
        assert len(topo[key]) == 54, key
        assert all(1 <= len(row) <= 3 and None not in row for row in topo[key]), key
    assert len(topo["edge_vertices"]) == 72
    assert len(topo["port_slots"]) == 9

    # Hex adjacency is symmetric; None marks the sea.
    for h, row in enumerate(topo["hex_neighbors"]):
        for n in row:
            if n is not None:
                assert h in topo["hex_neighbors"][n]
    assert sum(n is None for row in topo["hex_neighbors"] for n in row) == 30
    # Vertex adjacency is symmetric and every vertex/edge/hex incidence is listed both ways.
    for v, row in enumerate(topo["vertex_vertices"]):
        assert row == sorted(row)
        for w in row:
            assert v in topo["vertex_vertices"][w]
    for e, (v0, v1) in enumerate(topo["edge_vertices"]):
        assert v0 < v1
        assert e in topo["vertex_edges"][v0] and e in topo["vertex_edges"][v1]
        assert v1 in topo["vertex_vertices"][v0]
    for h in range(19):
        corners, sides = topo["hex_vertices"][h], topo["hex_edges"][h]
        for i in range(6):
            assert h in topo["vertex_hexes"][corners[i]]
            ends = sorted((corners[i], corners[(i + 1) % 6]))
            assert list(topo["edge_vertices"][sides[i]]) == ends
    for v, hexes in enumerate(topo["vertex_hexes"]):
        for h in hexes:
            assert v in topo["hex_vertices"][h]
    assert sum(len(row) for row in topo["vertex_edges"]) == 2 * 72

    # The harbor table of topology.h.
    assert [tuple(slot) for slot in topo["port_slots"]] == [
        (0, 0, 3, 0), (3, 1, 5, 1), (17, 10, 15, 6), (38, 26, 32, 11), (61, 42, 46, 15),
        (69, 49, 52, 17), (66, 47, 51, 16), (49, 33, 38, 12), (18, 11, 16, 3),
    ]


@pytest.mark.parametrize("seed,standard,want", [(1, False, RANDOM_BOARD_1),
                                                (99, True, STANDARD_BOARD)])
def test_board_is_random_board_of_the_seed_or_the_standard_board(seed, standard, want):
    board = ce.Game(seed, standard_board=standard).board()
    assert board == {"hex_type": want[0], "token": want[1], "port_type": want[2],
                     "robber": want[3]}


# ---- driving the engine exactly as C++ does -------------------------------------------------


@pytest.mark.parametrize("seed,standard,hash_,ply,turn,winner", GOLDENS)
def test_golden_games_replay_from_python(seed, standard, hash_, ply, turn, winner):
    g = ce.Game(seed, standard_board=standard)
    assert g.seed == seed
    last = play_to_end(g, seed)
    assert (g.state_hash(), g.ply, g.turn, g.winner) == (hash_, ply, turn, winner)
    over = [e for e in last if e.type == ce.EventType.GAME_OVER]
    assert len(over) == 1 and over[0].player == winner


def test_golden_setup_and_first_plies_with_chance_resolved_in_python():
    setup = ce.Game(1)
    for _ in range(16):
        legal = setup.legal_actions()
        setup.step(*legal[ce.random_below(1, 99, setup.ply, len(legal))])
    assert setup.state_hash() == SETUP_HASH_SEED_1
    assert setup.phase == PH.ROLL and setup.turn == 0

    g = ce.Game(1, auto_chance=False)
    acc = 0
    for _ in range(50):
        if g.kind == CHANCE:
            g.apply_chance(sample_chance(g, 1))
        else:
            legal = g.legal_actions()
            g.step(*legal[random_index(1, g.ply, len(legal))])
        acc = fold(acc, g.state_hash())
    assert g.ply == 50
    assert acc == ROLLING_50_SEED_1


@pytest.mark.parametrize("golden", [GOLDENS[2], GOLDENS[5], GOLDENS[6]], ids=str)
def test_golden_games_replay_with_chance_resolved_in_python(golden):
    seed, standard, hash_, ply, turn, winner = golden
    g = ce.Game(seed, standard_board=standard, auto_chance=False)
    kinds = Counter()
    victim = None
    drawn = 0
    while not g.done:
        if g.kind == CHANCE:
            weights = g.chance_weights()
            kind = chance_kind(g)
            kinds[kind] += 1
            if kind == "dice":
                assert weights == DICE_WEIGHTS
            else:
                assert len(weights) >= 5 and not any(weights[5:])
                if kind == "steal":
                    assert weights[:5] == g.hand(victim)
                else:
                    assert all(w <= d for w, d in zip(weights, DEV_DECK))
                    assert sum(weights) == DEV_DECK_SIZE - drawn
                    drawn += 1
            assert g.decider is None and g.legal_actions() == []
            assert not g.mask().any()
            g.apply_chance(sample_chance(g, seed))
        else:
            assert g.kind == PLAYER and g.chance_weights() == []
            legal = g.legal_actions()
            t, a = legal[random_index(seed, g.ply, len(legal))]
            if t == AT.STEAL:
                victim = (g.decider + 1 + a) % 4
            g.step(t, a)
    assert (g.state_hash(), g.ply, g.turn, g.winner) == (hash_, ply, turn, winner)
    assert kinds["dice"] in (turn, turn + 1)  # the last turn may end before its roll
    assert kinds["steal"] > 0 and kinds["dev"] > 0


@pytest.mark.parametrize("game", GREEDY_GAMES, ids=str)
def test_four_greedy_seats_play_the_cpp_greedy_game(game):
    seed, standard, winner, vp, turns, decisions = game
    g = ce.Game(seed, standard_board=standard)
    while not g.done:
        action = g.greedy_action()
        assert action in g.legal_actions()
        g.step(*action)
    assert g.winner == winner
    assert tuple(g.vp) == vp
    assert g.turn == turns
    assert g.ply == decisions


# ---- legal actions, mask and errors ---------------------------------------------------------


def test_legal_actions_are_exactly_what_the_mask_flags():
    phases = Counter()
    for seed in range(11, 19):
        g = ce.Game(seed, standard_board=seed % 2 == 0)
        while not g.done:
            phase = g.phase
            phases[phase] += 1
            legal = g.legal_actions()
            mask = g.mask()
            assert mask.dtype == np.uint8 and mask.shape == (ce.MASK_SIZE,)
            assert legal == actions_from_mask(mask, phase)
            np.testing.assert_array_equal(mask, mask_from_actions(legal, phase))
            assert legal == sorted(set(legal)) and legal
            if g.ply % 37 == 0:
                legal_set = set(legal)
                for t in range(ce.NUM_ACTION_TYPES):
                    for a in range(80):
                        assert g.is_legal(t, a) == ((t, a) in legal_set), (t, a)
            g.step(*legal[ce.random_below(seed, 5000, g.ply, len(legal))])
    for phase in (PH.SETUP, PH.ROLL, PH.MAIN, PH.DISCARD, PH.MOVE_ROBBER, PH.STEAL,
                  PH.ROAD_BUILDING, PH.YOP_PICK, PH.MONOPOLY_PICK, PH.EXCHANGE_RECV):
        assert phases[phase] > 0, phase


def test_step_rejects_illegal_actions_and_leaves_the_game_unchanged():
    g = ce.Game(5)
    checked = 0
    while not g.done and g.ply < 600:
        legal = set(g.legal_actions())
        if g.ply % 7 == 0:
            candidates = [(t, a) for t in range(ce.NUM_ACTION_TYPES) for a in (0, 1, 2, 5, 53)]
            candidates += [(13, 0), (255, 0), (1, 72), (2, 54), (3, 54), (7, 19), (1, 255)]
            before = (g.state_hash(), g.ply, g.observation().tobytes())
            for t, a in candidates:
                if (t, a) in legal:
                    continue
                assert not g.is_legal(t, a)
                with pytest.raises(ValueError):
                    g.step(t, a)
                checked += 1
            assert (g.state_hash(), g.ply, g.observation().tobytes()) == before
        legal = sorted(legal)
        g.step(*legal[ce.random_below(5, 5000, g.ply, len(legal))])
    assert checked > 1000


def test_step_and_is_legal_reject_out_of_range_ints_with_value_error():
    g = ce.Game(3)
    h = g.state_hash()
    t0, a0 = g.legal_actions()[0]
    # Including values that would wrap to that legal action in a uint8.
    for t, a in [(t0, a0 + 256), (t0 + 256, a0), (t0, a0 - 256), (1, 300), (2, -1), (-1, 0),
                 (1000, 0), (0, 4096)]:
        assert g.is_legal(t, a) is False
        with pytest.raises(ValueError):
            g.step(t, a)
    assert g.state_hash() == h


def test_step_needs_a_player_decision():
    g = ce.Game(3, auto_chance=False)
    while g.kind == PLAYER:
        legal = g.legal_actions()
        g.step(*legal[random_index(3, g.ply, len(legal))])
    assert g.kind == CHANCE
    h = g.state_hash()
    with pytest.raises(ValueError):
        g.step(AT.ROLL, 0)
    with pytest.raises(ValueError):
        g.greedy_action()
    with pytest.raises(ValueError):
        g.observation()
    assert g.state_hash() == h
    assert g.observation(0).shape == (ce.OBS_SIZE,)


def test_apply_chance_validates_the_outcome():
    g = ce.Game(3)  # auto_chance: never waits on chance
    with pytest.raises(ValueError):
        g.apply_chance(0)
    assert g.chance_weights() == []

    g = ce.Game(3, auto_chance=False)
    zero_seen = Counter()
    while not g.done and g.ply < 800:
        if g.kind == CHANCE:
            weights = g.chance_weights()
            h = g.state_hash()
            bad = [i for i, w in enumerate(weights) if w == 0] + [len(weights), 11, 200, 255]
            for outcome in bad:
                with pytest.raises(ValueError):
                    g.apply_chance(outcome)
                zero_seen[chance_kind(g)] += outcome < len(weights)
            assert g.state_hash() == h
            g.apply_chance(sample_chance(g, 3))
        else:
            with pytest.raises(ValueError):
                g.apply_chance(0)
            legal = g.legal_actions()
            g.step(*legal[random_index(3, g.ply, len(legal))])
    assert zero_seen["steal"] > 0


def test_default_args_and_keywords():
    g = ce.Game(seed=4, standard_board=False, auto_chance=True)
    for _ in range(16):
        legal = g.legal_actions()
        g.step(*legal[0])
    assert g.phase == PH.ROLL
    assert g.is_legal(AT.ROLL)
    assert g.describe(AT.ROLL) == g.describe(type=AT.ROLL, arg=0)
    events = g.step(AT.ROLL)  # arg defaults to 0
    assert any(e.type == ce.EventType.DICE_ROLLED for e in events)
    assert g.render() == g.render(observer=None, vertex_ids=False, edge_ids=False)
    np.testing.assert_array_equal(g.observation(), g.observation(observer=g.decider))


# ---- state and events -----------------------------------------------------------------------


def test_state_properties_during_play_and_at_the_end():
    seed = GOLDENS[2][0]
    g = ce.Game(seed)
    assert isinstance(g.kind, ce.DecisionKind) and isinstance(g.phase, ce.Phase)
    assert (g.turn, g.ply, g.current_player, g.decider) == (0, 0, 0, 0)
    last = []
    while not g.done:
        assert g.kind == PLAYER and g.winner is None
        assert g.decider in range(4) and g.current_player in range(4)
        pub, vp = g.public_vp, g.vp
        for p in range(4):
            assert pub[p] <= vp[p]
            assert vp[p] - pub[p] == g.dev_cards(p)[VP_CARD]
            assert len(g.hand(p)) == 5 and len(g.dev_cards(p)) == 5
        if g.phase not in (PH.SETUP, PH.DISCARD):  # discards: any seat may owe cards
            assert g.decider == g.current_player
        legal = g.legal_actions()
        last = g.step(*legal[random_index(seed, g.ply, len(legal))])
        for e in last:
            assert isinstance(e.type, ce.EventType)
            assert e.player is None or e.player in range(4)
            assert e.target is None or e.target in range(4)
            assert isinstance(e.a, int) and isinstance(e.b, int)

    assert g.done and g.kind == ce.DecisionKind.TERMINAL and g.phase == PH.TERMINAL
    assert g.winner == GOLDENS[2][5]
    assert g.vp[g.winner] >= ce.VICTORY_POINTS_TO_WIN
    assert all(p <= v for p, v in zip(g.public_vp, g.vp))
    assert last[-1].type == ce.EventType.GAME_OVER and last[-1].player == g.winner
    assert g.decider is None
    assert g.legal_actions() == [] and g.chance_weights() == []
    assert not g.mask().any()
    h = g.state_hash()
    for call in (lambda: g.step(AT.END_TURN, 0), g.greedy_action, g.observation,
                 lambda: g.apply_chance(0)):
        with pytest.raises(ValueError):
            call()
    assert g.state_hash() == h
    for o in range(4):
        assert g.observation(o).shape == (ce.OBS_SIZE,)
    assert isinstance(g.render(), str) and g.render()


def test_out_of_range_seats_raise():
    g = ce.Game(2)
    for call in (lambda: g.hand(4), lambda: g.dev_cards(4), lambda: g.observation(4),
                 lambda: g.hand(-1)):
        with pytest.raises((ValueError, IndexError, TypeError)):
            call()


def test_events_describe_and_print():
    g = ce.Game(6)
    seen = Counter()
    while not g.done and g.ply < 1500:
        legal = g.legal_actions()
        for e in g.step(*legal[random_index(6, g.ply, len(legal))]):
            seen[e.type] += 1
            text = str(e)
            assert text and "\n" not in text
            assert text == e.describe() == e.describe(None)
            assert repr(e)
            for o in range(4):
                line = e.describe(o)
                assert line and "\n" not in line
                if e.type not in (ce.EventType.STEAL, ce.EventType.DEV_BOUGHT):
                    assert line == text
    assert seen[ce.EventType.DICE_ROLLED] > 0 and seen[ce.EventType.SETTLEMENT_BUILT] >= 8


def test_describe_names_every_legal_action_distinctly():
    g = ce.Game(8)
    steals = 0
    while not g.done:
        legal = g.legal_actions()
        texts = [g.describe(t, a) for t, a in legal]
        assert all(texts) and len(set(texts)) == len(texts)
        for (t, a), text in zip(legal, texts):
            if t == AT.STEAL:
                steals += 1
                assert text == f"steal from P{(g.decider + 1 + a) % 4}"
        g.step(*legal[random_index(8, g.ply, len(legal))])
    assert steals > 0


def test_render_is_text_and_str_is_the_god_view():
    g = ce.Game(9)
    for _ in range(300):
        legal = g.legal_actions()
        g.step(*legal[random_index(9, g.ply, len(legal))])
    god = g.render()
    assert isinstance(god, str) and god == str(g) and god == g.render(None)
    for o in range(4):
        others_hold_cards = any(sum(g.hand(p)) + sum(g.dev_cards(p)) for p in range(4) if p != o)
        assert (g.render(o) != god) == others_hold_cards
    with_ids = g.render(vertex_ids=True, edge_ids=True)
    assert with_ids != god
    assert len(re.findall(r"\d+", with_ids)) > len(re.findall(r"\d+", god))


# ---- observations ---------------------------------------------------------------------------


def test_observation_shape_default_observer_and_own_cards():
    for seed in (21, 22):
        g = ce.Game(seed)
        steps = 0
        while not g.done:
            obs = g.observation()
            assert obs.dtype == np.uint8 and obs.shape == (ce.OBS_SIZE,)
            np.testing.assert_array_equal(obs, g.observation(g.decider))
            if steps % 5 == 0:
                views = [g.observation(o) for o in range(4)]
                for o, view in enumerate(views):
                    own = self_section(view)
                    assert list(own[SELF_HAND:SELF_HAND + 5]) == g.hand(o)
                    assert list(own[SELF_DEV_HELD:SELF_DEV_HELD + 5]) == g.dev_cards(o)
                    assert own[SELF_TOTAL_VP] == g.vp[o]
                    for rel in range(4):
                        seat = (o + rel) % 4
                        row = player_row(view, rel)
                        hand = g.hand(seat)
                        # The belief the module feeds is sound: its bounds hold the truth,
                        # exactly for the observer's own hand.
                        low = list(row[PLAYER_HAND_MIN:PLAYER_HAND_MIN + 5])
                        high = list(row[PLAYER_HAND_MAX:PLAYER_HAND_MAX + 5])
                        assert all(lo <= h <= hi for lo, h, hi in zip(low, hand, high))
                        if rel == 0:
                            assert low == hand == high
                        assert row[PLAYER_HAND_SIZE] == sum(hand)
                        assert row[PLAYER_DEV_COUNT] == sum(g.dev_cards(seat))
                        assert row[PLAYER_PUBLIC_VP] == g.public_vp[seat]
                        assert row[PLAYER_IS_CURRENT] == (seat == g.current_player)
                    glob = view[OBS_GLOBAL:]
                    assert glob[GLOBAL_PHASE + int(g.phase)] == 1
                    assert glob[GLOBAL_TURN] == min(g.turn, 255)
                distinct = {v.tobytes() for v in views}
                assert len(distinct) == 4
            legal = g.legal_actions()
            g.step(*legal[random_index(seed, g.ply, len(legal))])
            steps += 1


def test_observation_and_mask_are_deterministic_snapshots():
    a, b = ce.Game(31), ce.Game(31)
    for _ in range(400):
        assert a.state_hash() == b.state_hash()
        for o in range(4):
            np.testing.assert_array_equal(a.observation(o), b.observation(o))
        np.testing.assert_array_equal(a.mask(), b.mask())
        obs, mask = a.observation(), a.mask()
        kept_obs, kept_mask = obs.copy(), mask.copy()
        legal = a.legal_actions()
        action = legal[random_index(31, a.ply, len(legal))]
        a.step(*action)
        b.step(*action)
        # Arrays already returned are not views of the game.
        np.testing.assert_array_equal(obs, kept_obs)
        np.testing.assert_array_equal(mask, kept_mask)


def _play_until_fork(seed, want, max_ply=2500):
    """Plays seed (auto_chance off) with random legal moves until a chance of kind `want`
    ("steal" or "dev") with two outcomes that change only hidden information. Returns
    (history, outcomes, involved seats) or None."""
    g = ce.Game(seed, auto_chance=False)
    history = []
    victim = None
    while not g.done and g.ply < max_ply:
        if g.kind == CHANCE:
            weights = g.chance_weights()
            kind = chance_kind(g)
            live = [i for i, w in enumerate(weights) if w > 0]
            if kind == "dev":
                live = [i for i in live if i != VP_CARD]  # a VP card could end the game
            if kind == want and len(live) >= 2:
                involved = {g.current_player} | ({victim} if kind == "steal" else set())
                return history, live[:2], involved
            outcome = sample_chance(g, seed)
            history.append(("chance", outcome))
            g.apply_chance(outcome)
        else:
            legal = g.legal_actions()
            t, a = legal[ce.random_below(seed, 7000, g.ply, len(legal))]
            if t == AT.STEAL:
                victim = (g.decider + 1 + a) % 4
            # Buy dev cards whenever possible so draws come early.
            if (AT.BUY_DEV, 0) in legal and want == "dev":
                t, a = AT.BUY_DEV, 0
            history.append(("step", t, a))
            g.step(t, a)
    return None


@pytest.mark.parametrize("want", ["steal", "dev"])
def test_hidden_chance_outcomes_reach_only_the_seats_that_see_them(want):
    """Two games equal up to a steal (or dev draw) that went two ways: every seat that may not
    see the outcome gets byte-identical observations, renders and event descriptions."""
    forks = 0
    for seed in range(1, 7):
        found = _play_until_fork(seed, want)
        if found is None:
            continue
        history, (x, y), involved = found
        g1, g2 = game_with_history(seed, history), game_with_history(seed, history)
        e1, e2 = g1.apply_chance(x), g2.apply_chance(y)
        forks += 1
        assert g1.state_hash() != g2.state_hash()
        hidden_type = ce.EventType.STEAL if want == "steal" else ce.EventType.DEV_BOUGHT
        h1 = [e for e in e1 if e.type == hidden_type]
        h2 = [e for e in e2 if e.type == hidden_type]
        assert len(h1) == len(h2) == 1 and h1[0].a == x and h2[0].a == y
        for o in range(4):
            same_obs = np.array_equal(g1.observation(o), g2.observation(o))
            same_render = g1.render(o) == g2.render(o)
            same_event = h1[0].describe(o) == h2[0].describe(o)
            if o in involved:
                assert not same_obs and not same_render and not same_event, o
            else:
                assert same_obs and same_render and same_event, o
        assert g1.render() != g2.render()
        assert str(h1[0]) != str(h2[0])
    assert forks >= 3


# ---- copy -----------------------------------------------------------------------------------


@pytest.mark.parametrize("auto_chance", [True, False])
def test_copy_is_independent_and_behaves_like_the_original(auto_chance):
    seed = 41
    g = ce.Game(seed, auto_chance=auto_chance)
    for _ in range(200):
        if g.kind == CHANCE:
            g.apply_chance(sample_chance(g, seed))
        else:
            legal = g.legal_actions()
            g.step(*legal[random_index(seed, g.ply, len(legal))])
    c = g.copy()
    assert isinstance(c, ce.Game)
    assert c.state_hash() == g.state_hash() and c.seed == g.seed
    assert (c.kind, c.phase, c.ply, c.turn) == (g.kind, g.phase, g.ply, g.turn)
    if g.kind == PLAYER:
        np.testing.assert_array_equal(c.observation(), g.observation())
        assert c.legal_actions() == g.legal_actions()

    # Driven alike, the copy follows the original exactly (same chance, same auto_chance).
    for _ in range(300):
        if g.done:
            break
        if g.kind == CHANCE:
            outcome = sample_chance(g, seed)
            g.apply_chance(outcome)
            c.apply_chance(outcome)
        else:
            legal = g.legal_actions()
            action = legal[random_index(seed, g.ply, len(legal))]
            g.step(*action)
            c.step(*action)
        assert c.state_hash() == g.state_hash()
        assert c.kind == g.kind

    # Stepping the copy alone leaves the original untouched.
    d = g.copy()
    before = (g.state_hash(), g.ply, g.render())
    for _ in range(50):
        if d.done:
            break
        if d.kind == CHANCE:
            d.apply_chance(sample_chance(d, seed))
        else:
            legal = d.legal_actions()
            d.step(*legal[0])
    assert d.state_hash() != before[0]
    assert (g.state_hash(), g.ply, g.render()) == before


def test_copy_of_a_finished_game_is_finished():
    g = ce.Game(GOLDENS[5][0])
    play_to_end(g, GOLDENS[5][0])
    c = g.copy()
    assert c.done and c.winner == g.winner and c.vp == g.vp and c.state_hash() == g.state_hash()


# ---- VecGames -------------------------------------------------------------------------------


def test_vecgames_shapes_dtypes_and_initial_seeds():
    vg = ce.VecGames(5, first_seed=10)
    assert vg.n == 5
    assert vg.seeds.dtype == np.uint64
    np.testing.assert_array_equal(vg.seeds, np.arange(10, 15, dtype=np.uint64))
    assert ce.VecGames(3).seeds.tolist() == [1, 2, 3]
    obs, mask, dec = vg.observe()
    assert obs.dtype == np.uint8 and obs.shape == (5, ce.OBS_SIZE)
    assert mask.dtype == np.uint8 and mask.shape == (5, ce.MASK_SIZE)
    assert dec.dtype == np.int8 and dec.shape == (5,)
    assert dec.tolist() == [0] * 5  # setup starts with seat 0
    types, args = vg.greedy_actions()
    assert types.dtype == np.uint8 and args.dtype == np.uint8
    assert types.shape == (5,) and args.shape == (5,)
    done, winner, vp, turns = vg.step(types, args)
    assert done.dtype == np.bool_ and done.shape == (5,)
    assert winner.dtype == np.int8 and winner.shape == (5,)
    assert vp.dtype == np.uint8 and vp.shape == (5, 4)
    assert turns.dtype == np.uint16 and turns.shape == (5,)
    assert not done.any() and (winner == -1).all() and not vp.any() and not turns.any()
    for i in range(5):
        gi = vg.game(i)
        assert isinstance(gi, ce.Game) and gi.seed == 10 + i


def _vec_against_shadows(vg, shadows, next_seed, steps, choose, standard=False):
    """Steps vg and its shadow Games together, checking every output. choose(i, game) gives
    slot i's action. Returns (finished results, next_seed)."""
    finished = []
    for step in range(steps):
        obs, mask, dec = vg.observe()
        for i, g in enumerate(shadows):
            assert dec[i] == g.decider
            np.testing.assert_array_equal(obs[i], g.observation())
            np.testing.assert_array_equal(mask[i], g.mask())
        if step % 50 == 0:
            for i, g in enumerate(shadows):
                copy = vg.game(i)
                assert copy.state_hash() == g.state_hash() and copy.seed == g.seed
        actions = [choose(i, g) for i, g in enumerate(shadows)]
        types = np.array([t for t, _ in actions], dtype=np.int64)
        args = np.array([a for _, a in actions], dtype=np.int64)
        done, winner, vp, turns = vg.step(types, args)
        for i, g in enumerate(shadows):
            g.step(*actions[i])
            if g.done:
                assert done[i]
                assert winner[i] == (-1 if g.winner is None else g.winner)
                assert vp[i].tolist() == g.vp
                assert turns[i] == g.turn
                finished.append((g.seed, g.winner, tuple(g.vp), g.turn))
                shadows[i] = ce.Game(next_seed, standard_board=standard)
                next_seed += 1
                # The slot restarted: its next observation is a fresh game in setup.
                fresh = vg.game(i)
                assert (fresh.turn, fresh.ply, fresh.phase) == (0, 0, PH.SETUP)
            else:
                assert not done[i] and winner[i] == -1 and not vp[i].any() and turns[i] == 0
        assert vg.seeds.tolist() == [g.seed for g in shadows]
        if done.any():
            new_obs = vg.observe()[0]
            for i in np.flatnonzero(done):
                glob = new_obs[i, OBS_GLOBAL:]
                assert glob[GLOBAL_PHASE + PH.SETUP] == 1
                assert glob[GLOBAL_TURN] == 0 and glob[GLOBAL_SETUP_STEP] == 0
    return finished, next_seed


def test_vecgames_slots_match_standalone_games_under_random_play():
    n, first = 4, 300
    vg = ce.VecGames(n, first_seed=first)
    shadows = [ce.Game(first + i) for i in range(n)]

    def choose(i, g):
        legal = g.legal_actions()
        return legal[ce.random_below(first + i, 8000, g.ply, len(legal))]

    _vec_against_shadows(vg, shadows, first + n, 700, choose)


def test_vecgames_restarts_in_order_and_reports_finished_games():
    n, first = 8, 1
    vg = ce.VecGames(n, first_seed=first)
    shadows = [ce.Game(first + i) for i in range(n)]
    greedy_types, greedy_args = None, None

    def choose(i, g):
        nonlocal greedy_types, greedy_args
        if i == 0:
            greedy_types, greedy_args = vg.greedy_actions()
        action = g.greedy_action()
        assert (greedy_types[i], greedy_args[i]) == action
        return action

    finished, next_seed = _vec_against_shadows(vg, shadows, first + n, 1300, choose)
    assert len(finished) >= 16
    assert next_seed == first + n + len(finished)
    # Each slot's first game is seed first + i, played as eval's greedy play_game plays it.
    by_seed = {seed: (winner, vp, turns) for seed, winner, vp, turns in finished}
    for seed, standard, winner, vp, turns, _ in GREEDY_GAMES:
        if not standard:
            assert by_seed[seed] == (winner, vp, turns), seed


def test_vecgames_on_the_standard_board():
    vg = ce.VecGames(2, first_seed=7, standard_board=True)
    shadows = [ce.Game(7, standard_board=True), ce.Game(8, standard_board=True)]
    assert vg.game(0).board() == shadows[0].board()
    finished, _ = _vec_against_shadows(vg, shadows, 9, 600,
                                       lambda i, g: g.greedy_action(), standard=True)
    want = next(g for g in GREEDY_GAMES if g[1])
    assert (7, want[2], want[3], want[4]) in finished


def test_vecgames_is_deterministic():
    def run():
        vg = ce.VecGames(5, first_seed=77)
        out = []
        for step in range(500):
            obs, mask, dec = vg.observe()
            types, args = vg.greedy_actions()
            if step % 10 == 0:
                # Some random legal moves too, decoded from the mask.
                for i in range(5):
                    legal = actions_from_mask(mask[i], vg.game(i).phase)
                    types[i], args[i] = legal[ce.random_below(77, i, step, len(legal))]
            result = vg.step(types, args)
            out.append((obs.tobytes(), mask.tobytes(), dec.tobytes(),
                        tuple(r.tobytes() for r in result), vg.seeds.tobytes()))
        return out

    assert run() == run()


def test_vecgames_rejects_an_illegal_action_without_changing_any_game():
    vg = ce.VecGames(6, first_seed=50)
    for _ in range(120):
        vg.step(*vg.greedy_actions())
    before = snapshot(vg)
    types, args = vg.greedy_actions()
    for bad in (0, 3, 5):
        t, a = types.copy(), args.copy()
        t[bad], a[bad] = AT.RESPOND_TRADE, 0  # declared, always masked
        with pytest.raises(ValueError, match=rf"\b{bad}\b"):
            vg.step(t, a)
        assert_same_snapshot(snapshot(vg), before)
    # Two illegal games: the first one is named.
    t, a = types.copy(), args.copy()
    t[2] = t[4] = AT.PROPOSE_TRADE
    a[2] = a[4] = 0
    with pytest.raises(ValueError, match=r"\b2\b") as info:
        vg.step(t, a)
    assert not re.search(r"\b4\b", str(info.value))
    assert_same_snapshot(snapshot(vg), before)
    # Out-of-range values are illegal actions too, even if they would wrap to a legal one.
    a = args.astype(np.int64)
    a[1] += 256
    with pytest.raises(ValueError, match=r"\b1\b"):
        vg.step(types.astype(np.int64), a)
    assert_same_snapshot(snapshot(vg), before)
    # The legal batch still steps afterwards.
    vg.step(types, args)


def test_vecgames_rejects_wrong_shapes_without_changing_any_game():
    vg = ce.VecGames(3, first_seed=60)
    before = snapshot(vg)
    types, args = vg.greedy_actions()
    for bad_types, bad_args in [(types[:2], args[:2]), (types, args[:2]),
                                (types.reshape(3, 1), args), ([0.5, 1.5, 2.5], args)]:
        with pytest.raises((ValueError, TypeError)):
            vg.step(bad_types, bad_args)
    assert_same_snapshot(snapshot(vg), before)
    with pytest.raises((ValueError, IndexError)):
        vg.game(3)


def test_vecgames_accepts_lists_and_integer_dtypes():
    results = []
    for convert in (list, lambda x: np.asarray(x, np.int64), lambda x: np.asarray(x, np.uint8),
                    lambda x: np.asarray(x, np.int32), lambda x: np.asarray(x, np.uint16)):
        vg = ce.VecGames(3, first_seed=90)
        trace = []
        for _ in range(60):
            types, args = vg.greedy_actions()
            out = vg.step(convert([int(t) for t in types]), convert([int(a) for a in args]))
            trace.append((vg.observe()[0].tobytes(), tuple(x.tobytes() for x in out)))
        results.append(trace)
    assert all(r == results[0] for r in results)


def test_vecgames_in_threads_match_sequential_runs():
    """The GIL is released while stepping: separate VecGames in parallel threads, and threads
    sharing one, still give exactly the sequential results."""
    import threading

    def run(first_seed, steps=150):
        vg = ce.VecGames(4, first_seed=first_seed)
        for _ in range(steps):
            vg.step(*vg.greedy_actions())
        return vg.observe()[0].tobytes(), vg.seeds.tobytes()

    want = {seed: run(seed) for seed in (1, 11, 21, 31)}
    got = {}
    threads = [threading.Thread(target=lambda s=seed: got.__setitem__(s, run(s)))
               for seed in want]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert got == want

    shared = ce.VecGames(4, first_seed=1)
    errors = []

    def poke():
        try:
            for _ in range(200):
                obs, mask, dec = shared.observe()
                assert obs.shape == (4, ce.OBS_SIZE)
                shared.game(1).state_hash()
        except Exception as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    readers = [threading.Thread(target=poke) for _ in range(3)]
    for t in readers:
        t.start()
    for _ in range(150):
        shared.step(*shared.greedy_actions())
    for t in readers:
        t.join()
    assert not errors
    assert (shared.observe()[0].tobytes(), shared.seeds.tobytes()) == want[1]


def test_vecgames_observe_arrays_are_snapshots():
    vg = ce.VecGames(2, first_seed=5)
    obs, mask, dec = vg.observe()
    kept = obs.copy(), mask.copy(), dec.copy()
    vg.step(*vg.greedy_actions())
    for x, y in zip((obs, mask, dec), kept):
        np.testing.assert_array_equal(x, y)


def test_two_hundred_greedy_games_through_vecgames():
    n, want = 32, 200
    vg = ce.VecGames(n, first_seed=1000)
    results = []
    steps = 0
    start = time.perf_counter()
    while len(results) < want:
        types, args = vg.greedy_actions()
        done, winner, vp, turns = vg.step(types, args)
        steps += 1
        for i in np.flatnonzero(done):
            results.append((int(winner[i]), vp[i].tolist(), int(turns[i])))
        assert steps < 20000
    elapsed = time.perf_counter() - start
    print(f"\nVecGames greedy: {steps * n / elapsed:,.0f} decisions/s, "
          f"{len(results) / elapsed:,.1f} games/s ({n} games, {steps} steps, {elapsed:.2f} s)")

    wins = Counter(w for w, _, _ in results)
    assert set(wins) <= {-1, 0, 1, 2, 3}
    assert wins[-1] <= want // 20  # the turn cap ends few games
    for seat in range(4):
        assert wins[seat] >= want // 10, wins
    for w, vp, turns in results:
        assert 0 < turns <= ce.TURN_CAP
        if w >= 0:
            assert vp[w] >= ce.VICTORY_POINTS_TO_WIN
        else:
            assert turns == ce.TURN_CAP


def test_vecgames_throughput_with_observations():
    n = 64
    vg = ce.VecGames(n, first_seed=1)
    start = time.perf_counter()
    steps = 300
    for _ in range(steps):
        vg.observe()
        vg.step(*vg.greedy_actions())
    elapsed = time.perf_counter() - start
    print(f"\nVecGames observe+greedy+step: {steps * n / elapsed:,.0f} game-steps/s "
          f"({n} games, {steps} steps, {elapsed:.2f} s)")


# ---- play.py --------------------------------------------------------------------------------


def _play_script():
    env = os.environ.get("CATAN_PY_PLAY")
    path = Path(env) if env else Path(__file__).resolve().parents[1] / "python" / "play.py"
    if not path.exists():
        pytest.skip(f"play.py not found at {path}")
    return path


def _run_play(*args, stdin=""):
    return subprocess.run([sys.executable, str(_play_script()), *args], input=stdin,
                          capture_output=True, text=True, timeout=120)


def test_play_auto_runs_the_greedy_game_of_the_seed():
    """--auto seats four greedy bots: the result is eval's greedy game on that seed."""
    proc = _run_play("--auto", "--seed", "3")
    assert proc.returncode == 0, proc.stderr
    seed, _, winner, vp, turns, _ = GREEDY_GAMES[2]
    assert seed == 3
    result = [line for line in proc.stdout.splitlines() if re.search(r"\bwins?\b", line)]
    assert result, proc.stdout[-2000:]
    assert re.search(rf"\bP{winner}\b", result[-1]) and re.search(rf"\b{turns}\b", result[-1])
    # The final position is printed from the god view.
    g = ce.Game(3)
    while not g.done:
        g.step(*g.greedy_action())
    assert g.render() in proc.stdout


def test_play_takes_scripted_human_moves_then_quits():
    """Seat 0 places its two settlements and roads (choice 0 each time), then quits at its
    first roll."""
    proc = _run_play("--seed", "3", stdin="0\n0\n0\n0\nq\n")
    assert proc.returncode == 0, proc.stderr
    assert "Traceback" not in proc.stderr
    out = proc.stdout
    # The first menu lists the engine's own descriptions of the legal moves.
    g = ce.Game(3)
    assert all(g.describe(t, a) in out for t, a in g.legal_actions())
    # Seat 0's first choice was the first legal settlement spot.
    first = g.legal_actions()[0]
    event = next(e for e in g.step(*first) if e.type == ce.EventType.SETTLEMENT_BUILT)
    assert str(event) in out
    assert not any(re.search(r"\bwins?\b", line) for line in out.splitlines())


def test_play_quits_cleanly_when_input_ends():
    proc = _run_play("--seed", "5", stdin="")
    assert proc.returncode == 0, proc.stderr
    assert "Traceback" not in proc.stderr
