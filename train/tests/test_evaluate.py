"""Tests for catan_rl.evaluate (random bot, Wilson interval, matches), from the contracts."""

import dataclasses
import math
import re

import numpy as np
import pytest
import torch

from catan_rl.engine import ce
from catan_rl.evaluate import MatchResult, play_match, random_actions, wilson
from catan_rl.policy import Agent, PolicyNet

PH = ce.Phase


def gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def make_net(seed: int = 0, **kwargs) -> PolicyNet:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return PolicyNet(**kwargs)


@pytest.fixture(scope="module")
def net() -> PolicyNet:
    return make_net(0)


# ---- random_actions ---------------------------------------------------------------------------

def test_random_actions_legal(positions):
    types, args = random_actions(positions.obs, positions.mask, generator=gen(0))
    for x in (types, args):
        assert type(x) is np.ndarray and x.dtype == np.int64 and x.shape == (len(positions),)
    for i, (t, a) in enumerate(zip(types.tolist(), args.tolist())):
        assert positions.games[i].is_legal(t, a), (
            f"{PH(int(positions.phases[i])).name}: illegal ({ce.ActionType(t).name}, {a})")


def test_random_actions_empty_batch():
    types, args = random_actions(np.zeros((0, ce.OBS_SIZE), np.uint8),
                                 np.zeros((0, ce.MASK_SIZE), np.uint8), generator=gen(0))
    assert types.shape == args.shape == (0,)
    assert types.dtype == args.dtype == np.int64


def test_random_actions_reproducible(positions):
    first = random_actions(positions.obs, positions.mask, generator=gen(5))
    second = random_actions(positions.obs, positions.mask, generator=gen(5))
    assert all(np.array_equal(x, y) for x, y in zip(first, second))


RANDOM_DRAWS = 20_000


@pytest.mark.parametrize("which", ["main_many_types", "yop_pick", "move_robber", "discard"])
def test_random_actions_uniform_type_then_uniform_arg(positions, which):
    """P(type, arg) = 1 / (#legal types) * 1 / (#legal args of the type), checked by frequency
    (5 standard errors) on one position drawn many times."""
    ph, legal = positions.phases, positions.legal
    if which == "main_many_types":
        i = max(positions.where(lambda j: ph[j] == PH.MAIN),
                key=lambda j: (len({t for t, _ in legal[j]}), len(legal[j]), -j))
    else:
        phase = {"yop_pick": PH.YOP_PICK, "move_robber": PH.MOVE_ROBBER,
                 "discard": PH.DISCARD}[which]
        i = positions.where(lambda j: ph[j] == phase and len(legal[j]) >= 2)[0]
    by_type: dict = {}
    for t, a in legal[i]:
        by_type.setdefault(t, []).append(a)
    want = {(t, a): 1 / (len(by_type) * len(args)) for t, args in by_type.items() for a in args}
    obs = np.repeat(positions.obs[i:i + 1], RANDOM_DRAWS, axis=0)
    mask = np.repeat(positions.mask[i:i + 1], RANDOM_DRAWS, axis=0)
    types, args = random_actions(obs, mask, generator=gen(11))
    counts: dict = {}
    for m in zip(types.tolist(), args.tolist()):
        counts[m] = counts.get(m, 0) + 1
    assert set(counts) <= set(want), f"illegal draws: {set(counts) - set(want)}"

    def check(n, p, what):
        se = math.sqrt(p * (1 - p) / RANDOM_DRAWS)
        assert abs(n / RANDOM_DRAWS - p) <= 5 * se, f"{what}: {n / RANDOM_DRAWS:.4f} vs {p:.4f}"

    for move, p in want.items():
        check(counts.get(move, 0), p, f"move {move}")
    for t in by_type:
        check(int((types == t).sum()), 1 / len(by_type), f"type {ce.ActionType(t).name}")


# ---- wilson -----------------------------------------------------------------------------------

def ref_wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """The Wilson score interval, written out independently."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


@pytest.mark.parametrize("k,n,lo,hi", [
    (0, 10, 0.0, 0.27754),
    (5, 10, 0.23659, 0.76341),
    (10, 10, 0.72246, 1.0),
    (50, 100, 0.40383, 0.59617),
    (52, 200, 0.20414, 0.32491),
])
def test_wilson_known_values(k, n, lo, hi):
    got = wilson(k, n)
    assert len(got) == 2
    assert got[0] == pytest.approx(lo, abs=1e-5) and got[1] == pytest.approx(hi, abs=1e-5)


def test_wilson_no_games_is_whole_interval():
    assert tuple(wilson(0, 0)) == (0.0, 1.0)


def test_wilson_matches_formula_and_bounds():
    """Matches the formula for every k <= n <= 40 and several z; the bounds lie in [0, 1]
    exactly (an artefact below 0 would print as "-0.0%") and contain k/n up to rounding."""
    for z in (1.0, 1.96, 2.576):
        for n in range(1, 41):
            for k in range(n + 1):
                lo, hi = wilson(k, n, z)
                rlo, rhi = ref_wilson(k, n, z)
                assert lo == pytest.approx(rlo, abs=1e-9) and hi == pytest.approx(rhi, abs=1e-9)
                assert 0.0 <= lo and hi <= 1.0, (k, n, z, lo, hi)
                assert lo <= k / n + 1e-12 and k / n <= hi + 1e-12, (k, n, z, lo, hi)


def test_wilson_symmetry_and_width():
    for n in (7, 50, 400):
        for k in (0, 1, n // 3, n):
            lo, hi = wilson(k, n)
            mlo, mhi = wilson(n - k, n)
            assert lo == pytest.approx(1 - mhi, abs=1e-12)
            assert hi == pytest.approx(1 - mlo, abs=1e-12)
            wide = wilson(k, n, z=2.576)
            assert wide[0] <= lo + 1e-12 and wide[1] >= hi - 1e-12


# ---- MatchResult ----------------------------------------------------------------------------

LINE = re.compile(
    r"vs (?P<opp>\S+): won (?P<wins>\d+)/(?P<games>\d+) = (?P<rate>\d+\.\d)% "
    r"\[(?P<lo>\d+\.\d)%, (?P<hi>\d+\.\d)%\], (?P<capped>\d+) capped, "
    r"(?P<turns>\d+) turns/game, (?P<sec>\d+\.\d) s")


def check_line(result: MatchResult):
    """str(result) is one line in the documented format, with numbers matching the fields."""
    text = str(result)
    assert "\n" not in text
    m = LINE.fullmatch(text)
    assert m, f"unexpected format: {text!r}"
    lo, hi = wilson(result.wins, result.games)
    assert m["opp"] == result.opponent
    assert int(m["wins"]) == result.wins and int(m["games"]) == result.games
    assert float(m["rate"]) == pytest.approx(100 * result.win_rate, abs=0.05 + 1e-9)
    assert float(m["lo"]) == pytest.approx(100 * lo, abs=0.05 + 1e-9)
    assert float(m["hi"]) == pytest.approx(100 * hi, abs=0.05 + 1e-9)
    assert int(m["capped"]) == result.capped
    assert abs(int(m["turns"]) - result.mean_turns) <= 1
    assert float(m["sec"]) == pytest.approx(result.seconds, abs=0.05 + 1e-9)


def test_match_result_str_exact():
    r = MatchResult(opponent="random", games=200, wins=52, capped=1, mean_turns=366.0,
                    agent_decisions=9000, seconds=12.31)
    assert str(r) == ("vs random: won 52/200 = 26.0% [20.4%, 32.5%], 1 capped, "
                      "366 turns/game, 12.3 s")
    assert r.win_rate == 0.26 and r.ci == wilson(52, 200)


@pytest.mark.parametrize("r", [
    MatchResult("greedy", 400, 104, 2, 366.4, 1, 12.34),
    MatchResult("self", 8, 0, 0, 100.0, 1, 0.04),
    MatchResult("random", 3, 3, 0, 812.6, 1, 125.0),
])
def test_match_result_str_format(r):
    check_line(r)


# ---- play_match: arguments and wiring -------------------------------------------------------

class _Stop(Exception):
    pass


class FirstCallSpy(Agent):
    """Records the first non-empty batch play_match hands the agent, then aborts the match."""

    def act(self, obs, mask):
        if len(obs) == 0:
            return super().act(obs, mask)
        self.first = (np.array(obs, copy=True), np.array(mask, copy=True))
        raise _Stop


class CountingAgent(Agent):
    """An Agent that counts the rows it is asked to decide and checks they are real decisions."""

    rows = 0
    calls = 0

    def act(self, obs, mask):
        assert obs.dtype == np.uint8 and mask.dtype == np.uint8
        assert obs.shape == (len(obs), ce.OBS_SIZE) and mask.shape == (len(obs), ce.MASK_SIZE)
        assert np.all(mask[:, :ce.NUM_ACTION_TYPES].any(axis=1)), "row with no legal type"
        self.rows += len(obs)
        self.calls += 1
        return super().act(obs, mask)


@pytest.mark.parametrize("opponent", ["bogus", "Random", ""])
def test_play_match_rejects_unknown_opponent(net, opponent):
    with pytest.raises(ValueError):
        play_match(Agent(net, generator=gen(0)), opponent, games=4, generator=gen(0))


@pytest.mark.parametrize("games", [0, -1])
def test_play_match_rejects_no_games(net, games):
    with pytest.raises(ValueError):
        play_match(Agent(net, generator=gen(0)), "random", games=games, generator=gen(0))


@pytest.mark.parametrize("opponent,standard_board", [
    ("random", False), ("random", True), ("greedy", False), ("self", False)])
def test_play_match_slots_seeds_and_seats(opponent, standard_board):
    """Game i runs on seed first_seed + i with the agent in seat i % 4, and agent.act sees just
    the rows it decides. At the first decision every game waits on its seat-0 player, so the
    agent's first batch is exactly the opening observation/mask of slots 0, 4, 8, ... (every
    slot in "self")."""
    games, first_seed = 9, 11
    spy = FirstCallSpy(make_net(1, hidden=8, layers=1), generator=gen(0))
    with pytest.raises(_Stop):
        play_match(spy, opponent, games=games, first_seed=first_seed,
                   standard_board=standard_board, generator=gen(0))
    obs, mask = spy.first
    starts = [ce.Game(first_seed + i, standard_board=standard_board) for i in range(games)]
    mine = [g for i, g in enumerate(starts) if opponent == "self" or g.decider == i % 4]
    assert obs.dtype == np.uint8 and mask.dtype == np.uint8
    assert obs.shape == (len(mine), ce.OBS_SIZE) and mask.shape == (len(mine), ce.MASK_SIZE)
    want = sorted(g.observation().tobytes() + g.mask().tobytes() for g in mine)
    got = sorted(o.tobytes() + m.tobytes() for o, m in zip(obs, mask))
    assert got == want


def check_consistent(result: MatchResult, opponent: str, games: int):
    assert result.opponent == opponent and result.games == games
    assert 0 <= result.wins <= games
    assert 0 <= result.capped <= games - result.wins
    assert 0 < result.mean_turns <= ce.TURN_CAP
    seats = 4 if opponent == "self" else 1
    # Setup alone is 2 settlements + 2 roads per seat.
    assert result.agent_decisions >= 4 * seats * games
    assert result.seconds > 0
    check_line(result)


def test_play_match_is_deterministic(net):
    """Same arguments, agent and generator seeds -> the same MatchResult (bar wall time)."""
    results = []
    for _ in range(2):
        agent = CountingAgent(net, generator=gen(3))
        r = play_match(agent, "random", games=6, first_seed=21, generator=gen(4))
        check_consistent(r, "random", 6)
        assert r.agent_decisions <= agent.rows
        results.append((dataclasses.replace(r, seconds=0.0), agent.rows, agent.calls))
    assert results[0] == results[1]


# ---- play_match: strength of an untrained agent -------------------------------------------

@pytest.fixture(scope="module")
def random_match(net) -> MatchResult:
    return play_match(Agent(net, generator=gen(1)), "random", games=200, first_seed=1,
                      generator=gen(2))


def test_untrained_agent_vs_random_is_a_coin_among_four(random_match):
    """An untrained net plays like the random bot, so it wins about 1 game in 4 (200 games:
    3 standard errors either side of 25%)."""
    check_consistent(random_match, "random", 200)
    assert 0.15 <= random_match.win_rate <= 0.35, str(random_match)


def test_untrained_agent_loses_to_greedy(net):
    r = play_match(Agent(net, generator=gen(5)), "greedy", games=100, first_seed=1001,
                   generator=gen(6))
    check_consistent(r, "greedy", 100)
    assert r.win_rate < 0.10, str(r)


def test_self_play_wins_about_a_quarter(net):
    """Every seat is the agent, so by symmetry seat i % 4 wins about 25%."""
    agent = CountingAgent(net, generator=gen(7))
    r = play_match(agent, "self", games=100, first_seed=2001, generator=gen(8))
    check_consistent(r, "self", 100)
    assert r.agent_decisions <= agent.rows
    assert 0.10 <= r.win_rate <= 0.40, str(r)
