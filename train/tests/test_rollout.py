"""Tests for catan_rl.rollout (the random bot, play, Episodes), written from the contracts."""

import dataclasses
import zlib

import numpy as np
import pytest
import torch

from catan_rl import actions, evaluate, rollout
from catan_rl.engine import ce
from catan_rl.evaluate import play_match
from catan_rl.policy import Agent, PolicyNet
from catan_rl.rollout import Episodes, play, random_actions

AT = ce.ActionType
MOVE_FIELDS = ("obs", "mask", "types", "args", "log_prob", "game")
GAME_FIELDS = ("seat", "won", "capped", "vp", "turns")


def gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def make_net(seed: int = 0, **kwargs) -> PolicyNet:
    """A PolicyNet built under a fixed torch seed, without touching the global RNG state."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return PolicyNet(**kwargs)


# ---- agents -----------------------------------------------------------------------------------

class Spy(Agent):
    """An Agent that logs every call play makes: copies of what it was shown and what it
    returned. decide() is the actual policy (the network, unless a subclass overrides it)."""

    def __init__(self, net: PolicyNet | None = None, generator: torch.Generator | None = None):
        super().__init__(net if net is not None else make_net(0, hidden=8, layers=1),
                         generator=generator)
        self.calls: list[tuple[np.ndarray, ...]] = []

    def decide(self, obs, mask):
        return super().act(obs, mask)

    def act(self, obs, mask):
        assert obs.dtype == np.uint8 and mask.dtype == np.uint8
        assert obs.shape == (len(obs), ce.OBS_SIZE) and mask.shape == (len(obs), ce.MASK_SIZE)
        # Every row handed to the agent is a real decision: some type is legal.
        assert np.all(mask[:, :ce.NUM_ACTION_TYPES].any(axis=1)), "row with no legal type"
        out = self.decide(obs, mask)
        if len(obs):
            self.calls.append((np.array(obs, copy=True), np.array(mask, copy=True),
                               *(np.array(x, copy=True) for x in out)))
        return out

    def stacked(self) -> dict[str, np.ndarray]:
        """Every logged row, in call order."""
        names = ("obs", "mask", "types", "args", "log_prob")
        if not self.calls:
            return {"obs": np.zeros((0, ce.OBS_SIZE), np.uint8),
                    "mask": np.zeros((0, ce.MASK_SIZE), np.uint8),
                    "types": np.zeros(0, np.int64), "args": np.zeros(0, np.int64),
                    "log_prob": np.zeros(0, np.float32)}
        return {n: np.concatenate([c[k] for c in self.calls]) for k, n in enumerate(names)}


# Build first, then buy, then roll / handle forced phases, then end the turn; never trade
# unless forced. Deterministic, so it plays the same game wherever it sits.
_PRIORITY = [AT.BUILD_CITY, AT.BUILD_SETTLEMENT, AT.BUY_DEV, AT.BUILD_ROAD, AT.ROLL,
             AT.MOVE_ROBBER, AT.STEAL, AT.DISCARD, AT.RESPOND_TRADE, AT.PLAY_DEV, AT.END_TURN,
             AT.BANK_EXCHANGE, AT.PROPOSE_TRADE]


def stand_in_move(obs: np.ndarray, mask: np.ndarray) -> tuple[int, int, float]:
    """A legal (type, arg, fake log-prob), a pure function of the observation and mask bytes:
    the first legal type in _PRIORITY, then a legal argument picked by a hash of the bytes."""
    h = zlib.crc32(obs.tobytes() + mask.tobytes())
    t = next(int(x) for x in _PRIORITY if mask[ce.MASK_SECTIONS["type"][0] + int(x)])
    section = int(actions.arg_sections(torch.tensor([t]),
                                       actions.phase_of(torch.as_tensor(obs[None])))[0])
    if section == 0:
        return t, 0, -(h % 997) / 100
    start, size = ce.MASK_SECTIONS[actions.SECTIONS[section]]
    legal = np.flatnonzero(mask[start:start + size])
    return t, int(legal[h % len(legal)]), -(h % 997) / 100


class StandIn(Spy):
    """A deterministic agent (stand_in_move on every row), also logging its calls."""

    def decide(self, obs, mask):
        moves = [stand_in_move(o, m) for o, m in zip(obs, mask)]
        return (np.array([m[0] for m in moves], np.int64),
                np.array([m[1] for m in moves], np.int64),
                np.array([m[2] for m in moves], np.float32))


# ---- the runs most tests share ----------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Setup:
    games: int
    first_seed: int
    generator_seed: int
    stand_in: bool  # StandIn, else a Spy around a real (untrained, sampling) network

    def agent(self) -> Spy:
        if self.stand_in:
            return StandIn()
        return Spy(make_net(1, hidden=32), generator=gen(11))


# Games against greedy are short (~75 turns), so the real network plays them. Games against
# random bots (and self-play) are long, so the cheap deterministic StandIn plays those.
SETUPS = {
    "greedy": Setup(games=8, first_seed=101, generator_seed=12, stand_in=False),
    "random": Setup(games=4, first_seed=201, generator_seed=13, stand_in=True),
    "self": Setup(games=4, first_seed=301, generator_seed=14, stand_in=True),
}


def run_play(opponent: str, record: bool, **overrides) -> tuple[Episodes, Spy]:
    setup = SETUPS[opponent]
    agent = setup.agent()
    kwargs = dict(games=setup.games, first_seed=setup.first_seed,
                  generator=gen(setup.generator_seed), record=record)
    kwargs.update(overrides)
    return play(agent, opponent, **kwargs), agent


@pytest.fixture(scope="module")
def runs():
    """runs(opponent, record) -> (Episodes, the agent that played), computed once."""
    cache: dict = {}

    def get(opponent: str, record: bool):
        if (opponent, record) not in cache:
            cache[opponent, record] = run_play(opponent, record)
        return cache[opponent, record]

    return get


def legal_by_mask(obs, mask, types, args) -> list[str]:
    """Rows whose (type, arg) is not legal by the mask: the type's byte in the "type" section,
    and the argument's byte in the section actions.arg_sections assigns it in that phase (an
    argument-less type must have arg 0)."""
    bad = []
    type_start = ce.MASK_SECTIONS["type"][0]
    sections = actions.arg_sections(torch.as_tensor(types),
                                    actions.phase_of(torch.as_tensor(obs))).numpy()
    for m, (t, a, s) in enumerate(zip(types.tolist(), args.tolist(), sections.tolist())):
        if not (0 <= t < ce.NUM_ACTION_TYPES and mask[m, type_start + t]):
            bad.append(f"row {m}: illegal type {t}")
        elif s == 0:
            if a != 0:
                bad.append(f"row {m}: {AT(t).name} takes no argument, got {a}")
        else:
            start, size = ce.MASK_SECTIONS[actions.SECTIONS[s]]
            if not (0 <= a < size and mask[m, start + a]):
                bad.append(f"row {m}: {AT(t).name} with illegal {actions.SECTIONS[s]} arg {a}")
    return bad


def assert_same_episodes(a: Episodes, b: Episodes):
    for f in dataclasses.fields(Episodes):
        if f.name == "seconds":
            continue
        x, y = getattr(a, f.name), getattr(b, f.name)
        if isinstance(x, np.ndarray):
            assert x.dtype == y.dtype and np.array_equal(x, y), f.name
        else:
            assert x == y, f.name


# ---- random_actions ---------------------------------------------------------------------------

def test_random_actions_is_reexported_by_evaluate():
    assert evaluate.random_actions is rollout.random_actions


def test_random_actions_behaves_as_before(positions):
    """The moved function: legal moves, int64, reproducible from the generator, and the same
    draws whichever module it is called through."""
    types, args = random_actions(positions.obs, positions.mask, gen(5))
    assert types.dtype == args.dtype == np.int64 and types.shape == args.shape == (len(positions),)
    assert legal_by_mask(positions.obs, positions.mask, types, args) == []
    for i, (t, a) in enumerate(zip(types.tolist(), args.tolist())):
        assert positions.games[i].is_legal(t, a)
    again = evaluate.random_actions(positions.obs, positions.mask, gen(5))
    assert np.array_equal(types, again[0]) and np.array_equal(args, again[1])
    empty = random_actions(np.zeros((0, ce.OBS_SIZE), np.uint8),
                           np.zeros((0, ce.MASK_SIZE), np.uint8), gen(0))
    assert empty[0].shape == empty[1].shape == (0,)


# ---- play: arguments ----------------------------------------------------------------------------

@pytest.mark.parametrize("opponent", ["bogus", "Random", "", "GREEDY"])
def test_play_rejects_unknown_opponent(opponent):
    with pytest.raises(ValueError):
        play(StandIn(), opponent, games=4, generator=gen(0))


@pytest.mark.parametrize("games", [0, -3])
def test_play_rejects_no_games(games):
    with pytest.raises(ValueError):
        play(StandIn(), "greedy", games=games, generator=gen(0))


def test_play_rejects_recording_self_play():
    agent = StandIn()
    with pytest.raises(ValueError):
        play(agent, "self", games=4, generator=gen(0), record=True)


def test_play_single_game():
    ep, agent = run_play("greedy", True, games=1, first_seed=77)
    assert ep.games == 1 and ep.seat.tolist() == [0]
    assert ep.moves == ep.decisions > 0 and set(ep.game.tolist()) == {0}


# ---- play: the Episodes it returns -------------------------------------------------------------

@pytest.mark.parametrize("opponent,record", [
    ("greedy", True), ("greedy", False), ("random", True), ("random", False), ("self", False)])
def test_episodes_fields(runs, opponent, record):
    ep, _ = runs(opponent, record)
    G = SETUPS[opponent].games
    M = ep.moves
    if not record:
        assert M == 0, "without record=True the move arrays have 0 rows"
    else:
        assert M > 0
    assert ep.games == G
    for name, shape, dtype in [
            ("obs", (M, ce.OBS_SIZE), np.uint8), ("mask", (M, ce.MASK_SIZE), np.uint8),
            ("types", (M,), np.int64), ("args", (M,), np.int64),
            ("log_prob", (M,), np.float32), ("game", (M,), np.int64),
            ("seat", (G,), np.int64), ("won", (G,), np.bool_), ("capped", (G,), np.bool_),
            ("vp", (G,), np.int64), ("turns", (G,), np.int64)]:
        x = getattr(ep, name)
        assert type(x) is np.ndarray, name
        assert x.shape == shape and x.dtype == dtype, (name, x.shape, x.dtype)
    assert isinstance(ep.decisions, (int, np.integer)) and not isinstance(ep.decisions, bool)
    assert isinstance(ep.seconds, float) and ep.seconds > 0
    if record:
        assert ep.game.min() >= 0 and ep.game.max() < G


@pytest.mark.parametrize("opponent,record", [
    ("greedy", True), ("greedy", False), ("random", True), ("self", False)])
def test_per_game_fields(runs, opponent, record):
    """seat i % 4; a won game has >= 10 VP and was not capped; a capped game was not won;
    every game took some turns, and at most the cap."""
    ep, _ = runs(opponent, record)
    assert np.array_equal(ep.seat, np.arange(ep.games) % ce.NUM_PLAYERS)
    assert np.all(ep.vp[ep.won] >= ce.VICTORY_POINTS_TO_WIN)
    assert not np.any(ep.won & ep.capped)
    assert np.all(ep.vp >= 0)
    assert np.all(ep.turns > 0) and np.all(ep.turns <= ce.TURN_CAP + 1)


def test_self_play_has_winners(runs):
    """With the agent in every seat a game is won by one of its seats, or capped; the stand-in
    self-play games here do produce wins for the scored seat (so `won` is wired up)."""
    ep, _ = runs("self", False)
    assert ep.won.any()


# ---- play: recording ----------------------------------------------------------------------------

@pytest.mark.parametrize("opponent", ["greedy", "random"])
def test_recorded_moves_are_exactly_the_agents_calls(runs, opponent):
    """The recorded rows are every row the agent was asked to decide, in call order, with what
    it answered; each call decides each slot at most once, so its rows have distinct games."""
    ep, agent = runs(opponent, True)
    calls = agent.stacked()
    assert ep.moves == len(calls["types"]) == ep.decisions
    for name in ("obs", "mask", "types", "args"):
        assert np.array_equal(getattr(ep, name), calls[name]), name
    np.testing.assert_allclose(ep.log_prob, calls["log_prob"], rtol=1e-6, atol=0)
    start = 0
    for call in agent.calls:
        n = len(call[0])
        assert n <= ep.games
        games = ep.game[start:start + n]
        assert len(set(games.tolist())) == n, f"call with repeated slots: {games}"
        start += n


@pytest.mark.parametrize("opponent", ["greedy", "random"])
def test_recorded_moves_are_legal(runs, opponent):
    ep, _ = runs(opponent, True)
    assert legal_by_mask(ep.obs, ep.mask, ep.types, ep.args) == []


def test_recorded_log_prob_is_the_networks(runs):
    """log_prob is the log-probability of the move under the network that played it."""
    ep, agent = runs("greedy", True)
    obs = torch.as_tensor(ep.obs)
    with torch.no_grad():
        lp, _ = actions.log_prob_entropy(agent.net(obs), torch.as_tensor(ep.mask),
                                         actions.phase_of(obs), torch.as_tensor(ep.types),
                                         torch.as_tensor(ep.args))
    np.testing.assert_allclose(ep.log_prob, lp.numpy(), rtol=1e-5, atol=1e-5)
    assert np.all(ep.log_prob <= 1e-6)


def game_outcome(g, seat: int) -> tuple[bool, bool, int, int]:
    assert g.done
    return g.winner == seat, g.winner is None, int(g.vp[seat]), int(g.turn)


def episode_outcome(ep: Episodes, i: int) -> tuple[bool, bool, int, int]:
    return bool(ep.won[i]), bool(ep.capped[i]), int(ep.vp[i]), int(ep.turns[i])


def test_recorded_moves_replay_each_slot(runs):
    """Slot i is a game on seed first_seed + i with the agent in seat i % 4. Replaying the
    rows with game == i in order (greedy_action for the other seats) through ce.Game must
    meet exactly those observations and masks at the agent's turns, use up every row, and end
    as won/capped/vp/turns say."""
    ep, _ = runs("greedy", True)
    first_seed = SETUPS["greedy"].first_seed
    for i in range(ep.games):
        rows = np.flatnonzero(ep.game == i)
        seat = i % ce.NUM_PLAYERS
        g = ce.Game(first_seed + i)
        k = 0
        while not g.done:
            if g.decider == seat:
                assert k < len(rows), f"slot {i}: the agent moved more often than recorded"
                m = rows[k]
                assert np.array_equal(ep.obs[m], g.observation()), f"slot {i}, move {k}: obs"
                assert np.array_equal(ep.mask[m], g.mask()), f"slot {i}, move {k}: mask"
                g.step(int(ep.types[m]), int(ep.args[m]))
                k += 1
            else:
                g.step(*g.greedy_action())
        assert k == len(rows), f"slot {i}: {len(rows) - k} recorded moves after the game ended"
        assert episode_outcome(ep, i) == game_outcome(g, seat), f"slot {i}"


@pytest.mark.parametrize("opponent", ["greedy", "random"])
def test_record_does_not_change_the_games(runs, opponent):
    """record=True and record=False with the same seeds play the same games."""
    rec, _ = runs(opponent, True)
    plain, agent = runs(opponent, False)
    for name in GAME_FIELDS:
        assert np.array_equal(getattr(rec, name), getattr(plain, name)), name
    assert rec.decisions == plain.decisions == rec.moves
    # Without recording the agent still makes (and is counted for) the same moves.
    assert len(agent.stacked()["types"]) == plain.decisions
    for name in MOVE_FIELDS:
        assert len(getattr(plain, name)) == 0, name


# ---- play: against a replay with a deterministic stand-in -------------------------------------

@pytest.mark.parametrize("standard_board", [False, True])
def test_greedy_games_match_a_stand_in_replay(standard_board):
    """The stand-in (a pure function of the observation) played through ce.Game(first_seed + i)
    in seat i % 4, greedy_action() for the other seats, reproduces every slot's result and the
    total number of agent decisions."""
    games, first_seed = 6, 401
    ep = play(StandIn(), "greedy", games=games, first_seed=first_seed,
              standard_board=standard_board, generator=gen(3))
    decisions = 0
    for i in range(games):
        seat = i % ce.NUM_PLAYERS
        g = ce.Game(first_seed + i, standard_board=standard_board)
        while not g.done:
            if g.decider == seat:
                t, a, _ = stand_in_move(g.observation(), g.mask())
                decisions += 1
            else:
                t, a = g.greedy_action()
            g.step(t, a)
        assert episode_outcome(ep, i) == game_outcome(g, seat), f"slot {i}"
    assert ep.decisions == decisions


def test_self_play_counts_each_slots_first_game_only(runs):
    """With the stand-in in every seat, slot i's first game is fully determined: replaying it
    through ce.Game(first_seed + i) gives won (seat i % 4 won), capped, vp and turns. These
    games are long and uneven, so slots that finish early restart and can finish again before
    the batch ends; only the first game may count."""
    ep, _ = runs("self", False)
    setup = SETUPS["self"]
    decisions = 0
    for i in range(ep.games):
        g = ce.Game(setup.first_seed + i)
        while not g.done:
            t, a, _ = stand_in_move(g.observation(), g.mask())
            g.step(t, a)
            decisions += 1
        assert episode_outcome(ep, i) == game_outcome(g, i % ce.NUM_PLAYERS), f"slot {i}"
    assert ep.decisions == decisions


# ---- play vs play_match -------------------------------------------------------------------------

@pytest.mark.parametrize("opponent", ["random", "greedy", "self"])
def test_play_match_reports_plays_counts(runs, opponent):
    ep, _ = runs(opponent, False)
    setup = SETUPS[opponent]
    agent = setup.agent()
    r = play_match(agent, opponent, games=setup.games, first_seed=setup.first_seed,
                   generator=gen(setup.generator_seed))
    assert r.opponent == opponent and r.games == ep.games
    assert r.wins == int(ep.won.sum())
    assert r.capped == int(ep.capped.sum())
    assert r.mean_turns == pytest.approx(float(ep.turns.mean()), rel=1e-12)
    assert r.agent_decisions == ep.decisions
    # In "self" every seat of every counted game is the agent.
    assert len(agent.stacked()["types"]) == r.agent_decisions


# ---- determinism ---------------------------------------------------------------------------------

@pytest.mark.parametrize("opponent", ["greedy", "random"])
def test_play_is_deterministic(runs, opponent):
    ep, _ = runs(opponent, True)
    again, _ = run_play(opponent, True)
    assert_same_episodes(ep, again)


def test_generator_drives_the_random_bot(runs):
    """A different `generator` seed changes the random bots' play (so the games)."""
    ep, _ = runs("random", False)
    other, _ = run_play("random", False, generator=gen(999))
    assert any(not np.array_equal(getattr(ep, n), getattr(other, n))
               for n in ("won", "vp", "turns")) or ep.decisions != other.decisions


def test_agent_generator_drives_the_agent(runs):
    ep, _ = runs("greedy", True)
    agent = Spy(make_net(1, hidden=32), generator=gen(12345))
    other = play(agent, "greedy", games=SETUPS["greedy"].games,
                 first_seed=SETUPS["greedy"].first_seed,
                 generator=gen(SETUPS["greedy"].generator_seed), record=True)
    n = min(ep.moves, other.moves)
    assert not np.array_equal(ep.types[:n], other.types[:n])
