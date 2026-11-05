"""Tests for catan_rl.trainer (the loop every algorithm shares), written from the contracts.

test_reinforce.py already runs the whole loop through reinforce.Trainer (resuming, crashes,
evaluation, atomic saves). This file tests the rest: Config.check and its validators,
minibatches, and the base Trainer itself, through Probe, a test-only algorithm whose learn()
records how it was called.
"""

import copy
import dataclasses
import inspect
import json
import math
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from catan_rl import actions, actor_critic, reinforce
from catan_rl import trainer as base
from catan_rl.engine import ce
from catan_rl.policy import PolicyNet
from catan_rl.rollout import Episodes, play
from catan_rl.trainer import Config, check_choice, check_int, check_number, minibatches

STEP_KEYS = {"update", "games", "moves", "win_rate", "mean_vp", "mean_turns", "capped", "reward",
             "loss", "entropy", "grad_norm", "rollout_seconds", "update_seconds"}


def gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


# ---- Config.check --------------------------------------------------------------------------------

POSITIVE_INTS = ("updates", "games_per_update", "minibatch", "hidden", "layers", "eval_every",
                 "eval_games")
SEEDS = {"seed": 2**32 - 1, "train_first_seed": 2**62}  # the largest each may be
POSITIVE_NUMBERS = ("lr", "max_grad_norm")
NONNEGATIVE_NUMBERS = ("entropy_coef",)
CHOICES = {"reward": base.REWARDS, "opponent": base.OPPONENTS}

NOT_INTS = [1.0, 2.5, True, False, "3", None, math.nan, math.inf]
NOT_NUMBERS = [math.nan, math.inf, -math.inf, True, False, "0.1", None, [0.1]]

BAD = (
    [(name, v) for name in POSITIVE_INTS for v in [0, -1, -100, *NOT_INTS]]
    + [(name, v) for name, top in SEEDS.items() for v in [-1, top + 1, 2**64, *NOT_INTS]]
    + [(name, v) for name in POSITIVE_NUMBERS for v in [0, 0.0, -1e-3, -1, *NOT_NUMBERS]]
    + [(name, v) for name in NONNEGATIVE_NUMBERS for v in [-1e-9, -1, *NOT_NUMBERS]]
    + [("reward", v) for v in ["", "Win", "points", "loss", None, 1]]
    + [("opponent", v) for v in ["self", "Random", "bogus", "", None]]
)
GOOD = (
    [(name, v) for name in POSITIVE_INTS for v in [1, 3, 10_000]]
    + [(name, v) for name, top in SEEDS.items() for v in [0, 1, 123_456_789, top]]
    + [(name, v) for name in POSITIVE_NUMBERS for v in [1e-12, 1e-3, 1, 2.5]]
    + [(name, v) for name in NONNEGATIVE_NUMBERS for v in [0, 0.0, 0.01, 3]]
    + [(name, v) for name, choices in CHOICES.items() for v in choices]
)


def test_every_base_field_is_covered():
    """The lists above name every field of the base Config (a new field needs its checks)."""
    covered = {*POSITIVE_INTS, *SEEDS, *POSITIVE_NUMBERS, *NONNEGATIVE_NUMBERS, *CHOICES}
    assert covered == {f.name for f in dataclasses.fields(Config)}


def test_the_defaults_pass():
    Config().check()


@pytest.mark.parametrize("name,value", BAD, ids=[f"{n}={v!r}" for n, v in BAD])
def test_check_rejects(name, value):
    with pytest.raises(ValueError) as exc:
        Config(**{name: value}).check()
    assert name in str(exc.value), f"the error should name {name}: {exc.value}"


@pytest.mark.parametrize("name,value", GOOD, ids=[f"{n}={v!r}" for n, v in GOOD])
def test_check_accepts(name, value):
    Config(**{name: value}).check()


# ---- check_int, check_number, check_choice -------------------------------------------------------

@dataclass(frozen=True)
class XConfig(Config):
    x: object = 0


def raises_naming_x(fn, *args, **kwargs):
    with pytest.raises(ValueError) as exc:
        fn(*args, **kwargs)
    assert "x" in str(exc.value)


def test_check_int():
    for value in (0, 5, -3, 10**20):
        check_int(XConfig(x=value), "x")
    check_int(XConfig(x=5), "x", low=5)
    check_int(XConfig(x=-2), "x", low=-2)
    raises_naming_x(check_int, XConfig(x=4), "x", low=5)
    for value in (5.0, 0.5, True, False, "5", None, math.nan, np.float64(2.0)):
        raises_naming_x(check_int, XConfig(x=value), "x")


def test_check_number():
    for value in (0, 0.0, 1e-300, 7, 2.5, 1e300):
        check_number(XConfig(x=value), "x", low=0)
    check_number(XConfig(x=1), "x", low=0, high=1)  # both bounds inclusive by default
    check_number(XConfig(x=0.0), "x", low=0, high=1)
    check_number(XConfig(x=1e-9), "x", low=0, low_ok=False)
    raises_naming_x(check_number, XConfig(x=0), "x", low=0, low_ok=False)
    raises_naming_x(check_number, XConfig(x=-1e-12), "x", low=0)
    raises_naming_x(check_number, XConfig(x=1 + 1e-9), "x", low=0, high=1)
    for value in (math.nan, math.inf, -math.inf, True, False, "1", None, [1.0]):
        raises_naming_x(check_number, XConfig(x=value), "x", low=-math.inf)


def test_check_choice():
    for value in ("a", "b"):
        check_choice(XConfig(x=value), "x", ("a", "b"))
    for value in ("A", "", "c", None, 1, ("a",)):
        raises_naming_x(check_choice, XConfig(x=value), "x", ("a", "b"))


# ---- minibatches ---------------------------------------------------------------------------------

def make_episodes(won, vp, game=(), *, rows=None) -> Episodes:
    """Episodes with the given per-game results and M = len(game) moves. rows = (obs, mask,
    types, args) gives the moves real content; otherwise they are zeros."""
    won = np.asarray(won, dtype=bool)
    G = len(won)
    game = np.asarray(game, dtype=np.int64)
    M = len(game)
    if rows is None:
        rows = (np.zeros((M, ce.OBS_SIZE), np.uint8), np.zeros((M, ce.MASK_SIZE), np.uint8),
                np.zeros(M, np.int64), np.zeros(M, np.int64))
    obs, mask, types, args = rows
    return Episodes(
        obs=obs, mask=mask, types=np.asarray(types, np.int64), args=np.asarray(args, np.int64),
        log_prob=np.zeros(M, np.float32), game=game,
        seat=np.arange(G, dtype=np.int64) % 4, won=won, capped=np.zeros(G, bool),
        vp=np.asarray(vp, dtype=np.int64), turns=np.full(G, 80, np.int64),
        decisions=M, seconds=0.1)


@pytest.fixture(scope="module")
def moves(positions):
    """23 moves (a random legal one each) at fixture positions of many phases, over 3 games."""
    rng = np.random.default_rng(0)
    idx = rng.choice(len(positions), 23, replace=False)
    picked = [positions.legal[i][rng.integers(len(positions.legal[i]))] for i in idx]
    ep = make_episodes([True, False, False], [10, 4, 6], rng.integers(3, size=23), rows=(
        positions.obs[idx], positions.mask[idx], [m[0] for m in picked], [m[1] for m in picked]))
    return SimpleNamespace(ep=ep, phases=positions.phases[idx])


@pytest.mark.parametrize("size", [1, 5, 22, 23, 24, 1000])
def test_minibatches_cover_every_move_in_order(moves, size):
    ep = moves.ep
    batches = list(minibatches(ep, size, torch.device("cpu")))
    rows = []
    for b in batches:
        r = list(range(ep.moves)[b.rows])
        assert 1 <= len(b) == len(r) <= size
        rows += r
    assert rows == list(range(ep.moves)), "every move once, in order"


@pytest.mark.parametrize("size", [1, 7, 23, 64])
def test_minibatch_contents(moves, size):
    """Each Minibatch holds its rows of the episodes, as tensors of the documented dtypes on
    the requested device, with the phase read from the observation."""
    ep = moves.ep
    device = torch.device("cpu")
    for b in minibatches(ep, size, device):
        for name, dtype in (("obs", torch.uint8), ("mask", torch.uint8), ("types", torch.int64),
                            ("args", torch.int64), ("phases", torch.int64)):
            t = getattr(b, name)
            assert isinstance(t, torch.Tensor) and t.dtype == dtype, name
            assert t.device == device, name
        assert b.obs.shape == (len(b), ce.OBS_SIZE) and b.mask.shape == (len(b), ce.MASK_SIZE)
        assert b.types.shape == b.args.shape == b.phases.shape == (len(b),)
        for name in ("obs", "mask", "types", "args"):
            assert np.array_equal(getattr(b, name).numpy(), getattr(ep, name)[b.rows]), name
        assert torch.equal(b.phases, actions.phase_of(b.obs))
        assert np.array_equal(b.phases.numpy(), moves.phases[b.rows])


def test_minibatch_log_prob_entropy_is_actions(moves):
    ep = moves.ep
    logits = torch.randn(ep.moves, ce.MASK_SIZE, generator=gen(1))
    for b in minibatches(ep, 9, torch.device("cpu")):
        got = b.log_prob_entropy(logits[b.rows])
        want = actions.log_prob_entropy(logits[b.rows], b.mask, b.phases, b.types, b.args)
        for x, y in zip(got, want):
            assert torch.equal(x, y)


def test_minibatches_of_no_moves_is_nothing():
    ep = make_episodes([True, False], [10, 2])
    for size in (1, 100):
        assert list(minibatches(ep, size, torch.device("cpu"))) == []


@pytest.mark.parametrize("size", [0, -1])
def test_minibatches_rejects_a_size_below_one(moves, size):
    # Raised by the call itself, before anything is iterated.
    with pytest.raises(ValueError):
        minibatches(moves.ep, size, torch.device("cpu"))
    with pytest.raises(ValueError):
        minibatches(make_episodes([True], [10]), size, torch.device("cpu"))


# ---- the base Trainer, through Probe -------------------------------------------------------------

def test_the_base_trainer_is_abstract(tmp_path):
    with pytest.raises(TypeError):
        base.Trainer(Config(), tmp_path)

    class NoLearn(base.Trainer):
        algorithm = "no-learn"
        config_type = Config

        def make_net(self):
            return PolicyNet(16, 1)

    with pytest.raises(TypeError):
        NoLearn(Config(), tmp_path)


@dataclass(frozen=True)
class ProbeConfig(Config):
    pass


class Probe(base.Trainer):
    """The smallest algorithm: imitates its own moves (loss -mean log-prob). learn() records
    what it was given, whether the gradients were zero when it was called, and the gradient it
    left; log_line marks its lines."""

    algorithm = "test"
    config_type = ProbeConfig

    def __init__(self, *args, **kwargs):
        self.calls: list[SimpleNamespace] = []
        self.nets_made = 0
        super().__init__(*args, **kwargs)

    def make_net(self) -> PolicyNet:
        self.nets_made += 1
        return PolicyNet(self.config.hidden, self.config.layers)

    def learn(self, episodes: Episodes) -> dict[str, float]:
        zeroed = all(p.grad is None or torch.count_nonzero(p.grad) == 0
                     for p in self.net.parameters())
        entropy = 0.0
        for batch in minibatches(episodes, self.config.minibatch, self.device):
            log_prob, h = batch.log_prob_entropy(self.net(batch.obs))
            (-log_prob.sum() / episodes.moves).backward()
            entropy += h.detach().sum().item() / episodes.moves
        grads = {n: p.grad.clone() for n, p in self.net.named_parameters()}
        norm = float(torch.linalg.vector_norm(torch.cat([g.flatten() for g in grads.values()])))
        self.calls.append(SimpleNamespace(episodes=episodes, zeroed=zeroed, grads=grads,
                                          norm=norm))
        return {"reward": 0.25, "loss": 1.5, "entropy": entropy, "probe": 42.0}

    def log_line(self, metrics: dict) -> str:
        return f"probe update {metrics['update']}: {metrics['probe']}"


PROBE = ProbeConfig(updates=2, games_per_update=2, eval_games=2, eval_every=1, hidden=16,
                    layers=1, opponent="greedy", reward="vp", seed=3, minibatch=64, lr=3e-3)


@pytest.fixture
def play_calls(monkeypatch):
    """Spies on the play() the training loop calls: [(bound arguments, Episodes)]."""
    calls = []
    signature = inspect.signature(play)

    def spy(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        ep = play(*args, **kwargs)
        calls.append((dict(bound.arguments), ep))
        return ep

    monkeypatch.setattr(base, "play", spy)
    return calls


def test_make_net_is_called_once_under_the_config_seed(tmp_path):
    """The initial weights depend on config.seed alone, and building a Trainer leaves torch's
    global generator where it was."""
    state = torch.get_rng_state()
    a = Probe(PROBE, tmp_path / "a")
    assert torch.equal(torch.get_rng_state(), state)
    b = Probe(PROBE, tmp_path / "b")
    c = Probe(dataclasses.replace(PROBE, seed=4), tmp_path / "c")
    assert a.nets_made == b.nets_made == c.nets_made == 1
    sa, sb, sc = a.net.state_dict(), b.net.state_dict(), c.net.state_dict()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    assert not all(torch.equal(sa[k], sc[k]) for k in sa)


def test_step_learns_once_from_the_games_play_returned(tmp_path, play_calls):
    trainer = Probe(PROBE, tmp_path)
    trainer.step()
    (args, ep), = play_calls
    (call,) = trainer.calls
    assert call.episodes is ep
    assert args["opponent"] == PROBE.opponent and args["games"] == PROBE.games_per_update
    assert args["record"] is True and args["first_seed"] >= PROBE.train_first_seed
    trainer.step()
    assert len(trainer.calls) == 2 and trainer.calls[1].episodes is play_calls[1][1]
    assert play_calls[1][0]["first_seed"] >= args["first_seed"] + PROBE.games_per_update


def test_step_zeroes_the_gradients_before_learn(tmp_path):
    trainer = Probe(PROBE, tmp_path)
    g = gen(5)
    for p in trainer.net.parameters():
        p.grad = torch.randn(p.shape, generator=g)
    trainer.step()
    assert trainer.calls[0].zeroed, "learn() got gradients that were not zeroed"


def test_step_clips_learns_gradient_and_takes_one_adam_step(tmp_path):
    """The metrics hold learn()'s statistics and the norm of its gradient (before clipping);
    the weights move by one Adam step on that gradient clipped to max_grad_norm."""
    cfg = dataclasses.replace(PROBE, max_grad_norm=1e-4)
    trainer = Probe(cfg, tmp_path)
    before = copy.deepcopy(trainer.net)
    metrics = trainer.step()
    (call,) = trainer.calls
    assert STEP_KEYS <= set(metrics) and metrics["probe"] == 42.0
    assert metrics["reward"] == 0.25 and metrics["loss"] == 1.5
    assert metrics["update"] == 1 == trainer.update
    assert metrics["grad_norm"] == pytest.approx(call.norm, rel=1e-5)
    assert call.norm > 100 * cfg.max_grad_norm  # so the clipping mattered
    ref = copy.deepcopy(before)
    for n, p in ref.named_parameters():
        p.grad = call.grads[n].clone()
    torch.nn.utils.clip_grad_norm_(ref.parameters(), cfg.max_grad_norm)
    torch.optim.Adam(ref.parameters(), lr=cfg.lr).step()
    b, r, t = before.state_dict(), ref.state_dict(), trainer.net.state_dict()
    for k in b:
        torch.testing.assert_close(t[k] - b[k], r[k] - b[k], rtol=1e-2, atol=cfg.lr * 1e-3,
                                   msg=k)


def test_run_logs_each_update_with_log_line(tmp_path):
    trainer = Probe(PROBE, tmp_path)
    log: list[str] = []
    trainer.run(log=log.append)
    probe_lines = [s for s in log if s.startswith("probe update")]
    assert probe_lines == [f"probe update {u}: 42.0" for u in range(1, PROBE.updates + 1)]
    assert len(log) == PROBE.updates + PROBE.updates + 1  # and an evaluation before each update


def test_undefined_statistics_are_null_in_the_metrics_file(tmp_path):
    """learn() may report NaN (an undefined statistic): the step's metrics keep it, the file
    gets null, and every line is strict JSON."""
    class Undefined(Probe):
        def learn(self, episodes):
            return {**super().learn(episodes), "probe": math.nan, "big": math.inf}

    trainer = Undefined(PROBE, tmp_path)
    trainer.run(log=None)
    lines = (tmp_path / "metrics.jsonl").read_text().splitlines()

    def strict(token):
        raise AssertionError(f"not strict JSON: {token}")

    parsed = [json.loads(line, parse_constant=strict) for line in lines]
    assert [x["update"] for x in parsed] == list(range(PROBE.updates + 1))
    for x in parsed[1:]:
        assert x["probe"] is None and x["big"] is None
        assert x["loss"] == 1.5 and x["reward"] == 0.25
    assert math.isnan(trainer.step()["probe"])  # in memory it is still NaN


def test_save_writes_the_algorithm_name(tmp_path):
    trainer = Probe(PROBE, tmp_path)
    trainer.save()
    checkpoint = torch.load(tmp_path / "checkpoint.pt", weights_only=True)
    assert checkpoint["algorithm"] == "test"
    assert Probe.config_from_dict(checkpoint["config"]) == PROBE
    assert Probe.saved_config(tmp_path) == PROBE
    assert Probe(PROBE, tmp_path).update == 0  # and it resumes


def test_another_algorithms_checkpoint_is_rejected(tmp_path):
    """Both ways, by the Trainer and by saved_config, naming both algorithms, and the
    checkpoint is left alone."""
    reinforce.Trainer(reinforce.Config(hidden=16, layers=1), tmp_path / "r").save()
    Probe(PROBE, tmp_path / "p").save()
    for trainer_type, config, other, run_dir in [
            (Probe, PROBE, "reinforce", tmp_path / "r"),
            (reinforce.Trainer, reinforce.Config(hidden=16, layers=1), "test", tmp_path / "p")]:
        before = (run_dir / "checkpoint.pt").read_bytes()
        for call in (lambda: trainer_type(config, run_dir),
                     lambda: trainer_type.saved_config(run_dir)):
            with pytest.raises(ValueError) as exc:
                call()
            assert other in str(exc.value) and trainer_type.algorithm in str(exc.value)
        assert (run_dir / "checkpoint.pt").read_bytes() == before


def test_a_checkpoint_without_an_algorithm_is_a_reinforce_run(tmp_path):
    """Checkpoints from before there was more than one algorithm have no "algorithm": they
    resume as REINFORCE runs, and other algorithms refuse them."""
    cfg = reinforce.Config(updates=2, games_per_update=2, eval_games=2, hidden=16, layers=1,
                           opponent="greedy", reward="vp", seed=2)
    old = reinforce.Trainer(cfg, tmp_path)
    old.step()
    old.save()
    path = tmp_path / "checkpoint.pt"
    checkpoint = torch.load(path, weights_only=True)
    del checkpoint["algorithm"]
    torch.save(checkpoint, path)

    assert reinforce.Trainer.saved_config(tmp_path) == cfg
    resumed = reinforce.Trainer(cfg, tmp_path)
    assert resumed.update == 1
    for k, v in resumed.net.state_dict().items():
        assert torch.equal(v, old.net.state_dict()[k]), k
    before = path.read_bytes()
    for call in (lambda: actor_critic.Trainer(actor_critic.Config(hidden=16, layers=1), tmp_path),
                 lambda: actor_critic.Trainer.saved_config(tmp_path),
                 lambda: Probe(PROBE, tmp_path)):
        with pytest.raises(ValueError) as exc:
            call()
        assert "reinforce" in str(exc.value)
    assert path.read_bytes() == before


@pytest.mark.parametrize("trainer_type,config", [
    (Probe, Config()), (Probe, reinforce.Config()), (Probe, actor_critic.Config()),
    (reinforce.Trainer, Config()), (reinforce.Trainer, actor_critic.Config()),
    (reinforce.Trainer, ProbeConfig()), (actor_critic.Trainer, reinforce.Config()),
    (Probe, {"updates": 1}), (reinforce.Trainer, None)])
def test_a_config_of_the_wrong_type_is_a_type_error(tmp_path, trainer_type, config):
    with pytest.raises(TypeError):
        trainer_type(config, tmp_path)


def test_config_from_dict_rejects_another_algorithms_fields():
    with pytest.raises(ValueError) as exc:
        reinforce.Trainer.config_from_dict(dataclasses.asdict(actor_critic.Config()))
    assert "gamma" in str(exc.value) and "value_coef" in str(exc.value)
    with pytest.raises(ValueError):
        Probe.config_from_dict(dataclasses.asdict(reinforce.Config()))
    assert Probe.config_from_dict(dataclasses.asdict(PROBE)) == PROBE


def test_saved_config_of_a_directory_without_a_checkpoint_is_none(tmp_path):
    assert Probe.saved_config(tmp_path) is None
    assert Probe.saved_config(tmp_path / "does-not-exist") is None
