"""Tests for catan_rl.reinforce (rewards, advantages, the gradient, Trainer), written from the
contracts. The Trainer tests here also cover the loop it shares with every algorithm
(trainer.Trainer: resuming, evaluation, checkpoints); test_trainer.py tests the rest of that
module and test_train_cli.py the train/train.py CLI."""

import copy
import dataclasses
import inspect
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from catan_rl import actions
from catan_rl import trainer as base
from catan_rl.engine import ce
from catan_rl.evaluate import MatchResult, play_match, wilson
from catan_rl.policy import Agent, PolicyNet
from catan_rl.reinforce import Config, Trainer, accumulate_gradients, move_advantages
from catan_rl.rollout import Episodes, play
from catan_rl.trainer import game_rewards

AT = ce.ActionType

STEP_KEYS = {"update", "games", "moves", "win_rate", "mean_vp", "mean_turns", "capped", "reward",
             "loss", "log_prob", "entropy", "grad_norm", "rollout_seconds", "update_seconds"}
EVAL_KEYS = {"eval_win_rate", "eval_ci_low", "eval_ci_high", "eval_games"}


def gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def make_net(seed: int = 0, **kwargs) -> PolicyNet:
    """A PolicyNet built under a fixed torch seed, without touching the global RNG state."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return PolicyNet(**kwargs)


# ---- building Episodes by hand ------------------------------------------------------------------

def make_episodes(won, vp, game=(), *, capped=None, turns=None, rows=None) -> Episodes:
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
        seat=np.arange(G, dtype=np.int64) % 4, won=won,
        capped=np.zeros(G, bool) if capped is None else np.asarray(capped, bool),
        vp=np.asarray(vp, dtype=np.int64),
        turns=np.full(G, 80, np.int64) if turns is None else np.asarray(turns, np.int64),
        decisions=M, seconds=0.1)


def position_episodes(positions, idx, games: int, seed: int) -> Episodes:
    """Moves at the fixture positions `idx` (a random legal move each), spread over `games`
    games at random."""
    rng = np.random.default_rng(seed)
    idx = np.asarray(idx)
    moves = [positions.legal[i][rng.integers(len(positions.legal[i]))] for i in idx]
    game = rng.integers(games, size=len(idx))
    game[:games] = np.arange(min(games, len(idx)))  # every game has a move
    won = rng.random(games) < 0.4
    vp = np.where(won, 10, rng.integers(2, 10, size=games))
    return make_episodes(won, vp, game, rows=(
        positions.obs[idx], positions.mask[idx],
        [m[0] for m in moves], [m[1] for m in moves]))


def reference(net: PolicyNet, ep: Episodes, adv, entropy_coef: float = 0.0):
    """The documented loss, full batch, on a copy of `net`: (grads by name, loss, mean
    log-prob, mean entropy)."""
    net = copy.deepcopy(net)
    for p in net.parameters():
        p.grad = None
    obs = torch.as_tensor(ep.obs)
    lp, ent = actions.log_prob_entropy(net(obs), torch.as_tensor(ep.mask), actions.phase_of(obs),
                                       torch.as_tensor(ep.types), torch.as_tensor(ep.args))
    loss = -(torch.as_tensor(adv, dtype=torch.float32) * lp).mean() - entropy_coef * ent.mean()
    loss.backward()
    grads = {n: (p.grad.clone() if p.grad is not None else torch.zeros_like(p))
             for n, p in net.named_parameters()}
    return grads, float(loss.detach()), float(lp.detach().mean()), float(ent.detach().mean())


def grads_of(net: PolicyNet) -> dict[str, torch.Tensor]:
    return {n: (p.grad.clone() if p.grad is not None else torch.zeros_like(p))
            for n, p in net.named_parameters()}


def assert_grads_close(got: dict, want: dict, scale: float = 1.0):
    assert got.keys() == want.keys()
    for n in want:
        torch.testing.assert_close(got[n], want[n], rtol=2e-4, atol=2e-6 * scale, msg=n)


def log_probs(net: PolicyNet, ep: Episodes) -> torch.Tensor:
    obs = torch.as_tensor(ep.obs)
    with torch.no_grad():
        lp, _ = actions.log_prob_entropy(net(obs), torch.as_tensor(ep.mask),
                                         actions.phase_of(obs), torch.as_tensor(ep.types),
                                         torch.as_tensor(ep.args))
    return lp


@pytest.fixture(scope="module")
def recorded():
    """Real episodes: an untrained network (hidden 32) recorded against greedy."""
    net = make_net(21, hidden=32)
    ep = play(Agent(net, generator=gen(22)), "greedy", games=6, first_seed=5001,
              generator=gen(23), record=True)
    return SimpleNamespace(net=net, ep=ep)


# ---- game_rewards (trainer.py, shared by every algorithm) ---------------------------------------

def test_game_rewards_win():
    ep = make_episodes(won=[True, False, False, True, False], vp=[10, 9, 3, 12, 0],
                       capped=[False, False, True, False, False])
    r = game_rewards(ep, "win")
    assert type(r) is np.ndarray and r.dtype == np.float32 and r.shape == (5,)
    assert r.tolist() == [1.0, 0.0, 0.0, 1.0, 0.0]


def test_game_rewards_vp_is_capped_at_one():
    ep = make_episodes(won=[False, False, False, True, True, False], vp=[0, 3, 9, 10, 12, 7])
    r = game_rewards(ep, "vp")
    assert type(r) is np.ndarray and r.dtype == np.float32 and r.shape == (6,)
    np.testing.assert_allclose(r, np.array([0, 0.3, 0.9, 1, 1, 0.7], np.float32), rtol=1e-6)


@pytest.mark.parametrize("reward", ["", "Win", "points", "loss"])
def test_game_rewards_rejects_unknown(reward):
    with pytest.raises(ValueError):
        game_rewards(make_episodes([True], [10]), reward)


# ---- move_advantages ----------------------------------------------------------------------------

def test_move_advantages_mean_baseline_is_over_games_not_moves():
    """Game 0 (won) has 5 moves, games 1 and 2 one each, game 3 none: the baseline is the mean
    over the 4 games (1/4), not over the 7 moves (5/7)."""
    ep = make_episodes(won=[True, False, False, False], vp=[10, 4, 5, 6],
                       game=[0, 0, 1, 0, 0, 2, 0])
    rewards = game_rewards(ep, "win")
    adv = move_advantages(ep, rewards, "mean")
    assert type(adv) is np.ndarray and adv.dtype == np.float32 and adv.shape == (7,)
    np.testing.assert_allclose(adv, [0.75, 0.75, -0.25, 0.75, 0.75, -0.25, 0.75], rtol=1e-6)


def test_move_advantages_mean_baseline_with_vp_rewards():
    ep = make_episodes(won=[False, False, True], vp=[2, 6, 10], game=[2, 1, 1, 0, 2])
    rewards = game_rewards(ep, "vp")  # 0.2, 0.6, 1.0: mean 0.6
    adv = move_advantages(ep, rewards, "mean")
    np.testing.assert_allclose(adv, [0.4, 0.0, 0.0, -0.4, 0.4], atol=1e-6)


def test_move_advantages_no_baseline():
    ep = make_episodes(won=[False, True, False], vp=[3, 10, 7], game=[1, 0, 2, 1, 1])
    for reward in ("win", "vp"):
        rewards = game_rewards(ep, reward)
        adv = move_advantages(ep, rewards, "none")
        assert adv.dtype == np.float32 and adv.shape == (5,)
        np.testing.assert_allclose(adv, rewards[ep.game], rtol=1e-6)


def test_move_advantages_accepts_float64_rewards_and_no_moves():
    ep = make_episodes(won=[True, False], vp=[10, 1], game=[1, 0, 1])
    adv = move_advantages(ep, np.array([0.5, 0.25]), "mean")
    assert adv.dtype == np.float32
    np.testing.assert_allclose(adv, [-0.125, 0.125, -0.125], rtol=1e-6)
    empty = move_advantages(make_episodes([True, False], [10, 2]), np.array([1, 0], np.float32),
                            "mean")
    assert empty.shape == (0,) and empty.dtype == np.float32


@pytest.mark.parametrize("baseline", ["", "Mean", "median", "zero"])
def test_move_advantages_rejects_unknown(baseline):
    ep = make_episodes([True, False], [10, 2], game=[0, 1])
    with pytest.raises(ValueError):
        move_advantages(ep, game_rewards(ep, "win"), baseline)


# ---- accumulate_gradients -----------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic(positions):
    """61 moves at fixture positions over 9 games, with mixed-sign advantages."""
    idx = np.random.default_rng(0).choice(len(positions), 61, replace=False)
    ep = position_episodes(positions, idx, games=9, seed=1)
    adv = move_advantages(ep, game_rewards(ep, "vp"), "mean")
    assert np.any(adv > 0) and np.any(adv < 0)
    return ep, adv


@pytest.mark.parametrize("minibatch", [1, 7, 60, 61, 100_000])
@pytest.mark.parametrize("entropy_coef", [0.0, 0.05])
def test_accumulate_gradients_is_the_full_batch_gradient(synthetic, minibatch, entropy_coef):
    """Whatever the chunking, the accumulated gradient is that of the documented loss over all
    moves at once, the returned means are the loss / mean log-prob / mean entropy, nothing is
    stepped, and the inputs are left alone."""
    ep, adv = synthetic
    net = make_net(2, hidden=32)
    want, loss, mean_lp, mean_ent = reference(net, ep, adv, entropy_coef)
    before = copy.deepcopy(net.state_dict())
    ep_copy = copy.deepcopy(ep)
    adv_copy = adv.copy()
    out = accumulate_gradients(net, ep, adv, minibatch=minibatch, entropy_coef=entropy_coef)
    assert_grads_close(grads_of(net), want)
    assert {"loss", "log_prob", "entropy"} <= set(out)
    assert all(isinstance(out[k], float) for k in ("loss", "log_prob", "entropy"))
    assert out["loss"] == pytest.approx(loss, rel=1e-4, abs=1e-6)
    assert out["log_prob"] == pytest.approx(mean_lp, rel=1e-4, abs=1e-6)
    assert out["entropy"] == pytest.approx(mean_ent, rel=1e-4, abs=1e-6)
    for k, v in net.state_dict().items():
        assert torch.equal(v, before[k]), f"{k} changed: the optimizer must not be stepped"
    assert np.array_equal(adv, adv_copy)
    for f in dataclasses.fields(Episodes):
        x, y = getattr(ep, f.name), getattr(ep_copy, f.name)
        assert (np.array_equal(x, y) if isinstance(x, np.ndarray) else x == y), f.name


def test_accumulate_gradients_adds_to_existing_grads(synthetic):
    ep, adv = synthetic
    net = make_net(3, hidden=32)
    want, *_ = reference(net, ep, adv, 0.01)
    g = torch.Generator().manual_seed(4)
    existing = {n: torch.randn(p.shape, generator=g) * 1e-2 for n, p in net.named_parameters()}
    for n, p in net.named_parameters():
        p.grad = existing[n].clone()
    accumulate_gradients(net, ep, adv, minibatch=16, entropy_coef=0.01)
    assert_grads_close(grads_of(net), {n: existing[n] + want[n] for n in want})
    # And a second call adds the same gradient again.
    accumulate_gradients(net, ep, adv, minibatch=16, entropy_coef=0.01)
    assert_grads_close(grads_of(net), {n: existing[n] + 2 * want[n] for n in want})


def test_accumulate_gradients_zero_advantage_zero_gradient(synthetic):
    ep, adv = synthetic
    net = make_net(5, hidden=32)
    out = accumulate_gradients(net, ep, np.zeros_like(adv), minibatch=10, entropy_coef=0.0)
    for n, p in net.named_parameters():
        assert p.grad is None or torch.count_nonzero(p.grad) == 0, n
    assert out["loss"] == 0.0
    # The entropy term alone does push.
    accumulate_gradients(net, ep, np.zeros_like(adv), minibatch=10, entropy_coef=0.1)
    assert any(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in net.parameters())


def test_accumulate_gradients_rejects_no_moves():
    net = make_net(6, hidden=16, layers=1)
    ep = make_episodes(won=[True, False], vp=[10, 3])
    with pytest.raises(ValueError):
        accumulate_gradients(net, ep, np.zeros(0, np.float32))


def test_accumulate_gradients_on_recorded_episodes(recorded):
    """Real episodes from play(record=True), rewards and advantages from this module."""
    ep = recorded.ep
    net = copy.deepcopy(recorded.net)
    for reward, baseline in [("vp", "mean"), ("win", "none")]:
        adv = move_advantages(ep, game_rewards(ep, reward), baseline)
        want, loss, mean_lp, mean_ent = reference(net, ep, adv, 0.02)
        for p in net.parameters():
            p.grad = None
        out = accumulate_gradients(net, ep, adv, minibatch=97, entropy_coef=0.02)
        assert_grads_close(grads_of(net), want)
        assert out["loss"] == pytest.approx(loss, rel=1e-4, abs=1e-6)
        # The episodes were played by this very network, so the mean log-prob is the recorded
        # one.
        assert out["log_prob"] == pytest.approx(float(ep.log_prob.mean()), rel=1e-4, abs=1e-5)
        assert out["log_prob"] == pytest.approx(mean_lp, rel=1e-4, abs=1e-6)
        assert out["entropy"] == pytest.approx(mean_ent, rel=1e-4, abs=1e-6)


# ---- the direction of learning ---------------------------------------------------------------

@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_one_step_moves_log_prob_with_the_advantage(positions, sign):
    """One small SGD step on a single move raises its log-prob for a positive advantage and
    lowers it for a negative one, at many different positions."""
    net = make_net(7, hidden=32)
    choices = positions.where(lambda i: len(positions.legal[i]) >= 2)
    idx = choices[:: max(1, len(choices) // 12)][:12]
    for k, i in enumerate(idx):
        ep = position_episodes(positions, [i], games=1, seed=k)
        n = copy.deepcopy(net)
        before = float(log_probs(n, ep)[0])
        accumulate_gradients(n, ep, np.array([sign], np.float32))
        torch.optim.SGD(n.parameters(), lr=1e-3).step()
        after = float(log_probs(n, ep)[0])
        assert (after - before) * sign > 0, (i, before, after)


def test_one_step_increases_the_weighted_objective(synthetic):
    """A small step against the loss raises sum_m adv_m * log pi(m), the objective."""
    ep, adv = synthetic
    net = make_net(8, hidden=32)
    before = float((torch.as_tensor(adv) * log_probs(net, ep)).sum())
    accumulate_gradients(net, ep, adv, minibatch=13)
    torch.optim.SGD(net.parameters(), lr=1e-3).step()
    after = float((torch.as_tensor(adv) * log_probs(net, ep)).sum())
    assert after > before


def type_prob(net: PolicyNet, obs: np.ndarray, mask: np.ndarray, t: int) -> np.ndarray:
    """P(type t) at each row (t takes no argument, so this is the move's probability)."""
    o = torch.as_tensor(obs)
    n = len(obs)
    with torch.no_grad():
        lp, _ = actions.log_prob_entropy(net(o), torch.as_tensor(mask), actions.phase_of(o),
                                         torch.full((n,), t), torch.zeros(n, dtype=torch.int64))
    return lp.exp().numpy()


def test_training_on_fixed_episodes_favours_the_rewarded_type(recorded, positions):
    """Rewarding the agent's END_TURN moves (advantage 1, others 0) for a few steps raises
    P(END_TURN) at held-out positions where it competes with other types."""
    ep = recorded.ep
    target = int(AT.END_TURN)
    adv = (ep.types == target).astype(np.float32)
    assert 0 < adv.sum() < len(adv)
    held = positions.where(lambda i: positions.mask[i, target]
                           and positions.mask[i, :ce.NUM_ACTION_TYPES].sum() >= 2)
    assert len(held) >= 10
    net = copy.deepcopy(recorded.net)
    p0 = type_prob(net, positions.obs[held], positions.mask[held], target)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(15):
        opt.zero_grad()
        accumulate_gradients(net, ep, adv, minibatch=128)
        opt.step()
    p1 = type_prob(net, positions.obs[held], positions.mask[held], target)
    assert p1.mean() > p0.mean() + 0.2, (p0.mean(), p1.mean())
    assert (p1 > p0).mean() > 0.9


# ---- Config -------------------------------------------------------------------------------------

def test_config_from_dict_round_trips():
    for c in (Config(), Config(updates=7, lr=3e-4, reward="vp", baseline="none",
                               entropy_coef=0.01, opponent="greedy", seed=4, hidden=64)):
        assert Trainer.config_from_dict(dataclasses.asdict(c)) == c
        # As stored in JSON, too.
        assert Trainer.config_from_dict(json.loads(json.dumps(dataclasses.asdict(c)))) == c


def test_config_from_dict_rejects_unknown_keys():
    d = dataclasses.asdict(Config())
    d["learning_rate"] = 0.1
    with pytest.raises(ValueError):
        Trainer.config_from_dict(d)


# ---- Trainer ----------------------------------------------------------------------------------

# Small and fast: games against greedy are ~75 turns. minibatch < moves per update, so the
# Trainer's gradient is chunked. An untrained agent never beats greedy, so with reward "win"
# every advantage would be 0 and nothing would train: "vp" gives a real gradient.
TINY = Config(updates=4, games_per_update=6, eval_games=4, eval_every=2, hidden=32,
              opponent="greedy", reward="vp", seed=5, minibatch=150)


def read_metrics(run_dir) -> list[dict]:
    text = (Path(run_dir) / "metrics.jsonl").read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def untimed(line: dict) -> dict:
    return {k: v for k, v in line.items() if not k.endswith("seconds")}


def assert_same_weights(a: torch.nn.Module, b: torch.nn.Module):
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    for k in sa:
        assert torch.equal(sa[k], sb[k]), f"{k} differs"


def assert_same_optimizer(a: torch.optim.Optimizer, b: torch.optim.Optimizer):
    sa, sb = a.state_dict()["state"], b.state_dict()["state"]
    assert sa.keys() == sb.keys()
    for k in sa:
        assert sa[k].keys() == sb[k].keys()
        for name in sa[k]:
            x, y = sa[k][name], sb[k][name]
            assert (torch.equal(x, y) if torch.is_tensor(x) else x == y), (k, name)


def new_dir(tmp_path: Path, name: str) -> Path:
    d = tmp_path / name
    d.mkdir()
    return d


def eval_updates(config: Config, start: int = 0) -> set[int]:
    """Updates whose metrics line carries an evaluation in a run from `start` (0 = fresh)."""
    done = {0} if start == 0 else set()
    done |= {u for u in range(start + 1, config.updates + 1) if u % config.eval_every == 0}
    done.add(config.updates)
    return done


@pytest.fixture(scope="module")
def straight(tmp_path_factory):
    """A complete fresh run of TINY."""
    run_dir = tmp_path_factory.mktemp("straight")
    trainer = Trainer(TINY, run_dir)
    fresh_update = trainer.update
    log: list[str] = []
    trainer.run(log=log.append)
    return SimpleNamespace(trainer=trainer, dir=run_dir, log=log, metrics=read_metrics(run_dir),
                           fresh_update=fresh_update)


def test_run_metrics_lines(straight):
    """Update 0 is the evaluation alone; then one line per update with the documented keys,
    with the evaluation fields on every eval_every-th update and on the last."""
    lines = straight.metrics
    assert [line["update"] for line in lines] == list(range(TINY.updates + 1))
    first = lines[0]
    assert EVAL_KEYS <= set(first)
    assert not ({"games", "moves", "loss", "grad_norm", "entropy"} & set(first))
    evals = eval_updates(TINY)
    for line in lines[1:]:
        u = line["update"]
        assert STEP_KEYS <= set(line), f"update {u} misses {STEP_KEYS - set(line)}"
        assert (EVAL_KEYS <= set(line)) == (u in evals), f"update {u}"
        assert not (EVAL_KEYS & set(line)) or u in evals
        assert line["games"] == TINY.games_per_update
        assert line["moves"] > 4 * TINY.games_per_update  # setup alone is 4 moves a game
        assert 0 <= line["win_rate"] <= 1 and 0 <= line["reward"] <= 1
        assert 0 <= line["capped"] <= TINY.games_per_update
        assert 0 <= line["mean_vp"] <= 12 and line["mean_turns"] > 0
        assert line["log_prob"] < 0 and line["entropy"] > 0 and line["grad_norm"] > 0
        assert np.isfinite(line["loss"])
        assert line["rollout_seconds"] >= 0 and line["update_seconds"] >= 0
    for line in lines:
        if line["update"] in evals:
            assert line["eval_games"] == TINY.eval_games
            wins = line["eval_win_rate"] * TINY.eval_games
            assert wins == pytest.approx(round(wins), abs=1e-9)
            low, high = wilson(round(wins), TINY.eval_games)
            assert line["eval_ci_low"] == pytest.approx(low, abs=1e-9)
            assert line["eval_ci_high"] == pytest.approx(high, abs=1e-9)


def test_run_logs_one_line_per_update_and_evaluation(straight):
    assert len(straight.log) == TINY.updates + len(eval_updates(TINY))
    assert all(isinstance(s, str) and s.strip() and "\n" not in s.strip() for s in straight.log)


def test_run_leaves_a_checkpoint_and_counts_updates(straight):
    assert straight.fresh_update == 0
    assert straight.trainer.update == TINY.updates
    assert (straight.dir / "checkpoint.pt").is_file()
    leftovers = [p.name for p in straight.dir.iterdir()
                 if p.name not in ("checkpoint.pt", "metrics.jsonl")
                 and ("checkpoint" in p.name or p.suffix in (".tmp", ".temp", ".partial"))]
    assert leftovers == [], "temporary checkpoint files left behind"


def test_run_changes_the_weights(straight, tmp_path):
    fresh = Trainer(TINY, tmp_path)  # never run: the initial weights
    changed = [k for k, v in straight.trainer.net.state_dict().items()
               if not torch.equal(v, fresh.net.state_dict()[k])]
    assert changed


def test_two_fresh_runs_are_identical(straight, tmp_path):
    trainer = Trainer(TINY, tmp_path)
    trainer.run(log=None)
    assert_same_weights(trainer.net, straight.trainer.net)
    assert_same_optimizer(trainer.optimizer, straight.trainer.optimizer)
    assert [untimed(x) for x in read_metrics(tmp_path)] == [untimed(x) for x in straight.metrics]


@pytest.mark.parametrize("stop", [1, 2, 3])
def test_stop_and_resume_equals_never_stopping(straight, tmp_path, stop):
    """Run `stop` updates, then a new Trainer on the same directory with updates=4: the same
    weights and optimizer state as the straight 4-update run, and the same metrics lines apart
    from timing (and from the extra evaluation the stopped run made after its last update)."""
    first = Trainer(dataclasses.replace(TINY, updates=stop), tmp_path)
    first.run(log=None)
    assert first.update == stop
    resumed = Trainer(TINY, tmp_path)
    assert resumed.update == stop
    assert resumed.config == TINY
    resumed.run(log=None)
    assert resumed.update == TINY.updates
    assert_same_weights(resumed.net, straight.trainer.net)
    assert_same_optimizer(resumed.optimizer, straight.trainer.optimizer)

    got = [untimed(x) for x in read_metrics(tmp_path)]
    want = [untimed(x) for x in straight.metrics]
    assert [x["update"] for x in got] == [x["update"] for x in want]
    for g, w in zip(got, want):
        if g["update"] == stop and stop not in eval_updates(TINY):
            assert EVAL_KEYS <= set(g)  # the stopped run evaluated after its last update
            g = {k: v for k, v in g.items() if k not in EVAL_KEYS}
        assert g == w, f"update {g['update']}"


@pytest.mark.parametrize("crash_at", [2, 5])
def test_resume_after_a_crash_equals_never_stopping(straight, tmp_path, crash_at):
    """A run that dies mid-way (here: the log callback raises on its crash_at-th line: during
    update 1, before any save but the initial one, or during update 3) resumes from its last
    checkpoint and ends exactly like the run that never stopped."""
    calls = []

    def dying_log(line: str):
        calls.append(line)
        if len(calls) == crash_at:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        Trainer(TINY, tmp_path).run(log=dying_log)
    resumed = Trainer(TINY, tmp_path)
    assert 0 <= resumed.update < TINY.updates
    resumed.run(log=None)
    assert_same_weights(resumed.net, straight.trainer.net)
    assert [untimed(x) for x in read_metrics(tmp_path)] == [untimed(x) for x in straight.metrics]


def test_resume_drops_metrics_past_the_checkpoint(straight, tmp_path):
    Trainer(dataclasses.replace(TINY, updates=2), tmp_path).run(log=None)
    with open(tmp_path / "metrics.jsonl", "a") as f:
        f.write(json.dumps({"update": 3, "fake": True, "loss": 123.0}) + "\n")
        f.write(json.dumps({"update": 9, "fake": True}) + "\n")
    resumed = Trainer(TINY, tmp_path)
    assert resumed.update == 2
    resumed.run(log=None)
    lines = read_metrics(tmp_path)
    assert not any(x.get("fake") for x in lines)
    assert [untimed(x) for x in lines] == [untimed(x) for x in straight.metrics]
    assert_same_weights(resumed.net, straight.trainer.net)


def test_run_after_the_last_update_does_nothing(straight, tmp_path):
    """Resuming a finished run (a copy of its directory) with the same config trains no more and
    writes nothing."""
    run_dir = tmp_path / "copy"
    shutil.copytree(straight.dir, run_dir)
    before = (run_dir / "metrics.jsonl").read_bytes()
    trainer = Trainer(TINY, run_dir)
    assert trainer.update == TINY.updates
    trainer.run(log=None)
    assert trainer.update == TINY.updates
    assert (run_dir / "metrics.jsonl").read_bytes() == before
    assert_same_weights(trainer.net, straight.trainer.net)


@pytest.mark.parametrize("changes,names", [
    ({"lr": 3e-3}, ["lr"]),
    ({"lr": 3e-3, "entropy_coef": 0.01}, ["lr", "entropy_coef"]),
    ({"hidden": 64}, ["hidden"]),
    ({"reward": "win"}, ["reward"]),
])
def test_resume_rejects_a_different_config(tmp_path, changes, names):
    Trainer(TINY, tmp_path).save()
    with pytest.raises(ValueError) as exc:
        Trainer(dataclasses.replace(TINY, **changes), tmp_path)
    for name in names:
        assert name in str(exc.value)


def test_resume_allows_more_updates(tmp_path):
    first = Trainer(dataclasses.replace(TINY, updates=1), tmp_path)
    first.step()
    first.save()
    more = Trainer(dataclasses.replace(TINY, updates=9), tmp_path)
    assert more.update == 1
    assert more.config == dataclasses.replace(TINY, updates=9)
    assert_same_weights(more.net, first.net)


def test_run_silent_without_log(tmp_path, capsys):
    Trainer(dataclasses.replace(TINY, updates=1), tmp_path).run(log=None)
    out = capsys.readouterr()
    assert out.out == "" and out.err == ""
    assert [x["update"] for x in read_metrics(tmp_path)] == [0, 1]


@pytest.mark.parametrize("changes", [
    {"reward": "loss"}, {"reward": ""}, {"baseline": "median"}, {"baseline": "Mean"},
    {"opponent": "self"}, {"opponent": "bogus"}])
def test_invalid_config_rejected(tmp_path, changes):
    with pytest.raises(ValueError):
        Trainer(dataclasses.replace(TINY, **changes), tmp_path)


@pytest.fixture
def play_calls(monkeypatch):
    """Spies on the play() the training loop (trainer.py) calls: [(bound arguments, Episodes)]."""
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


def test_step_is_one_clipped_adam_step_on_the_documented_loss(tmp_path, play_calls):
    """Reproduces step() independently from the episodes it played: rewards and advantages,
    the full-batch gradient of the loss (with entropy), its norm (reported before clipping),
    clipping to max_grad_norm, then one Adam step with lr; and the metrics from the episodes.
    max_grad_norm is tiny so that the clipping visibly changes Adam's first step."""
    cfg = Config(updates=1, games_per_update=5, eval_games=4, hidden=32, opponent="greedy",
                 seed=11, lr=3e-3, max_grad_norm=1e-6, minibatch=100, reward="vp",
                 baseline="mean", entropy_coef=0.01)
    trainer = Trainer(cfg, tmp_path)
    before = copy.deepcopy(trainer.net)
    metrics = trainer.step()
    assert trainer.update == 1 and metrics["update"] == 1
    (args, ep), = play_calls
    assert args["opponent"] == "greedy" and args["games"] == 5 and args["record"] is True
    assert args["first_seed"] >= cfg.train_first_seed
    # The moves were sampled from the network as it was before the step (sampled: an untrained
    # net is nearly uniform, so most of its choices are not the most likely move).
    np.testing.assert_allclose(ep.log_prob, log_probs(before, ep).numpy(), rtol=1e-5, atol=1e-5)
    obs = torch.as_tensor(ep.obs)
    with torch.no_grad():
        mode_types, mode_args = actions.mode(before(obs), torch.as_tensor(ep.mask),
                                             actions.phase_of(obs))
    chose_mode = (mode_types.numpy() == ep.types) & (mode_args.numpy() == ep.args)
    assert chose_mode.mean() < 0.9

    rewards = np.minimum(ep.vp, 10) / 10
    adv = (rewards[ep.game] - rewards.mean()).astype(np.float32)
    ref = copy.deepcopy(before)
    grads, loss, mean_lp, mean_ent = reference(ref, ep, adv, cfg.entropy_coef)
    for n, p in ref.named_parameters():
        p.grad = grads[n].clone()
    norm = float(torch.nn.utils.clip_grad_norm_(ref.parameters(), cfg.max_grad_norm))
    torch.optim.Adam(ref.parameters(), lr=cfg.lr).step()

    assert metrics["grad_norm"] == pytest.approx(norm, rel=1e-4)
    assert norm > 100 * cfg.max_grad_norm  # so the clipping mattered
    b, r, t = before.state_dict(), ref.state_dict(), trainer.net.state_dict()
    for k in b:
        torch.testing.assert_close(t[k] - b[k], r[k] - b[k], rtol=1e-2, atol=cfg.lr * 1e-3,
                                   msg=k)

    assert STEP_KEYS <= set(metrics)
    assert metrics["games"] == 5 and metrics["moves"] == ep.moves
    assert metrics["win_rate"] == pytest.approx(ep.won.mean())
    assert metrics["mean_vp"] == pytest.approx(ep.vp.mean())
    assert metrics["mean_turns"] == pytest.approx(ep.turns.mean())
    assert metrics["capped"] in (pytest.approx(ep.capped.sum()), pytest.approx(ep.capped.mean()))
    assert metrics["reward"] == pytest.approx(rewards.mean(), rel=1e-5)
    assert metrics["loss"] == pytest.approx(loss, rel=1e-3, abs=1e-6)
    assert metrics["log_prob"] == pytest.approx(mean_lp, rel=1e-4)
    assert metrics["entropy"] == pytest.approx(mean_ent, rel=1e-4)
    assert metrics["rollout_seconds"] >= 0 and metrics["update_seconds"] >= 0
    json.dumps(metrics)  # a metrics line


def test_training_seeds_advance(tmp_path, play_calls):
    """Consecutive steps play recorded games on fresh, non-overlapping board seeds."""
    trainer = Trainer(TINY, tmp_path)
    m1 = trainer.step()
    m2 = trainer.step()
    assert (m1["update"], m2["update"], trainer.update) == (1, 2, 2)
    (a1, _), (a2, _) = play_calls
    assert a1["first_seed"] >= TINY.train_first_seed
    assert a2["first_seed"] >= a1["first_seed"] + TINY.games_per_update
    for a in (a1, a2):
        assert a["games"] == TINY.games_per_update and a["record"] is True
        assert a["opponent"] == TINY.opponent


def test_evaluate(tmp_path, monkeypatch):
    """play_match against config.opponent, eval_games games on its default seeds; repeatable
    at a given update; and it changes neither the weights nor the training that follows."""
    seen = []
    signature = inspect.signature(play_match)

    def spy(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        seen.append(dict(bound.arguments))
        return play_match(*args, **kwargs)

    monkeypatch.setattr(base, "play_match", spy)
    a = Trainer(TINY, new_dir(tmp_path, "a"))
    b = Trainer(TINY, new_dir(tmp_path, "b"))
    a1, b1 = a.step(), b.step()
    weights = copy.deepcopy(a.net.state_dict())
    r1 = a.evaluate()
    r2 = a.evaluate()
    assert isinstance(r1, MatchResult)
    assert r1.opponent == TINY.opponent and r1.games == TINY.eval_games
    assert dataclasses.replace(r1, seconds=0.0) == dataclasses.replace(r2, seconds=0.0)
    for args in seen:
        assert args["opponent"] == TINY.opponent and args["games"] == TINY.eval_games
        assert args["first_seed"] == 1 and args["standard_board"] is False
    for k, v in a.net.state_dict().items():
        assert torch.equal(v, weights[k]), k
    a2, b2 = a.step(), b.step()
    assert untimed(a1) == untimed(b1) and untimed(a2) == untimed(b2)
    assert_same_weights(a.net, b.net)


def test_step_level_resume_restores_every_generator(tmp_path):
    """Against random bots (whose generator now matters too): step, save, resume in a new
    Trainer, step == two steps in one Trainer."""
    cfg = Config(updates=2, games_per_update=2, eval_games=2, hidden=16, layers=1,
                 opponent="random", reward="vp", seed=9, minibatch=512)
    straight = Trainer(cfg, new_dir(tmp_path, "straight"))
    straight.step()
    want = straight.step()
    first = Trainer(cfg, new_dir(tmp_path, "resumed"))
    first.step()
    first.save()
    resumed = Trainer(cfg, tmp_path / "resumed")
    assert resumed.update == 1
    got = resumed.step()
    assert untimed(got) == untimed(want)
    assert_same_weights(resumed.net, straight.net)
    assert_same_optimizer(resumed.optimizer, straight.optimizer)


def test_save_is_atomic(tmp_path, monkeypatch):
    """A save that dies while writing leaves the previous checkpoint intact."""
    trainer = Trainer(TINY, tmp_path)
    trainer.save()
    good = (tmp_path / "checkpoint.pt").read_bytes()
    trainer.step()

    def broken_save(obj, f, *args, **kwargs):
        if isinstance(f, (str, os.PathLike)):
            Path(f).write_bytes(b"partial")
        else:
            f.write(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", broken_save)
    with pytest.raises(OSError):
        trainer.save()
    monkeypatch.undo()
    assert (tmp_path / "checkpoint.pt").read_bytes() == good
    assert Trainer(TINY, tmp_path).update == 0
