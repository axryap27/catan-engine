"""Tests for catan_rl.gae (stage 4), written from the contracts.

Stage 4's two steps are yours to write: td_errors (step 1) and gae_advantages (step 2). Each
step's tests are skipped until you write it (see lab_steps.py), and then they run.

Everything else in the module is tested now. The plumbing (value_estimates, the Trainer) runs
twice, under the `impl` fixture: "stand-in" swaps in ac_helpers.py's stand-ins for stage 3's
three steps and the definitions below for stage 4's two, so the loop is tested before you write
anything, and "yours" uses the modules' own once every step through stage 4 is written. A few
tests then need your code alone: step() is one clipped Adam step on the documented loss with
GAE's advantages, with lambda = 1 it is stage 3's step, and the critic learns.
"""

import copy
import dataclasses
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ac_helpers import (adam_step, assert_same_optimizer, assert_same_step, assert_same_weights,
                        by_hand_explained_variance, definition_returns, eval_updates, fresh_copy,
                        gen, grads_of, log_probs, make_ac, make_episodes, make_net, outputs,
                        play_calls, position_episodes, read_metrics, reference, untimed,
                        use_stage3_stand_ins)
from catan_rl import actor_critic, gae, reinforce
from catan_rl import trainer as base
from catan_rl.actor_critic import final_move_rewards
from catan_rl.gae import Config, Trainer, value_estimates
from catan_rl.policy import Agent
from catan_rl.rollout import play
from catan_rl.trainer import game_rewards
from lab_steps import STAGE4 as steps

# trainer.Trainer.step()'s metrics and actor-critic's (as in test_actor_critic.py), then what
# GAE adds.
STEP_KEYS = {"update", "games", "moves", "win_rate", "mean_vp", "mean_turns", "capped", "reward",
             "loss", "log_prob", "entropy", "grad_norm", "rollout_seconds", "update_seconds"}
AC_KEYS = {"return", "policy_loss", "value_loss", "value", "explained_variance"}
GAE_KEYS = {"target", "advantage_std"}
EVAL_KEYS = {"eval_win_rate", "eval_ci_low", "eval_ci_high", "eval_games"}


# ---- td_errors' and gae_advantages' definitions (also their stand-ins) ---------------------------

def _td_errors64(rewards, values, game, gamma: float) -> np.ndarray:
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    game = np.asarray(game)
    out = np.zeros(len(rewards))
    for m in range(len(rewards)):
        later = m + 1 + np.flatnonzero(game[m + 1:] == game[m])  # m's game's moves after m
        next_value = values[later[0]] if len(later) else 0.0
        out[m] = rewards[m] + gamma * next_value - values[m]
    return out


def definition_td_errors(rewards, values, game, games: int, gamma: float) -> np.ndarray:
    """td_errors' docstring, evaluated literally and slowly: delta[m] = rewards[m] + gamma *
    values[n] - values[m], n the next move of m's game (values[n] = 0 if there is none)."""
    return _td_errors64(rewards, values, game, gamma).astype(np.float32)


def definition_gae(rewards, values, game, games: int, gamma: float, lam: float) -> np.ndarray:
    """gae_advantages' docstring, evaluated literally and slowly: move m's advantage is the sum,
    over m's game's moves n_0 = m, n_1, n_2, ... from m on, of (gamma * lam)^j * delta[n_j]."""
    delta = _td_errors64(rewards, values, game, gamma)
    game = np.asarray(game)
    out = np.zeros(len(delta))
    for m in range(len(delta)):
        later = m + np.flatnonzero(game[m:] == game[m])  # m's game's moves from m on, in order
        out[m] = sum((gamma * lam) ** j * delta[n] for j, n in enumerate(later))
    return out.astype(np.float32)


@pytest.fixture(params=["stand-in", pytest.param("yours", marks=steps.all_steps)])
def impl(request, monkeypatch):
    """Which steps gae's plumbing runs on: stage 3's stand-ins and stage 4's definitions, or
    yours (all of stage 3's and stage 4's)."""
    if request.param == "stand-in":
        use_stage3_stand_ins(monkeypatch)
        monkeypatch.setattr(gae, "td_errors", definition_td_errors)
        monkeypatch.setattr(gae, "gae_advantages", definition_gae)
    return SimpleNamespace(name=request.param)


def random_games(seed: int):
    """(rewards, values, game, games): many games of uneven lengths, some (in the middle and
    past the last game that has moves) with no moves at all; rewards mostly 0, values of both
    signs, all float32."""
    rng = np.random.default_rng(seed)
    games = int(rng.integers(1, 15))
    M = int(rng.integers(1, 250))
    played = rng.choice(games, size=int(rng.integers(1, games + 1)), replace=False)
    weights = rng.random(len(played)) ** 3 + 1e-3
    game = rng.choice(played, size=M, p=weights / weights.sum()).astype(np.int64)
    rewards = np.where(rng.random(M) < 0.3, rng.normal(0, 1, M), 0).astype(np.float32)
    values = rng.normal(0, 1, M).astype(np.float32)
    return rewards, values, game, int(game.max()) + 1 + seed % 3


def perturb(net: torch.nn.Module, seed: int) -> None:
    """Adds noise to every weight, so the critic's values are far from 0 and differ between
    positions. An untrained critic's are all about 0, where values, targets and advantages are
    hard to tell apart."""
    g = gen(seed)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(0.1 * torch.randn(p.shape, generator=g))


def tolerance(rewards, values) -> float:
    """An absolute tolerance for sums over a game's moves in float32."""
    return float(1e-6 * (1 + np.abs(rewards).sum() + np.abs(values).sum()))


# ---- step 1: td_errors ---------------------------------------------------------------------------

def deltas_of(rewards, values, game, games: int, gamma: float) -> np.ndarray:
    return gae.td_errors(np.asarray(rewards, np.float32), np.asarray(values, np.float32),
                         np.asarray(game, np.int64), games, gamma)


@steps.step(1)
@pytest.mark.parametrize("gamma,want", [
    (0.5, [0.625, 0.125, 1.375, 0.25]),  # 1 + .25/2 - .5; .75/2 - .25; 2 + .25/2 - .75; .5 - .25
    (1.0, [0.75, 0.5, 1.5, 0.25]),
    (0.0, [0.5, -0.25, 1.25, 0.25]),
])
def test_td_errors_of_one_game_by_hand(gamma, want):
    d = deltas_of([1, 0, 2, 0.5], [0.5, 0.25, 0.75, 0.25], [0, 0, 0, 0], 1, gamma)
    assert type(d) is np.ndarray and d.dtype == np.float32 and d.shape == (4,)
    np.testing.assert_allclose(d, want, rtol=1e-6, atol=1e-7)


@steps.step(1)
@pytest.mark.parametrize("rewards,values,game,games,want", [
    # Game 0 is moves 0, 2, 4 (.15 - .1; .25 - .3; 1 - .5), game 1 moves 1, 3 (.2 - .2; 1 - .4).
    ([0, 0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4, 0.5], [0, 1, 0, 1, 0], 2,
     [0.05, 0.0, -0.05, 0.6, 0.5]),
    # Games 1 and 4 have no moves (4 is past the last game that has any).
    ([0, 0, 0, 0, 1, 0.5, 1], [0.25, 0.125, 1.0, 0.75, 0.5, 0.25, 0.5], [2, 0, 2, 3, 0, 2, 3], 5,
     [0.25, 0.125, -0.875, -0.5, 0.5, 0.25, 0.5]),
], ids=["two-games", "games-without-moves"])
def test_td_errors_of_interleaved_games_by_hand(rewards, values, game, games, want):
    """gamma 0.5: each move's next move is the next one of its own game."""
    d = deltas_of(rewards, values, game, games, 0.5)
    assert type(d) is np.ndarray and d.dtype == np.float32 and d.shape == (len(game),)
    np.testing.assert_allclose(d, want, rtol=1e-6, atol=1e-7)


@steps.step(1)
@pytest.mark.parametrize("gamma", [1.0, 0.9, 0.0])
def test_td_errors_of_interleaved_games_are_each_computed_alone(gamma):
    game = np.array([1, 0, 1, 1, 2, 0, 2, 1, 0, 0, 1, 2])  # game 3 has no moves
    rng = np.random.default_rng(3)
    rewards = rng.random(len(game)).astype(np.float32)
    values = rng.normal(0, 1, len(game)).astype(np.float32)
    d = deltas_of(rewards, values, game, 4, gamma)
    for g in range(3):
        rows = np.flatnonzero(game == g)
        alone = deltas_of(rewards[rows], values[rows], np.zeros(len(rows)), 1, gamma)
        np.testing.assert_allclose(d[rows], alone, rtol=1e-6, atol=1e-7, err_msg=f"game {g}")


@steps.step(1)
def test_after_a_games_last_move_the_next_value_is_zero():
    """A game's last move has no next move, so its TD error is rewards - values whatever gamma
    is; every other move's grows by gamma * (the next move's value)."""
    rewards, values, game, games = random_games(7)
    last = np.array([np.flatnonzero(game == g)[-1] for g in np.unique(game)])
    assert len(last) >= 3
    rest = np.setdiff1d(np.arange(len(game)), last)
    at_zero = deltas_of(rewards, values, game, games, 0.0)
    for gamma in (0.3, 0.9, 1.0):
        d = deltas_of(rewards, values, game, games, gamma)
        np.testing.assert_allclose(d[last], (rewards - values)[last], rtol=1e-6, atol=1e-7,
                                   err_msg=f"gamma {gamma}")
        assert np.all(np.abs(d[rest] - at_zero[rest]) > 0), f"gamma {gamma}"


@steps.step(1)
@pytest.mark.parametrize("seed", [1, 2])
def test_gamma_zero_gives_rewards_minus_values(seed):
    rewards, values, game, games = random_games(seed)
    d = gae.td_errors(rewards, values, game, games, 0.0)
    np.testing.assert_allclose(d, rewards - values, rtol=1e-6, atol=1e-7)


@steps.step(1)
@pytest.mark.parametrize("seed", range(10))
def test_td_errors_match_the_definition(seed):
    """Random cases (random_games); gamma 0, 1 and in between."""
    rewards, values, game, games = random_games(200 + seed)
    gamma = [0.0, 1.0, 0.5, 0.99][seed] if seed < 4 else np.random.default_rng(seed).random()
    d = gae.td_errors(rewards, values, game, games, gamma)
    assert type(d) is np.ndarray and d.dtype == np.float32 and d.shape == (len(game),)
    np.testing.assert_allclose(d, definition_td_errors(rewards, values, game, games, gamma),
                               rtol=1e-5, atol=1e-6)


@steps.step(1)
@pytest.mark.parametrize("seed", range(3))
def test_with_gamma_one_a_games_td_errors_add_up_to_its_rewards_minus_its_first_value(seed):
    """Over a game's moves, rewards[m] + values[n] - values[m] telescopes: every value but the
    first move's cancels (the last move's next value is 0)."""
    rewards, values, game, games = random_games(300 + seed)
    d = gae.td_errors(rewards, values, game, games, 1.0)
    for g in np.unique(game):
        rows = np.flatnonzero(game == g)
        want = rewards[rows].astype(np.float64).sum() - values[rows[0]]
        assert d[rows].astype(np.float64).sum() == pytest.approx(
            want, abs=tolerance(rewards[rows], values[rows])), f"game {g}"


@steps.step(1)
def test_td_errors_of_no_moves_is_empty():
    d = deltas_of([], [], [], 3, 0.9)
    assert type(d) is np.ndarray and d.dtype == np.float32 and d.shape == (0,)


@steps.step(1)
def test_td_errors_accepts_float64_inputs():
    rewards = np.array([0.1, 0.0, 0.3, 0.7, 0.0, 0.2])
    values = np.array([0.5, -0.2, 0.4, 0.1, 0.3, 0.6])
    game = np.array([0, 1, 0, 1, 1, 0], np.int64)
    d = gae.td_errors(rewards, values, game, 2, 0.8)
    assert type(d) is np.ndarray and d.dtype == np.float32 and d.shape == (6,)
    np.testing.assert_allclose(d, definition_td_errors(rewards, values, game, 2, 0.8), rtol=1e-6,
                               atol=1e-7)


@steps.step(1)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_td_errors_leaves_its_inputs_alone(dtype):
    rewards = np.array([0.5, 0.0, 1.0, 0.25], dtype)
    values = np.array([0.3, -0.1, 0.6, 0.2], dtype)
    game = np.array([1, 0, 1, 0], np.int64)
    copies = [x.copy() for x in (rewards, values, game)]
    gae.td_errors(rewards, values, game, 2, 0.9)
    for name, x, y in zip(("rewards", "values", "game"), (rewards, values, game), copies):
        assert np.array_equal(x, y), f"td_errors changed its {name}"


# ---- step 2: gae_advantages ----------------------------------------------------------------------

def advantages_of(rewards, values, game, games: int, gamma: float, lam: float) -> np.ndarray:
    return gae.gae_advantages(np.asarray(rewards, np.float32), np.asarray(values, np.float32),
                              np.asarray(game, np.int64), games, gamma, lam)


@steps.step(2)
@pytest.mark.parametrize("lam,want", [
    (0.0, [0.05, 0.0, -0.05, 0.6, 0.5]),      # the TD errors
    (0.5, [0.06875, 0.15, 0.075, 0.6, 0.5]),  # gamma * lam = 1/4: .05 + .075/4; .6/4; -.05 + .5/4
    (1.0, [0.15, 0.3, 0.2, 0.6, 0.5]),        # G - values, G = [.25, .5, .5, 1, 1]
])
def test_gae_advantages_by_hand(lam, want):
    """Two interleaved games at gamma 0.5, whose TD errors are [.05, 0, -.05, .6, .5] (as in
    test_td_errors_of_interleaved_games_by_hand)."""
    A = advantages_of([0, 0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4, 0.5], [0, 1, 0, 1, 0], 2, 0.5, lam)
    assert type(A) is np.ndarray and A.dtype == np.float32 and A.shape == (5,)
    np.testing.assert_allclose(A, want, rtol=1e-6, atol=1e-7)


@steps.step(2)
@pytest.mark.parametrize("seed,gamma", [(0, 0.99), (1, 1.0), (2, 0.5)])
def test_lambda_zero_is_the_td_errors(seed, gamma):
    rewards, values, game, games = random_games(400 + seed)
    A = gae.gae_advantages(rewards, values, game, games, gamma, 0.0)
    np.testing.assert_allclose(A, definition_td_errors(rewards, values, game, games, gamma),
                               rtol=1e-6, atol=1e-6)


@steps.step(2)
@pytest.mark.parametrize("gamma", [1.0, 0.97, 0.5, 0.0])
def test_lambda_one_is_stage_3s_advantage(gamma):
    """The values cancel in pairs: A = G - values, G by discounted_returns' definition."""
    rewards, values, game, games = random_games(500)
    A = gae.gae_advantages(rewards, values, game, games, gamma, 1.0)
    want = definition_returns(rewards, game, games, gamma) - values
    np.testing.assert_allclose(A, want, rtol=1e-5, atol=tolerance(rewards, values))


@steps.step(2)
@pytest.mark.parametrize("seed", range(10))
def test_gae_advantages_match_the_definition(seed):
    """Random cases (random_games); gamma and lambda 0, 1 and in between."""
    rewards, values, game, games = random_games(600 + seed)
    pairs = [(0.99, 0.95), (1.0, 1.0), (1.0, 0.0), (0.0, 0.8), (0.5, 0.5)]
    gamma, lam = pairs[seed] if seed < 5 else np.random.default_rng(seed).random(2).tolist()
    A = gae.gae_advantages(rewards, values, game, games, gamma, lam)
    assert type(A) is np.ndarray and A.dtype == np.float32 and A.shape == (len(game),)
    np.testing.assert_allclose(A, definition_gae(rewards, values, game, games, gamma, lam),
                               rtol=1e-5, atol=tolerance(rewards, values))


@steps.step(2)
@pytest.mark.parametrize("gamma,lam", [(1.0, 0.95), (0.9, 0.5), (0.97, 0.0)])
def test_gae_advantages_of_interleaved_games_are_each_computed_alone(gamma, lam):
    game = np.array([1, 0, 1, 1, 2, 0, 2, 1, 0, 0, 1, 2])  # game 3 has no moves
    rng = np.random.default_rng(4)
    rewards = rng.random(len(game)).astype(np.float32)
    values = rng.normal(0, 1, len(game)).astype(np.float32)
    A = advantages_of(rewards, values, game, 4, gamma, lam)
    for g in range(3):
        rows = np.flatnonzero(game == g)
        alone = advantages_of(rewards[rows], values[rows], np.zeros(len(rows)), 1, gamma, lam)
        np.testing.assert_allclose(A[rows], alone, rtol=1e-6, atol=1e-6, err_msg=f"game {g}")


@steps.step(2)
@pytest.mark.parametrize("lam", [0.0, 0.5, 0.95, 1.0])
def test_a_critic_that_predicts_the_returns_exactly_gives_zero_advantages(lam):
    """values = G (discounted_returns' definition): every TD error is G[m] - G[m] = 0, so every
    advantage is 0 whatever lambda is: the critic already knew what would happen."""
    rewards, _, game, games = random_games(700)
    values = definition_returns(rewards, game, games, 0.9)
    A = gae.gae_advantages(rewards, values, game, games, 0.9, lam)
    np.testing.assert_allclose(A, 0, atol=1e-5)


@steps.step(2)
def test_gae_advantages_of_no_moves_is_empty():
    A = advantages_of([], [], [], 3, 0.9, 0.95)
    assert type(A) is np.ndarray and A.dtype == np.float32 and A.shape == (0,)


@steps.step(2)
def test_gae_advantages_accepts_float64_inputs():
    rewards = np.array([0.1, 0.0, 0.3, 0.7, 0.0, 0.2])
    values = np.array([0.5, -0.2, 0.4, 0.1, 0.3, 0.6])
    game = np.array([0, 1, 0, 1, 1, 0], np.int64)
    A = gae.gae_advantages(rewards, values, game, 2, 0.8, 0.9)
    assert type(A) is np.ndarray and A.dtype == np.float32 and A.shape == (6,)
    np.testing.assert_allclose(A, definition_gae(rewards, values, game, 2, 0.8, 0.9), rtol=1e-6,
                               atol=1e-7)


@steps.step(2)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_gae_advantages_leaves_its_inputs_alone(dtype):
    rewards = np.array([0.5, 0.0, 1.0, 0.25], dtype)
    values = np.array([0.3, -0.1, 0.6, 0.2], dtype)
    game = np.array([1, 0, 1, 0], np.int64)
    copies = [x.copy() for x in (rewards, values, game)]
    gae.gae_advantages(rewards, values, game, 2, 0.9, 0.8)
    for name, x, y in zip(("rewards", "values", "game"), (rewards, values, game), copies):
        assert np.array_equal(x, y), f"gae_advantages changed its {name}"


# ---- value_estimates (under `impl`) --------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic(positions):
    """61 moves at fixture positions over 9 games."""
    idx = np.random.default_rng(0).choice(len(positions), 61, replace=False)
    return position_episodes(positions, idx, games=9, seed=1)


@pytest.mark.parametrize("minibatch", [1, 7, 61, 100_000])
def test_value_estimates_are_the_critics_values(impl, synthetic, minibatch):
    """Whatever the chunking: logits_and_value's values of all the moves at once."""
    net = make_ac(2, hidden=32)
    with torch.no_grad():
        want = net.logits_and_value(torch.as_tensor(synthetic.obs))[1].numpy()
    got = value_estimates(net, synthetic, minibatch)
    assert type(got) is np.ndarray and got.dtype == np.float32 and got.shape == (61,)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)


def test_value_estimates_compute_no_gradient(impl, synthetic):
    """The net runs with gradient off (no graph is built for values nothing backpropagates
    through), on at most `minibatch` moves at a time; afterwards gradient is on again, no
    parameter has a .grad and the weights are as they were."""
    net = make_ac(3, hidden=32)
    before = copy.deepcopy(net.state_dict())
    calls = []
    logits_and_value = net.logits_and_value

    def spy(obs):
        calls.append((len(obs), torch.is_grad_enabled()))
        return logits_and_value(obs)

    net.logits_and_value = spy
    value_estimates(net, synthetic, 16)
    assert calls and sum(n for n, _ in calls) == 61 and all(n <= 16 for n, _ in calls), calls
    assert not any(grad for _, grad in calls), "the values were computed with gradient on"
    assert torch.is_grad_enabled(), "value_estimates left gradient off"
    for n, p in net.named_parameters():
        assert p.grad is None, f"{n} got a .grad"
    for k, v in net.state_dict().items():
        assert torch.equal(v, before[k]), f"{k} changed"


def test_value_estimates_of_no_moves_is_empty(impl):
    v = value_estimates(make_ac(4, hidden=16, layers=1), make_episodes([True, False], [10, 3]))
    assert type(v) is np.ndarray and v.dtype == np.float32 and v.shape == (0,)


@pytest.mark.parametrize("minibatch", [0, -1])
def test_value_estimates_rejects_a_minibatch_below_one(impl, synthetic, minibatch):
    with pytest.raises(ValueError):
        value_estimates(make_ac(5, hidden=16, layers=1), synthetic, minibatch)


# ---- Config --------------------------------------------------------------------------------------

def test_config_defaults_and_fields():
    """gae_lambda (0.95) comes after actor-critic's fields, whose defaults are unchanged."""
    c = Config()
    c.check()
    assert c.gae_lambda == 0.95
    ac_fields = [f.name for f in dataclasses.fields(actor_critic.Config)]
    assert [f.name for f in dataclasses.fields(Config)] == ac_fields + ["gae_lambda"]
    for name in ac_fields:
        assert getattr(c, name) == getattr(actor_critic.Config(), name), name
    assert c.gamma == 1.0 and c.value_coef == 0.5


@pytest.mark.parametrize("lam", [0, 0.0, 0.5, 1, 1.0])
def test_config_accepts_gae_lambda_in_zero_to_one(lam):
    Config(gae_lambda=lam).check()


@pytest.mark.parametrize("lam", [-1e-9, -1, 1 + 1e-9, 2, math.nan, math.inf, -math.inf, True,
                                 False, "0.5", None])
def test_config_rejects_gae_lambda(lam):
    with pytest.raises(ValueError) as exc:
        Config(gae_lambda=lam).check()
    assert "gae_lambda" in str(exc.value)


@pytest.mark.parametrize("changes,name", [
    ({"gamma": 1.5}, "gamma"), ({"gamma": math.nan}, "gamma"), ({"value_coef": -1}, "value_coef"),
    ({"lr": 0}, "lr"), ({"updates": 0}, "updates"), ({"hidden": True}, "hidden"),
    ({"entropy_coef": -0.1}, "entropy_coef"), ({"reward": "loss"}, "reward"),
    ({"opponent": "self"}, "opponent"), ({"minibatch": 2.0}, "minibatch")])
def test_config_still_checks_actor_critics_and_the_base_fields(changes, name):
    with pytest.raises(ValueError) as exc:
        Config(**changes).check()
    assert name in str(exc.value)


def test_config_from_dict_round_trips():
    for c in (Config(), Config(gae_lambda=0.8, gamma=0.9, value_coef=1.5, updates=7, lr=3e-4,
                               reward="vp", entropy_coef=0.01, opponent="greedy", seed=4,
                               hidden=64),
              Config(gae_lambda=0, gamma=0), Config(gae_lambda=1)):
        assert Trainer.config_from_dict(dataclasses.asdict(c)) == c
        # As stored in JSON, too.
        assert Trainer.config_from_dict(json.loads(json.dumps(dataclasses.asdict(c)))) == c


def test_config_from_dict_rejects_a_reinforce_config():
    with pytest.raises(ValueError) as exc:
        Trainer.config_from_dict(dataclasses.asdict(reinforce.Config()))
    assert "baseline" in str(exc.value)


# ---- Trainer: what needs no network --------------------------------------------------------------

@pytest.mark.parametrize("config", [actor_critic.Config(), reinforce.Config(), base.Config()],
                         ids=["actor-critic", "reinforce", "base"])
def test_trainer_needs_a_gae_config(tmp_path, config):
    """Exactly gae.Config: an actor_critic.Config has every field but gae_lambda, and is still
    refused."""
    with pytest.raises(TypeError):
        Trainer(config, tmp_path)


@pytest.mark.parametrize("changes", [{"gae_lambda": 1.5}, {"gae_lambda": -0.1},
                                     {"gae_lambda": math.nan}, {"gamma": 1.5},
                                     {"value_coef": -1}, {"lr": 0}, {"reward": "loss"}])
def test_trainer_rejects_an_invalid_config(tmp_path, changes):
    with pytest.raises(ValueError):
        Trainer(Config(**changes), tmp_path)


def test_saved_config_of_an_empty_directory_is_none(tmp_path):
    assert Trainer.saved_config(tmp_path) is None


# ---- Trainer (under `impl`) ----------------------------------------------------------------------

# Small and fast, as test_actor_critic.py's TINY. gamma < 1 makes the returns differ within each
# game, so the explained variance is never NaN, and gae_lambda is neither the default nor 1.
TINY = Config(updates=4, games_per_update=6, eval_games=4, eval_every=2, hidden=32,
              opponent="greedy", reward="vp", seed=6, minibatch=150, gamma=0.97,
              value_coef=0.5, entropy_coef=0.01, gae_lambda=0.9)


@pytest.mark.parametrize("other", ["reinforce", "actor-critic"])
def test_another_algorithms_run_directory_is_rejected(impl, tmp_path, other):
    """The Trainer and saved_config both refuse a reinforce or actor-critic checkpoint, naming
    both algorithms, and leave it alone."""
    if other == "reinforce":
        reinforce.Trainer(reinforce.Config(hidden=16, layers=1), tmp_path).save()
    else:
        actor_critic.Trainer(actor_critic.Config(hidden=16, layers=1), tmp_path).save()
    before = (tmp_path / "checkpoint.pt").read_bytes()
    for call in (lambda: Trainer(Config(hidden=16, layers=1), tmp_path),
                 lambda: Trainer.saved_config(tmp_path)):
        with pytest.raises(ValueError) as exc:
            call()
        assert other in str(exc.value) and "gae" in str(exc.value)
    assert (tmp_path / "checkpoint.pt").read_bytes() == before


def test_a_gae_run_directory_is_rejected_by_the_other_algorithms(impl, tmp_path):
    Trainer(Config(hidden=16, layers=1), tmp_path).save()
    before = (tmp_path / "checkpoint.pt").read_bytes()
    for other, call in (
            ("actor-critic", lambda: actor_critic.Trainer(actor_critic.Config(hidden=16, layers=1),
                                                          tmp_path)),
            ("actor-critic", lambda: actor_critic.Trainer.saved_config(tmp_path)),
            ("reinforce", lambda: reinforce.Trainer(reinforce.Config(), tmp_path)),
            ("reinforce", lambda: reinforce.Trainer.saved_config(tmp_path))):
        with pytest.raises(ValueError) as exc:
            call()
        assert other in str(exc.value) and "gae" in str(exc.value)
    assert (tmp_path / "checkpoint.pt").read_bytes() == before


@pytest.fixture(scope="module")
def straight_runs() -> dict:
    return {}


@pytest.fixture
def straight(impl, straight_runs, tmp_path_factory):
    """A complete fresh run of TINY under `impl` (one per impl, shared by the tests)."""
    if impl.name not in straight_runs:
        run_dir = tmp_path_factory.mktemp(f"straight-{impl.name}")
        trainer = Trainer(TINY, run_dir)
        log: list[str] = []
        trainer.run(log=log.append)
        straight_runs[impl.name] = SimpleNamespace(trainer=trainer, dir=run_dir, log=log,
                                                   metrics=read_metrics(run_dir))
    return straight_runs[impl.name]


def test_step_learns_from_the_games_it_played(impl, tmp_path, play_calls):
    """step() plays games_per_update recorded games against config.opponent on fresh seeds and
    takes one clipped Adam step on accumulate_gradients' gradient of those games toward the
    targets A + V: V the values of the net from before the step, A gae_advantages' definition
    of them. The metrics are trainer.Trainer's, actor-critic's (the explained variance of the
    returns G, not of the targets) and GAE's, all JSON-serializable. max_grad_norm is tiny so
    the clipping matters, and the net is perturbed so its values matter."""
    cfg = dataclasses.replace(TINY, games_per_update=5, lr=3e-3, max_grad_norm=1e-6,
                              minibatch=100, gamma=0.9, gae_lambda=0.8, value_coef=0.7,
                              entropy_coef=0.02)
    trainer = Trainer(cfg, tmp_path)
    perturb(trainer.net, 7)
    before = copy.deepcopy(trainer.net)
    metrics = trainer.step()
    assert trainer.update == 1 and metrics["update"] == 1
    (args, ep), = play_calls
    assert args["opponent"] == cfg.opponent and args["games"] == cfg.games_per_update
    assert args["record"] is True and args["first_seed"] >= cfg.train_first_seed
    assert ep.games == cfg.games_per_update and ep.moves > 0

    game_reward = game_rewards(ep, cfg.reward)
    rewards = final_move_rewards(ep, game_reward)
    with torch.no_grad():
        values = outputs(before, ep)[0].numpy()
    advantages = definition_gae(rewards, values, ep.game, ep.games, cfg.gamma, cfg.gae_lambda)
    targets = advantages + values
    want = reference(before, ep, targets, cfg.value_coef, cfg.entropy_coef)
    ref = copy.deepcopy(before)
    norm = adam_step(ref, want.grads, cfg)
    assert norm > 100 * cfg.max_grad_norm  # so the clipping mattered
    assert metrics["grad_norm"] == pytest.approx(norm, rel=1e-4)
    assert_same_step(trainer.net, ref, before, cfg.lr)

    keys = STEP_KEYS | AC_KEYS | GAE_KEYS
    assert keys <= set(metrics), f"missing {keys - set(metrics)}"
    assert metrics["games"] == cfg.games_per_update and metrics["moves"] == ep.moves
    returns = definition_returns(rewards, ep.game, ep.games, cfg.gamma)
    assert metrics["reward"] == pytest.approx(game_reward.mean(), rel=1e-6)
    assert metrics["return"] == pytest.approx(returns.mean(), rel=1e-5)
    assert metrics["target"] == pytest.approx(targets.astype(np.float64).mean(), rel=1e-4,
                                              abs=1e-6)
    assert metrics["advantage_std"] == pytest.approx(advantages.astype(np.float64).std(),
                                                     rel=1e-4)
    for k in ("loss", "policy_loss", "value_loss", "entropy", "log_prob"):
        assert metrics[k] == pytest.approx(getattr(want, k), rel=1e-4, abs=1e-6), k
    assert metrics["value"] == pytest.approx(values.astype(np.float64).mean(), rel=1e-4,
                                             abs=1e-6)
    of_returns = by_hand_explained_variance(returns, values)
    of_targets = by_hand_explained_variance(targets, values)
    assert abs(of_returns - of_targets) > 0.05  # so the test tells the two apart
    assert metrics["explained_variance"] == pytest.approx(of_returns, rel=1e-4, abs=1e-5)
    assert json.loads(json.dumps(metrics, allow_nan=False)) == metrics

    trainer.step()
    (second, _) = play_calls[1]
    assert second["first_seed"] >= args["first_seed"] + cfg.games_per_update
    assert second["games"] == cfg.games_per_update and second["record"] is True


def test_run_writes_the_metrics_lines_and_a_checkpoint(impl, straight):
    lines = straight.metrics
    assert [line["update"] for line in lines] == list(range(TINY.updates + 1))
    assert EVAL_KEYS <= set(lines[0]) and not ((AC_KEYS | GAE_KEYS) & set(lines[0]))
    evals = eval_updates(TINY)
    for line in lines[1:]:
        u = line["update"]
        assert STEP_KEYS | AC_KEYS | GAE_KEYS <= set(line), f"update {u} misses some keys"
        assert (EVAL_KEYS <= set(line)) == (u in evals), f"update {u}"
        assert line["games"] == TINY.games_per_update
        for k, v in line.items():
            assert isinstance(v, (int, float)) and math.isfinite(v), (u, k, v)
    checkpoint = torch.load(straight.dir / "checkpoint.pt", weights_only=True)
    assert checkpoint["algorithm"] == "gae"
    assert checkpoint["update"] == TINY.updates
    assert Trainer.config_from_dict(checkpoint["config"]) == TINY
    assert Trainer.saved_config(straight.dir) == TINY


def test_run_logs_value_loss_and_explained_variance(impl, straight):
    """One log_line per update (with the critic's value loss and explained variance) and one
    line per evaluation."""
    assert len(straight.log) == TINY.updates + len(eval_updates(TINY))
    update_lines = [straight.trainer.log_line(m) for m in straight.metrics[1:]]
    assert [s for s in straight.log if s in update_lines] == update_lines
    for s in update_lines:
        assert "value loss" in s and "explained" in s and "\n" not in s


def test_two_fresh_runs_are_identical(impl, straight, tmp_path):
    trainer = Trainer(TINY, tmp_path)
    trainer.run(log=None)
    assert_same_weights(trainer.net, straight.trainer.net)
    assert_same_optimizer(trainer.optimizer, straight.trainer.optimizer)
    assert [untimed(x) for x in read_metrics(tmp_path)] == [untimed(x) for x in straight.metrics]


@pytest.mark.parametrize("stop", [1, 2])
def test_stop_and_resume_equals_never_stopping(impl, straight, tmp_path, stop):
    """Run `stop` updates, then a new Trainer on the same directory with all the updates: the
    same weights, optimizer state and metrics lines (but timing, and the extra evaluation the
    stopped run made after its last update) as the run that never stopped."""
    Trainer(dataclasses.replace(TINY, updates=stop), tmp_path).run(log=None)
    resumed = Trainer(TINY, tmp_path)
    assert resumed.update == stop and resumed.config == TINY
    resumed.run(log=None)
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


@pytest.mark.parametrize("changes", [{"gae_lambda": 0.95}, {"gae_lambda": 1.0},
                                     {"gae_lambda": 0.5, "gamma": 1.0}])
def test_resume_rejects_a_different_gae_lambda(impl, tmp_path, changes):
    Trainer(TINY, tmp_path).save()
    with pytest.raises(ValueError) as exc:
        Trainer(dataclasses.replace(TINY, **changes), tmp_path)
    for name in changes:
        assert name in str(exc.value)


def test_resume_allows_more_updates(impl, tmp_path):
    first = Trainer(dataclasses.replace(TINY, updates=1), tmp_path)
    first.step()
    first.save()
    more = Trainer(dataclasses.replace(TINY, updates=9), tmp_path)
    assert more.update == 1 and more.config == dataclasses.replace(TINY, updates=9)
    assert_same_weights(more.net, first.net)
    assert_same_optimizer(more.optimizer, first.optimizer)


# ---- every step written --------------------------------------------------------------------------

def documented_gae_loss(net, ep, rewards, cfg: Config) -> SimpleNamespace:
    """The module docstring's loss on all the moves at once, computed here with none of your
    code but the net: the values V of the net as it is, held fixed; the TD errors and the
    advantages A by their definitions; the critic trained toward A + V with stage 3's loss. Its
    parts in numpy, and its gradient pushed through a copy of `net` by autograd from the
    gradient with respect to the net's three outputs, worked out by hand: -A / M for each
    log-prob, value_coef * 2 * (value - (A + V)) / M for each value and -entropy_coef / M for
    each entropy."""
    net = fresh_copy(net)
    value, log_prob, entropy = outputs(net, ep)
    M = ep.moves
    V = value.detach().numpy()
    A = definition_gae(rewards, V, ep.game, ep.games, cfg.gamma, cfg.gae_lambda)
    A_t, T_t = torch.as_tensor(A), torch.as_tensor(A + V)
    torch.autograd.backward(
        [log_prob, value, entropy],
        [-A_t / M, cfg.value_coef * 2 * (value.detach() - T_t) / M,
         torch.full_like(entropy, -cfg.entropy_coef / M)])
    A64, V64 = A.astype(np.float64), V.astype(np.float64)
    lp, h = log_prob.detach().double().numpy(), entropy.detach().double().numpy()
    policy = -np.mean(A64 * lp)
    value_loss = np.mean((V64 - (A64 + V64)) ** 2)
    loss = policy + cfg.value_coef * value_loss - cfg.entropy_coef * h.mean()
    return SimpleNamespace(
        grads=grads_of(net), loss=loss, policy_loss=policy, value_loss=value_loss,
        entropy=h.mean(), log_prob=lp.mean(), values=V64, advantages=A64, targets=A64 + V64)


@steps.all_steps
def test_step_is_one_clipped_adam_step_on_the_documented_loss(tmp_path, play_calls):
    """Reproduces step() from the episodes it played, independently of your code: rewards,
    the values from before the step, TD errors and advantages by their definitions, the
    documented loss's gradient (see documented_gae_loss), its norm, clipping, one Adam step;
    and the metrics. The net is perturbed first so its values matter."""
    cfg = Config(updates=1, games_per_update=5, eval_games=4, hidden=32, opponent="greedy",
                 seed=11, lr=3e-3, max_grad_norm=1e-6, minibatch=100, reward="vp",
                 entropy_coef=0.01, gamma=0.95, value_coef=0.7, gae_lambda=0.7)
    trainer = Trainer(cfg, tmp_path)
    perturb(trainer.net, 8)
    before = copy.deepcopy(trainer.net)
    metrics = trainer.step()
    (args, ep), = play_calls
    assert args["games"] == 5 and args["record"] is True and args["opponent"] == "greedy"
    # The moves were sampled from the net as it was before the step.
    np.testing.assert_allclose(ep.log_prob, log_probs(before, ep).numpy(), rtol=1e-5, atol=1e-5)

    game_reward = np.minimum(ep.vp, 10) / 10
    rewards = final_move_rewards(ep, game_reward.astype(np.float32))
    want = documented_gae_loss(before, ep, rewards, cfg)
    ref = copy.deepcopy(before)
    norm = adam_step(ref, want.grads, cfg)
    assert metrics["grad_norm"] == pytest.approx(norm, rel=1e-4)
    assert norm > 100 * cfg.max_grad_norm  # so the clipping mattered
    assert_same_step(trainer.net, ref, before, cfg.lr)

    returns = definition_returns(rewards, ep.game, ep.games, cfg.gamma)
    assert metrics["reward"] == pytest.approx(game_reward.mean(), rel=1e-5)
    assert metrics["return"] == pytest.approx(returns.mean(), rel=1e-5)
    assert metrics["target"] == pytest.approx(want.targets.mean(), rel=1e-4, abs=1e-6)
    assert metrics["advantage_std"] == pytest.approx(want.advantages.std(), rel=1e-4)
    for k in ("loss", "policy_loss", "value_loss", "entropy", "log_prob"):
        assert metrics[k] == pytest.approx(getattr(want, k), rel=1e-3, abs=1e-6), k
    assert metrics["value"] == pytest.approx(want.values.mean(), rel=1e-4, abs=1e-6)
    assert metrics["explained_variance"] == pytest.approx(
        by_hand_explained_variance(returns, want.values), rel=1e-4, abs=1e-5)


@steps.all_steps
def test_with_lambda_one_the_step_is_stage_3s(tmp_path, play_calls):
    """With gae_lambda 1 the advantages are G - value (the values cancel in pairs) and the
    targets G: a gae Trainer takes stage 3's step. Under the same settings an actor-critic
    Trainer starts from the same net and plays the same games."""
    common = dict(updates=1, games_per_update=5, eval_games=4, hidden=32, opponent="greedy",
                  seed=12, lr=3e-3, max_grad_norm=1e-6, minibatch=100, reward="vp",
                  entropy_coef=0.01, gamma=0.95, value_coef=0.7)
    ours = Trainer(Config(**common, gae_lambda=1.0), tmp_path / "gae")
    theirs = actor_critic.Trainer(actor_critic.Config(**common), tmp_path / "ac")
    assert_same_weights(ours.net, theirs.net)
    before = copy.deepcopy(theirs.net)
    got, want = ours.step(), theirs.step()
    (_, ep), (_, ep_ac) = play_calls
    assert np.array_equal(ep.types, ep_ac.types) and np.array_equal(ep.args, ep_ac.args)
    assert np.array_equal(ep.game, ep_ac.game) and np.array_equal(ep.vp, ep_ac.vp)
    assert_same_step(ours.net, theirs.net, before, common["lr"])
    for k in ("reward", "return", "loss", "policy_loss", "value_loss", "entropy", "log_prob",
              "value", "explained_variance", "grad_norm"):
        assert got[k] == pytest.approx(want[k], rel=1e-4, abs=1e-6), k
    assert got["target"] == pytest.approx(want["return"], rel=1e-4, abs=1e-6)


@pytest.fixture(scope="module")
def recorded():
    """Real episodes: an untrained PolicyNet (hidden 32) recorded against greedy."""
    net = make_net(21, hidden=32)
    return play(Agent(net, generator=gen(22)), "greedy", games=6, first_seed=5001,
                generator=gen(23), record=True)


@steps.all_steps
def test_the_critic_learns_the_returns_of_a_fixed_batch(recorded, tmp_path):
    """Adam steps of learn() on one recorded batch, each from targets recomputed from the
    critic's current values: bootstrapping from itself, the critic still ends up explaining
    most of the variance of the returns G (where every TD error, and so every advantage, is
    0), and its loss against G falls to a small fraction of where it started."""
    cfg = Config(hidden=32, reward="vp", gamma=0.95, gae_lambda=0.5, minibatch=128)
    trainer = Trainer(cfg, tmp_path)
    ep = recorded
    returns = definition_returns(final_move_rewards(ep, game_rewards(ep, "vp")), ep.game,
                                 ep.games, cfg.gamma)
    assert np.var(returns) > 0
    start = value_estimates(trainer.net, ep)
    for _ in range(60):
        trainer.optimizer.zero_grad()
        trainer.learn(ep)
        trainer.optimizer.step()
    end = value_estimates(trainer.net, ep)
    loss_start = np.mean((start - returns) ** 2)
    loss_end = np.mean((end - returns) ** 2)
    assert loss_end < 0.1 * loss_start, (loss_start, loss_end)
    assert actor_critic.explained_variance(returns, end) > 0.5


@steps.all_steps
def test_a_short_run_against_random_stays_finite(tmp_path):
    cfg = Config(updates=3, games_per_update=4, eval_games=4, eval_every=3, hidden=32,
                 opponent="random", reward="vp", gamma=0.99, seed=1, minibatch=1024)
    trainer = Trainer(cfg, tmp_path)
    trainer.run(log=None)
    lines = read_metrics(tmp_path)
    assert [x["update"] for x in lines] == [0, 1, 2, 3]
    for line in lines:
        for k, v in line.items():
            assert math.isfinite(v), (line["update"], k, v)
    for k, v in trainer.net.state_dict().items():
        assert torch.isfinite(v).all(), k
