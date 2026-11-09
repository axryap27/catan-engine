"""Tests for catan_rl.actor_critic (stage 3), written from the contracts.

Stage 3's three steps are yours to write: ActorCriticNet (step 1), discounted_returns (step 2)
and actor_critic_loss (step 3). Each step's tests are skipped until you write it (see
lab_steps.py), and then they run.

Everything else in the module is tested now. The plumbing (accumulate_gradients, the Trainer)
runs twice, under the `impl` fixture: "stand-in" swaps in the deliberately wrong stand-ins of
ac_helpers.py for the three steps, so the loop is tested before you write anything, and "yours"
uses the module's own once all three are written. A few tests then need your code alone: step()
is one clipped Adam step on the documented loss, and the critic and the actor learn.
"""

import copy
import dataclasses
import io
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ac_helpers import (adam_step, assert_grads_close, assert_same_optimizer, assert_same_step,
                        assert_same_weights, by_hand_explained_variance, definition_returns,
                        documented_loss, eval_updates, gen, grads_of, log_probs, make_ac,
                        make_episodes, make_net, num, outputs, play_calls, position_episodes,
                        read_metrics, reference, untimed, use_stage3_stand_ins)
from catan_rl import actions, actor_critic, reinforce
from catan_rl import trainer as base
from catan_rl.actor_critic import (Config, Losses, Trainer, accumulate_gradients,
                                   explained_variance, final_move_rewards)
from catan_rl.engine import ce
from catan_rl.policy import Agent
from catan_rl.rollout import Episodes, play
from catan_rl.trainer import game_rewards
from lab_steps import STAGE3 as steps

AT = ce.ActionType

# trainer.Trainer.step()'s metrics (as in test_reinforce.py), then what actor-critic adds.
STEP_KEYS = {"update", "games", "moves", "win_rate", "mean_vp", "mean_turns", "capped", "reward",
             "loss", "log_prob", "entropy", "grad_norm", "rollout_seconds", "update_seconds"}
AC_KEYS = {"return", "policy_loss", "value_loss", "value", "explained_variance"}
EVAL_KEYS = {"eval_win_rate", "eval_ci_low", "eval_ci_high", "eval_games"}
# accumulate_gradients' statistics.
STAT_KEYS = ("loss", "policy_loss", "value_loss", "entropy", "log_prob", "value",
             "explained_variance")


@pytest.fixture(params=["stand-in", pytest.param("yours", marks=steps.all_steps)])
def impl(request, monkeypatch):
    """Which three steps actor_critic's plumbing runs on: the stand-ins, or yours."""
    if request.param == "stand-in":
        use_stage3_stand_ins(monkeypatch)
    return SimpleNamespace(name=request.param)


# ---- step 1: ActorCriticNet ----------------------------------------------------------------------

@steps.step(1)
@pytest.mark.parametrize("batch", [1, 5])
@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32], ids=["uint8", "float32"])
def test_logits_and_value_shapes_and_dtypes(positions, batch, dtype):
    net = make_ac(0, hidden=32)
    obs = torch.as_tensor(positions.obs[:batch]).to(dtype)
    logits, value = net.logits_and_value(obs)
    assert isinstance(logits, torch.Tensor) and isinstance(value, torch.Tensor)
    assert logits.shape == (batch, ce.MASK_SIZE) and logits.dtype == torch.float32
    assert value.shape == (batch,), f"value must be (B,), got {tuple(value.shape)}"
    assert value.dtype == torch.float32


@steps.step(1)
def test_uint8_and_float_obs_give_the_same_outputs(positions):
    net = make_ac(1, hidden=32)
    obs = torch.as_tensor(positions.obs)
    with torch.no_grad():
        u8 = net.logits_and_value(obs)
        f32 = net.logits_and_value(obs.float())
    for x, y in zip(u8, f32):
        torch.testing.assert_close(x, y, rtol=1e-6, atol=1e-6)


@steps.step(1)
def test_logits_are_exactly_forward(positions):
    """forward(obs) is still PolicyNet's logits (an Agent plays with it), and logits_and_value's
    logits equal it."""
    net = make_ac(2, hidden=32)
    obs = torch.as_tensor(positions.obs)
    with torch.no_grad():
        logits, _ = net.logits_and_value(obs)
        assert torch.equal(logits, net(obs))


@steps.step(1)
@pytest.mark.parametrize("hidden,layers", [(None, None), (8, 1), (32, 2), (48, 3)])
@pytest.mark.parametrize("seed", [0, 7])
def test_the_actor_is_a_policy_net_built_under_the_same_seed(positions, hidden, layers, seed):
    """Built under a torch seed, the actor (every PolicyNet entry: body.*, head.*) starts exactly
    as a PolicyNet built under that seed, so the two start with the same policy."""
    kwargs = {} if hidden is None else {"hidden": hidden, "layers": layers}
    policy = make_net(seed, **kwargs)
    net = make_ac(seed, **kwargs)
    want, got = policy.state_dict(), net.state_dict()
    assert not set(want) - set(got), f"missing actor entries: {sorted(set(want) - set(got))}"
    for k in want:
        assert got[k].shape == want[k].shape and torch.equal(got[k], want[k]), (
            f"{k} differs from a PolicyNet's: the actor's layers must stay exactly as "
            "super().__init__() made them (not re-initialized, re-seeded or replaced)")
    obs = torch.as_tensor(positions.obs)
    with torch.no_grad():
        logits, value = net.logits_and_value(obs)
        assert torch.equal(logits, policy(obs))
    assert value.shape == (len(positions),)


@steps.step(1)
def test_an_untrained_net_still_plays_like_the_random_bot(positions):
    """As test_policy.py checks for PolicyNet: the logits are about 0, so each legal move's
    log-prob and each position's entropy are close to the zero-logit (random-bot) ones."""
    net = make_ac(0)
    with torch.no_grad():
        logits, _ = net.logits_and_value(torch.as_tensor(positions.obs))
    assert logits.abs().max().item() < 0.1
    mask = torch.from_numpy(positions.mask)
    phases = torch.from_numpy(positions.phases)
    rows = torch.tensor([i for i, moves in enumerate(positions.legal) for _ in moves])
    types = torch.tensor([t for moves in positions.legal for t, _ in moves])
    args = torch.tensor([a for moves in positions.legal for _, a in moves])
    lp_net, h_net = actions.log_prob_entropy(logits[rows], mask[rows], phases[rows], types, args)
    lp_zero, h_zero = actions.log_prob_entropy(
        torch.zeros(len(rows), ce.MASK_SIZE), mask[rows], phases[rows], types, args)
    assert (lp_net - lp_zero).abs().max().item() < 0.4
    assert (h_net - h_zero).abs().max().item() < 0.2


@steps.step(1)
def test_the_value_depends_on_the_observation(positions):
    """Different positions get different values. Every parameter is randomized first, so a
    critic whose output layer starts at zero (a fine choice) shows it too."""
    net = make_ac(3, hidden=32)
    g = gen(4)
    with torch.no_grad():
        for p in net.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.3)
        _, value = net.logits_and_value(torch.as_tensor(positions.obs))
    assert torch.isfinite(value).all()
    assert len(torch.unique(value)) > len(positions) // 2, "the value ignores the observation"


@steps.step(1)
def test_the_value_trains_the_critic_and_never_the_actors_head(positions):
    """value.sum().backward() reaches some parameter outside head.* (the critic's own layers,
    and the body if it is shared) and leaves head.* alone: the actor's head is the policy's."""
    net = make_ac(5, hidden=32)
    _, value = net.logits_and_value(torch.as_tensor(positions.obs))
    value.sum().backward()
    params = dict(net.named_parameters())
    for n in ("head.weight", "head.bias"):
        p = params[n]
        assert p.grad is None or torch.count_nonzero(p.grad) == 0, f"the value moves {n}"
    reached = [n for n, p in params.items() if not n.startswith("head.")
               and p.grad is not None and torch.count_nonzero(p.grad) > 0]
    assert reached, "the value has no gradient to any parameter: the critic cannot learn"


@steps.step(1)
def test_the_logits_never_train_the_critic(positions):
    """logits.sum().backward() gives no gradient to any parameter a PolicyNet doesn't have."""
    net = make_ac(6, hidden=32)
    actor = set(make_net(6, hidden=32).state_dict())
    critic = [n for n, _ in net.named_parameters() if n not in actor]
    assert critic, "the critic has no parameters of its own"
    logits, _ = net.logits_and_value(torch.as_tensor(positions.obs))
    logits.sum().backward()
    for n, p in net.named_parameters():
        if n not in actor:
            assert p.grad is None or torch.count_nonzero(p.grad) == 0, f"the logits move {n}"


@steps.step(1)
def test_an_agent_plays_whole_games_with_it():
    """forward is PolicyNet's, so an Agent plays with the net unchanged. The engine refuses an
    illegal move (ValueError), so finishing the games means every move was legal."""
    net = make_ac(7, hidden=16, layers=1)
    ep = play(Agent(net, generator=gen(8)), "random", games=3, first_seed=4242,
              generator=gen(9), record=True)
    assert ep.games == 3 and ep.moves > 0 and (ep.turns > 0).all()
    assert np.isfinite(ep.log_prob).all() and (ep.log_prob > -100).all()
    # The recorded log-probs are those of logits_and_value's logits.
    np.testing.assert_allclose(ep.log_prob, log_probs(net, ep).numpy(), rtol=1e-5, atol=1e-5)


@steps.step(1)
def test_state_dict_round_trip(positions):
    """A checkpoint's net (torch.save, then torch.load with weights_only=True, as the Trainer
    does) loaded into a net built under another seed gives the same outputs."""
    a = make_ac(10, hidden=32)
    b = make_ac(11, hidden=32)
    buffer = io.BytesIO()
    torch.save(a.state_dict(), buffer)
    buffer.seek(0)
    b.load_state_dict(torch.load(buffer, weights_only=True))
    obs = torch.as_tensor(positions.obs)
    with torch.no_grad():
        for x, y in zip(a.logits_and_value(obs), b.logits_and_value(obs)):
            assert torch.equal(x, y)


@steps.step(1)
@pytest.mark.parametrize("hidden,layers", [(16, 1), (48, 3)])
def test_outputs_are_finite_at_every_position(positions, hidden, layers):
    net = make_ac(12, hidden=hidden, layers=layers)
    obs = torch.as_tensor(positions.obs)
    before = obs.clone()
    with torch.no_grad():
        logits, value = net.logits_and_value(obs)
    assert logits.shape == (len(positions), ce.MASK_SIZE) and value.shape == (len(positions),)
    assert torch.isfinite(logits).all() and torch.isfinite(value).all()
    assert torch.equal(obs, before)


# ---- step 2: discounted_returns ------------------------------------------------------------------

def returns_of(rewards, game, games: int, gamma: float) -> np.ndarray:
    return actor_critic.discounted_returns(np.asarray(rewards, np.float32),
                                           np.asarray(game, np.int64), games, gamma)


@steps.step(2)
@pytest.mark.parametrize("gamma,want", [
    (1.0, [7, 6, 6, 4, 4]),
    (0.5, [1.75, 1.5, 3, 2, 4]),  # 4; 0 + 4/2; 2 + 2/2; 0 + 3/2; 1 + 1.5/2
    (0.0, [1, 0, 2, 0, 4]),
])
def test_discounted_returns_of_one_game_by_hand(gamma, want):
    G = returns_of([1, 0, 2, 0, 4], [0, 0, 0, 0, 0], 1, gamma)
    assert type(G) is np.ndarray and G.dtype == np.float32 and G.shape == (5,)
    np.testing.assert_allclose(G, want, rtol=1e-6)


@steps.step(2)
@pytest.mark.parametrize("gamma", [1.0, 0.9, 0.0])
def test_interleaved_games_are_each_computed_alone(gamma):
    """The games' moves are interleaved; each game's returns are those of its moves alone."""
    game = np.array([1, 0, 1, 1, 2, 0, 2, 1, 0, 0, 1, 2])  # game 3 has no moves
    rewards = np.random.default_rng(3).random(len(game)).astype(np.float32)
    G = returns_of(rewards, game, 4, gamma)
    assert G.shape == (len(game),) and G.dtype == np.float32
    for g in range(3):
        rows = np.flatnonzero(game == g)
        alone = returns_of(rewards[rows], np.zeros(len(rows)), 1, gamma)
        np.testing.assert_allclose(G[rows], alone, rtol=1e-6, atol=1e-7, err_msg=f"game {g}")


@steps.step(2)
@pytest.mark.parametrize("gamma", [1.0, 0.97, 0.5, 0.0])
def test_a_final_reward_is_discounted_once_per_later_move(gamma):
    """The reward after each game's last move only (final_move_rewards, as the Trainer does):
    move m's return is gamma^k * R, k = its game's moves after m."""
    ep = make_episodes(won=[True, False, False, False, True], vp=[10, 5, 2, 7, 10],
                       game=[0, 1, 0, 2, 1, 0, 0, 2, 1, 0, 4])  # game 3 has no moves
    R = np.array([1.0, 0.5, 0.25, 0.75, 0.125], np.float32)
    G = actor_critic.discounted_returns(final_move_rewards(ep, R), ep.game, ep.games, gamma)
    for m in range(ep.moves):
        k = int(np.sum(ep.game[m + 1:] == ep.game[m]))
        assert G[m] == pytest.approx(gamma ** k * R[ep.game[m]], rel=1e-6, abs=1e-12), m


@steps.step(2)
@pytest.mark.parametrize("seed", range(10))
def test_discounted_returns_match_the_definition(seed):
    """Random cases: many games of uneven lengths, some (in the middle and past the last game
    that has moves) with no moves at all; gamma 0, 1 and in between."""
    rng = np.random.default_rng(100 + seed)
    games = int(rng.integers(1, 15))
    M = int(rng.integers(1, 250))
    played = rng.choice(games, size=int(rng.integers(1, games + 1)), replace=False)
    weights = rng.random(len(played)) ** 3 + 1e-3
    game = rng.choice(played, size=M, p=weights / weights.sum())
    rewards = np.where(rng.random(M) < 0.3, rng.normal(0, 1, M), 0).astype(np.float32)
    gamma = [0.0, 1.0, 0.5, 0.99][seed] if seed < 4 else float(rng.random())
    total_games = int(game.max()) + 1 + seed % 3
    G = actor_critic.discounted_returns(rewards, game.astype(np.int64), total_games, gamma)
    want = definition_returns(rewards, game, total_games, gamma)
    assert type(G) is np.ndarray and G.dtype == np.float32 and G.shape == (M,)
    np.testing.assert_allclose(G, want, rtol=1e-5, atol=1e-6 * (1 + np.abs(rewards).sum()))


@steps.step(2)
def test_discounted_returns_of_no_moves_is_empty():
    G = returns_of([], [], 3, 0.9)
    assert type(G) is np.ndarray and G.dtype == np.float32 and G.shape == (0,)


@steps.step(2)
def test_discounted_returns_accepts_float64_rewards():
    rewards = np.array([0.1, 0.0, 0.3, 0.7, 0.0, 0.2])
    game = np.array([0, 1, 0, 1, 1, 0], np.int64)
    G = actor_critic.discounted_returns(rewards, game, 2, 0.8)
    assert G.dtype == np.float32 and G.shape == (6,)
    np.testing.assert_allclose(G, definition_returns(rewards, game, 2, 0.8), rtol=1e-6)


@steps.step(2)
def test_discounted_returns_leaves_its_inputs_alone():
    rewards = np.array([0.5, 0.0, 1.0, 0.25], np.float32)
    game = np.array([1, 0, 1, 0], np.int64)
    r, g = rewards.copy(), game.copy()
    actor_critic.discounted_returns(rewards, game, 2, 0.9)
    assert np.array_equal(rewards, r) and np.array_equal(game, g)


# ---- step 3: actor_critic_loss -------------------------------------------------------------------

def loss_inputs(B: int = 7, seed: int = 0):
    """Leaf tensors (log_prob, entropy, value) with gradient, and returns (B,) without."""
    g = gen(seed)
    log_prob = (-3 * torch.rand(B, generator=g)).requires_grad_()
    entropy = (2 * torch.rand(B, generator=g)).requires_grad_()
    value = (0.5 * torch.randn(B, generator=g)).requires_grad_()
    returns = torch.rand(B, generator=g)
    return log_prob, entropy, value, returns


def as_float64(*tensors) -> list[np.ndarray]:
    return [t.detach().double().numpy() for t in tensors]


@steps.step(3)
@pytest.mark.parametrize("value_coef,entropy_coef", [(0.5, 0.0), (0.0, 0.1), (2.0, 0.01)])
def test_losses_are_the_documented_means(value_coef, entropy_coef):
    lp, h, v, G = loss_inputs()
    losses = actor_critic.actor_critic_loss(lp, h, v, G, value_coef, entropy_coef)
    assert isinstance(losses, Losses)
    for name in ("total", "policy", "value", "entropy"):
        x = getattr(losses, name)
        assert isinstance(x, torch.Tensor) and x.shape == (), f"{name} must be a scalar tensor"
    lp_, h_, v_, G_ = as_float64(lp, h, v, G)
    policy = -np.mean((G_ - v_) * lp_)
    value = np.mean((v_ - G_) ** 2)
    entropy = np.mean(h_)
    assert num(losses.policy) == pytest.approx(policy, rel=1e-5, abs=1e-7)
    assert num(losses.value) == pytest.approx(value, rel=1e-5, abs=1e-7)
    assert num(losses.entropy) == pytest.approx(entropy, rel=1e-5, abs=1e-7)
    assert num(losses.total) == pytest.approx(
        policy + value_coef * value - entropy_coef * entropy, rel=1e-5, abs=1e-7)


@steps.step(3)
def test_losses_are_means_not_sums():
    """Every part is a mean over the moves: the same moves twice give the same losses."""
    lp, h, v, G = loss_inputs(B=5, seed=1)
    once = actor_critic.actor_critic_loss(lp, h, v, G, 0.5, 0.01)
    twice = actor_critic.actor_critic_loss(*(torch.cat([x, x]) for x in (lp, h, v, G)), 0.5,
                                           0.01)
    for name in ("total", "policy", "value", "entropy"):
        assert num(getattr(twice, name)) == pytest.approx(num(getattr(once, name)), rel=1e-5,
                                                          abs=1e-7), name


@steps.step(3)
@pytest.mark.parametrize("value_coef,entropy_coef", [(0.5, 0.0), (2.0, 0.05)])
def test_the_gradient_with_respect_to_log_prob_and_entropy(value_coef, entropy_coef):
    """d total / d log_prob = -(returns - value) / B and d total / d entropy = -entropy_coef / B."""
    B = 7
    lp, h, v, G = loss_inputs(B, seed=2)
    actor_critic.actor_critic_loss(lp, h, v, G, value_coef, entropy_coef).total.backward()
    torch.testing.assert_close(lp.grad, -(G - v.detach()) / B, rtol=1e-5, atol=1e-7)
    if entropy_coef == 0:
        assert h.grad is None or torch.count_nonzero(h.grad) == 0
    else:
        torch.testing.assert_close(h.grad, torch.full((B,), -entropy_coef / B), rtol=1e-5,
                                   atol=1e-8)


@steps.step(3)
@pytest.mark.parametrize("value_coef", [0.5, 2.0])
def test_the_advantage_is_a_constant_so_the_policy_term_does_not_train_the_critic(value_coef):
    """d total / d value is exactly value_coef * 2 * (value - returns) / B: the value term's
    gradient and nothing else. A policy term whose advantage still carries the value's gradient
    adds log_prob / B here, and pushes the critic to whatever makes the policy loss small
    instead of to the returns."""
    B = 7
    lp, h, v, G = loss_inputs(B, seed=3)
    actor_critic.actor_critic_loss(lp, h, v, G, value_coef, 0.01).total.backward()
    torch.testing.assert_close(v.grad, value_coef * 2 * (v.detach() - G) / B, rtol=1e-5,
                               atol=1e-7)


@steps.step(3)
def test_value_coef_zero_gives_the_value_no_gradient():
    lp, h, v, G = loss_inputs(seed=4)
    actor_critic.actor_critic_loss(lp, h, v, G, 0.0, 0.01).total.backward()
    assert v.grad is None or torch.count_nonzero(v.grad) == 0, (
        "with value_coef 0 nothing should train the critic")
    assert lp.grad is not None and torch.count_nonzero(lp.grad) > 0


@steps.step(3)
def test_value_equal_to_the_returns_gives_no_policy_gradient():
    """Advantage 0 everywhere: the policy term has no gradient (only the entropy term would)."""
    lp, h, v, G = loss_inputs(seed=5)
    v = G.clone().requires_grad_()
    losses = actor_critic.actor_critic_loss(lp, h, v, G, 0.5, 0.0)
    losses.total.backward()
    assert lp.grad is None or torch.count_nonzero(lp.grad) == 0
    assert v.grad is None or torch.count_nonzero(v.grad) == 0
    assert num(losses.policy) == 0 and num(losses.value) == 0


@steps.step(3)
def test_a_descent_step_moves_each_log_prob_with_its_advantage():
    """One gradient-descent step on free log-probs raises those whose return beat the value
    (advantage > 0) and lowers those whose return fell short."""
    B = 8
    G = torch.tensor([1.0, 0.0, 0.6, 0.2, 1.0, 0.0, 0.9, 0.1])
    v = torch.tensor([0.4, 0.5, 0.1, 0.7, 0.9, 0.2, 0.3, 0.6], requires_grad=True)
    lp = torch.full((B,), -2.0, requires_grad=True)
    h = torch.ones(B, requires_grad=True)
    actor_critic.actor_critic_loss(lp, h, v, G, 0.5, 0.0).total.backward()
    after = lp.detach() - 0.1 * lp.grad
    up = G > v.detach()
    assert up.any() and (~up).any()
    assert (after[up] > -2.0).all(), "a better-than-expected move must get more likely"
    assert (after[~up] < -2.0).all(), "a worse-than-expected move must get less likely"


@steps.step(3)
def test_actor_critic_loss_leaves_its_inputs_alone():
    lp, h, v, G = loss_inputs(seed=6)
    copies = [x.detach().clone() for x in (lp, h, v, G)]
    actor_critic.actor_critic_loss(lp, h, v, G, 0.5, 0.01).total.backward()
    for x, y in zip((lp, h, v, G), copies):
        assert torch.equal(x.detach(), y)


# ---- final_move_rewards --------------------------------------------------------------------------

def test_final_move_rewards_on_interleaved_games():
    """Game g's reward after its last move, 0 after every other; game 3 has no moves."""
    ep = make_episodes(won=[True, False, False, False], vp=[10, 4, 5, 6],
                       game=[2, 0, 2, 1, 0, 2, 0])
    rewards = np.array([0.5, 0.25, 1.0, 0.75], np.float32)
    out = final_move_rewards(ep, rewards)
    assert type(out) is np.ndarray and out.dtype == np.float32 and out.shape == (7,)
    assert out.tolist() == [0, 0, 0, 0.25, 0, 1.0, 0.5]


def test_final_move_rewards_puts_each_reward_on_its_games_last_move():
    rng = np.random.default_rng(7)
    game = rng.choice([0, 2, 3, 5], size=200, p=[0.6, 0.3, 0.05, 0.05])  # 1, 4, 6: no moves
    ep = make_episodes(won=np.zeros(7, bool), vp=np.full(7, 4), game=game)
    rewards = rng.random(7).astype(np.float32) + 0.1
    out = final_move_rewards(ep, rewards)
    for g in range(7):
        rows = np.flatnonzero(game == g)
        if len(rows):
            assert out[rows[-1]] == rewards[g], g
            assert np.all(out[rows[:-1]] == 0), g
    assert np.count_nonzero(out) == len(np.unique(game))


def test_final_move_rewards_of_no_moves_and_float64_rewards():
    empty = final_move_rewards(make_episodes([True, False], [10, 3]), np.array([1.0, 0.3]))
    assert empty.dtype == np.float32 and empty.shape == (0,)
    ep = make_episodes(won=[False, True], vp=[3, 10], game=[1, 0, 1])
    rewards = np.array([0.3, 1.0])
    out = final_move_rewards(ep, rewards)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [0.0, 0.3, 1.0], rtol=1e-6)
    assert rewards.tolist() == [0.3, 1.0] and ep.game.tolist() == [1, 0, 1]


# ---- explained_variance --------------------------------------------------------------------------

RETURNS = np.array([0.2, 0.9, 0.4, 0.0, 0.7, 0.3, 1.0], np.float32)


def test_explained_variance_perfect_is_one():
    ev = explained_variance(RETURNS, RETURNS.copy())
    assert isinstance(ev, float) and ev == pytest.approx(1.0, abs=1e-12)


def test_explained_variance_of_predicting_the_mean_is_zero():
    assert explained_variance(RETURNS, np.full(7, RETURNS.mean())) == pytest.approx(0, abs=1e-6)
    # Any constant does as well as the mean: the variance ignores an offset.
    assert explained_variance(RETURNS, np.full(7, 5.0, np.float32)) == pytest.approx(0, abs=1e-5)


def test_explained_variance_worse_than_the_mean_is_negative():
    reflected = 2 * RETURNS.mean() - RETURNS  # errors twice the deviations: 1 - 4 = -3
    assert explained_variance(RETURNS, reflected) == pytest.approx(-3, rel=1e-5)
    noise = np.random.default_rng(0).normal(0, 3, 7).astype(np.float32)
    assert explained_variance(RETURNS, noise) < 0


def test_explained_variance_is_the_formula():
    rng = np.random.default_rng(1)
    returns = rng.random(50).astype(np.float32)
    values = (returns + rng.normal(0, 0.2, 50)).astype(np.float32)
    assert explained_variance(returns, values) == pytest.approx(
        by_hand_explained_variance(returns, values), rel=1e-5)


@pytest.mark.parametrize("returns", [np.zeros(5, np.float32), np.ones(4, np.float32),
                                     np.full(8, 0.5, np.float32), np.zeros(1, np.float32),
                                     np.full(7, 0.1, np.float32), np.full(334, 0.2, np.float32),
                                     np.full(1000, 0.2, np.float32), np.full(9, 0.7)])
def test_explained_variance_of_constant_returns_is_nan(returns):
    """Nothing to explain, including constants whose float32 mean is inexact (np.var of
    np.full(7, 0.1, float32) is about 1e-15, not 0)."""
    values = np.linspace(-1, 1, len(returns)).astype(np.float32)
    assert math.isnan(explained_variance(returns, values))
    assert math.isnan(explained_variance(returns, returns.copy()))


# ---- Config --------------------------------------------------------------------------------------

def test_config_defaults_and_fields():
    c = Config()
    c.check()
    assert c.gamma == 1.0 and c.value_coef == 0.5
    base_fields = [f.name for f in dataclasses.fields(base.Config)]
    assert [f.name for f in dataclasses.fields(Config)][:len(base_fields)] == base_fields


@pytest.mark.parametrize("gamma", [0, 0.0, 0.5, 0.999, 1, 1.0])
def test_config_accepts_gamma_in_zero_to_one(gamma):
    Config(gamma=gamma).check()


@pytest.mark.parametrize("gamma", [-1e-9, -1, 1 + 1e-9, 2, math.nan, math.inf, -math.inf, True,
                                   False, "0.9", None])
def test_config_rejects_gamma(gamma):
    with pytest.raises(ValueError) as exc:
        Config(gamma=gamma).check()
    assert "gamma" in str(exc.value)


@pytest.mark.parametrize("value_coef", [0, 0.0, 0.5, 3, 1e6])
def test_config_accepts_value_coef_at_least_zero(value_coef):
    Config(value_coef=value_coef).check()


@pytest.mark.parametrize("value_coef", [-1e-9, -1, math.nan, math.inf, -math.inf, True, "0.5",
                                        None])
def test_config_rejects_value_coef(value_coef):
    with pytest.raises(ValueError) as exc:
        Config(value_coef=value_coef).check()
    assert "value_coef" in str(exc.value)


@pytest.mark.parametrize("changes,name", [
    ({"lr": 0}, "lr"), ({"updates": 0}, "updates"), ({"hidden": True}, "hidden"),
    ({"entropy_coef": -0.1}, "entropy_coef"), ({"max_grad_norm": math.inf}, "max_grad_norm"),
    ({"reward": "loss"}, "reward"), ({"opponent": "self"}, "opponent"),
    ({"minibatch": 2.0}, "minibatch")])
def test_config_still_checks_the_base_fields(changes, name):
    with pytest.raises(ValueError) as exc:
        Config(**changes).check()
    assert name in str(exc.value)


def test_config_from_dict_round_trips():
    for c in (Config(), Config(gamma=0.9, value_coef=1.5, updates=7, lr=3e-4, reward="vp",
                               entropy_coef=0.01, opponent="greedy", seed=4, hidden=64),
              Config(gamma=0, value_coef=0)):
        assert Trainer.config_from_dict(dataclasses.asdict(c)) == c
        # As stored in JSON, too.
        assert Trainer.config_from_dict(json.loads(json.dumps(dataclasses.asdict(c)))) == c


def test_config_from_dict_rejects_a_reinforce_config():
    with pytest.raises(ValueError) as exc:
        Trainer.config_from_dict(dataclasses.asdict(reinforce.Config()))
    assert "baseline" in str(exc.value)


# ---- Trainer: what needs no network --------------------------------------------------------------

@pytest.mark.parametrize("config", [reinforce.Config(), base.Config()])
def test_trainer_needs_an_actor_critic_config(tmp_path, config):
    with pytest.raises(TypeError):
        Trainer(config, tmp_path)


@pytest.mark.parametrize("changes", [{"gamma": 1.5}, {"gamma": -0.1}, {"gamma": math.nan},
                                     {"value_coef": -1}, {"lr": 0}, {"reward": "loss"},
                                     {"opponent": "self"}])
def test_trainer_rejects_an_invalid_config(tmp_path, changes):
    with pytest.raises(ValueError):
        Trainer(Config(**changes), tmp_path)


def test_a_reinforce_run_directory_is_rejected(tmp_path):
    """The Trainer and saved_config both refuse another algorithm's checkpoint, naming both
    algorithms; the check comes before the net is built, and the checkpoint is left alone."""
    reinforce.Trainer(reinforce.Config(hidden=16, layers=1), tmp_path).save()
    before = (tmp_path / "checkpoint.pt").read_bytes()
    with pytest.raises(ValueError) as exc:
        Trainer(Config(hidden=16, layers=1), tmp_path)
    assert "reinforce" in str(exc.value) and "actor-critic" in str(exc.value)
    with pytest.raises(ValueError) as exc:
        Trainer.saved_config(tmp_path)
    assert "reinforce" in str(exc.value) and "actor-critic" in str(exc.value)
    assert (tmp_path / "checkpoint.pt").read_bytes() == before


def test_saved_config_of_an_empty_directory_is_none(tmp_path):
    assert Trainer.saved_config(tmp_path) is None


# ---- accumulate_gradients (under `impl`) ---------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic(positions):
    """61 moves at fixture positions over 9 games, with returns of both signs."""
    idx = np.random.default_rng(0).choice(len(positions), 61, replace=False)
    ep = position_episodes(positions, idx, games=9, seed=1)
    returns = np.random.default_rng(2).normal(0.2, 0.6, ep.moves).astype(np.float32)
    return ep, returns


@pytest.mark.parametrize("minibatch", [1, 7, 61, 100_000])
@pytest.mark.parametrize("value_coef,entropy_coef", [(0.5, 0.0), (2.0, 0.05)])
def test_accumulate_gradients_is_the_full_batch_gradient(impl, synthetic, minibatch,
                                                         value_coef, entropy_coef):
    """Whatever the chunking, the gradient is actor_critic_loss's over all the moves at once,
    the statistics are that batch's means, nothing is stepped and the inputs are left alone."""
    ep, returns = synthetic
    assert ep.moves == 61
    net = make_ac(2, hidden=32)
    want = reference(net, ep, returns, value_coef, entropy_coef)
    before = copy.deepcopy(net.state_dict())
    ep_copy, returns_copy = copy.deepcopy(ep), returns.copy()
    out = accumulate_gradients(net, ep, returns, minibatch=minibatch, value_coef=value_coef,
                               entropy_coef=entropy_coef)
    assert_grads_close(grads_of(net), want.grads)
    assert set(STAT_KEYS) <= set(out)
    assert all(isinstance(out[k], float) for k in STAT_KEYS)
    for k in ("loss", "policy_loss", "value_loss", "entropy", "log_prob"):
        assert out[k] == pytest.approx(getattr(want, k), rel=1e-4, abs=1e-6), k
    assert out["value"] == pytest.approx(want.values.mean(), rel=1e-4, abs=1e-6)
    assert out["explained_variance"] == pytest.approx(
        by_hand_explained_variance(returns, want.values), rel=1e-4, abs=1e-5)
    for k, v in net.state_dict().items():
        assert torch.equal(v, before[k]), f"{k} changed: the optimizer must not be stepped"
    assert np.array_equal(returns, returns_copy)
    for f in dataclasses.fields(Episodes):
        x, y = getattr(ep, f.name), getattr(ep_copy, f.name)
        assert (np.array_equal(x, y) if isinstance(x, np.ndarray) else x == y), f.name


def test_accumulate_gradients_adds_to_existing_grads(impl, synthetic):
    ep, returns = synthetic
    net = make_ac(3, hidden=32)
    want = reference(net, ep, returns, 0.7, 0.01).grads
    g = gen(4)
    existing = {n: torch.randn(p.shape, generator=g) * 1e-2 for n, p in net.named_parameters()}
    for n, p in net.named_parameters():
        p.grad = existing[n].clone()
    accumulate_gradients(net, ep, returns, minibatch=16, value_coef=0.7, entropy_coef=0.01)
    assert_grads_close(grads_of(net), {n: existing[n] + want[n] for n in want})
    # And a second call adds the same gradient again.
    accumulate_gradients(net, ep, returns, minibatch=16, value_coef=0.7, entropy_coef=0.01)
    assert_grads_close(grads_of(net), {n: existing[n] + 2 * want[n] for n in want})


def test_accumulate_gradients_rejects_no_moves(impl):
    net = make_ac(6, hidden=16, layers=1)
    ep = make_episodes(won=[True, False], vp=[10, 3])
    with pytest.raises(ValueError):
        accumulate_gradients(net, ep, np.zeros(0, np.float32))


@pytest.mark.parametrize("extra", [-1, 1, -61])
def test_accumulate_gradients_rejects_other_than_one_return_per_move(impl, synthetic, extra):
    ep, returns = synthetic
    net = make_ac(7, hidden=16, layers=1)
    with pytest.raises(ValueError):
        accumulate_gradients(net, ep, np.resize(returns, ep.moves + extra))


# ---- Trainer (under `impl`) ----------------------------------------------------------------------

# Small and fast, as test_reinforce.py's TINY: games against greedy are short, minibatch is
# smaller than the moves, and reward "vp" gives a gradient. gamma < 1 makes the returns differ
# within each game, so the explained variance is never NaN.
TINY = Config(updates=4, games_per_update=6, eval_games=4, eval_every=2, hidden=32,
              opponent="greedy", reward="vp", seed=6, minibatch=150, gamma=0.97,
              value_coef=0.5, entropy_coef=0.01)


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
    """step() plays games_per_update recorded games against config.opponent on fresh seeds,
    takes one clipped Adam step on accumulate_gradients' gradient of those games (returns by
    discounted_returns' definition), and reports trainer.Trainer's metrics plus actor-critic's,
    all finite and JSON-serializable. max_grad_norm is tiny so the clipping matters."""
    cfg = dataclasses.replace(TINY, games_per_update=5, lr=3e-3, max_grad_norm=1e-6,
                              minibatch=100, gamma=0.9, value_coef=0.7, entropy_coef=0.02)
    trainer = Trainer(cfg, tmp_path)
    before = copy.deepcopy(trainer.net)
    metrics = trainer.step()
    assert trainer.update == 1 and metrics["update"] == 1
    (args, ep), = play_calls
    assert args["opponent"] == cfg.opponent and args["games"] == cfg.games_per_update
    assert args["record"] is True and args["first_seed"] >= cfg.train_first_seed
    assert ep.games == cfg.games_per_update and ep.moves > 0

    rewards = game_rewards(ep, cfg.reward)
    returns = definition_returns(final_move_rewards(ep, rewards), ep.game, ep.games, cfg.gamma)
    want = reference(before, ep, returns, cfg.value_coef, cfg.entropy_coef)
    ref = copy.deepcopy(before)
    norm = adam_step(ref, want.grads, cfg)
    assert norm > 100 * cfg.max_grad_norm  # so the clipping mattered
    assert metrics["grad_norm"] == pytest.approx(norm, rel=1e-4)
    assert_same_step(trainer.net, ref, before, cfg.lr)

    assert STEP_KEYS | AC_KEYS <= set(metrics), f"missing {(STEP_KEYS | AC_KEYS) - set(metrics)}"
    assert metrics["games"] == cfg.games_per_update and metrics["moves"] == ep.moves
    assert metrics["reward"] == pytest.approx(rewards.mean(), rel=1e-6)
    assert metrics["return"] == pytest.approx(returns.mean(), rel=1e-5)
    for k in ("loss", "policy_loss", "value_loss", "entropy", "log_prob"):
        assert metrics[k] == pytest.approx(getattr(want, k), rel=1e-4, abs=1e-6), k
    assert metrics["value"] == pytest.approx(want.values.mean(), rel=1e-4, abs=1e-6)
    assert metrics["explained_variance"] == pytest.approx(
        by_hand_explained_variance(returns, want.values), rel=1e-4, abs=1e-5)
    assert json.loads(json.dumps(metrics, allow_nan=False)) == metrics

    trainer.step()
    (second, _) = play_calls[1]
    assert second["first_seed"] >= args["first_seed"] + cfg.games_per_update
    assert second["games"] == cfg.games_per_update and second["record"] is True


def test_run_writes_the_metrics_lines_and_a_checkpoint(impl, straight):
    lines = straight.metrics
    assert [line["update"] for line in lines] == list(range(TINY.updates + 1))
    assert EVAL_KEYS <= set(lines[0]) and not (AC_KEYS & set(lines[0]))
    evals = eval_updates(TINY)
    for line in lines[1:]:
        u = line["update"]
        assert STEP_KEYS | AC_KEYS <= set(line), f"update {u} misses some keys"
        assert (EVAL_KEYS <= set(line)) == (u in evals), f"update {u}"
        assert line["games"] == TINY.games_per_update
        for k, v in line.items():
            assert isinstance(v, (int, float)) and math.isfinite(v), (u, k, v)
    checkpoint = torch.load(straight.dir / "checkpoint.pt", weights_only=True)
    assert checkpoint["algorithm"] == "actor-critic"
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


@pytest.mark.parametrize("changes", [{"gamma": 0.9}, {"value_coef": 1.0},
                                     {"gamma": 1.0, "value_coef": 0.25}])
def test_resume_rejects_a_different_gamma_or_value_coef(impl, tmp_path, changes):
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


def test_an_actor_critic_run_is_not_a_reinforce_run(impl, straight):
    before = (straight.dir / "checkpoint.pt").read_bytes()
    for call in (lambda: reinforce.Trainer(reinforce.Config(), straight.dir),
                 lambda: reinforce.Trainer.saved_config(straight.dir)):
        with pytest.raises(ValueError) as exc:
            call()
        assert "reinforce" in str(exc.value) and "actor-critic" in str(exc.value)
    assert (straight.dir / "checkpoint.pt").read_bytes() == before


# ---- your three steps together -------------------------------------------------------------------

@pytest.fixture(scope="module")
def recorded():
    """Real episodes: an untrained PolicyNet (hidden 32) recorded against greedy."""
    net = make_net(21, hidden=32)
    return play(Agent(net, generator=gen(22)), "greedy", games=6, first_seed=5001,
                generator=gen(23), record=True)


@steps.all_steps
def test_step_is_one_clipped_adam_step_on_the_documented_loss(tmp_path, play_calls):
    """Reproduces step() from the episodes it played, independently of your code: rewards,
    returns by discounted_returns' definition, the documented loss's gradient (see
    documented_loss), its norm, clipping, one Adam step; and the metrics."""
    cfg = Config(updates=1, games_per_update=5, eval_games=4, hidden=32, opponent="greedy",
                 seed=11, lr=3e-3, max_grad_norm=1e-6, minibatch=100, reward="vp",
                 entropy_coef=0.01, gamma=0.95, value_coef=0.7)
    trainer = Trainer(cfg, tmp_path)
    before = copy.deepcopy(trainer.net)
    metrics = trainer.step()
    (args, ep), = play_calls
    assert args["games"] == 5 and args["record"] is True and args["opponent"] == "greedy"
    # The moves were sampled from the net as it was before the step.
    np.testing.assert_allclose(ep.log_prob, log_probs(before, ep).numpy(), rtol=1e-5, atol=1e-5)

    rewards = np.minimum(ep.vp, 10) / 10
    returns = definition_returns(final_move_rewards(ep, rewards.astype(np.float32)), ep.game,
                                 ep.games, cfg.gamma)
    want = documented_loss(before, ep, returns, cfg.value_coef, cfg.entropy_coef)
    ref = copy.deepcopy(before)
    norm = adam_step(ref, want.grads, cfg)
    assert metrics["grad_norm"] == pytest.approx(norm, rel=1e-4)
    assert norm > 100 * cfg.max_grad_norm  # so the clipping mattered
    assert_same_step(trainer.net, ref, before, cfg.lr)

    assert metrics["reward"] == pytest.approx(rewards.mean(), rel=1e-5)
    assert metrics["return"] == pytest.approx(returns.mean(), rel=1e-5)
    for k in ("loss", "policy_loss", "value_loss", "entropy", "log_prob"):
        assert metrics[k] == pytest.approx(getattr(want, k), rel=1e-3, abs=1e-6), k
    assert metrics["value"] == pytest.approx(want.values.mean(), rel=1e-4, abs=1e-6)
    assert metrics["explained_variance"] == pytest.approx(
        by_hand_explained_variance(returns, want.values), rel=1e-4, abs=1e-5)


@steps.all_steps
def test_the_critic_learns_the_returns_of_a_fixed_batch(recorded):
    """Adam steps on one recorded batch: the value loss falls to a small fraction of where it
    started, and the critic explains most of the returns' variance."""
    ep = recorded
    returns = definition_returns(final_move_rewards(ep, game_rewards(ep, "vp")), ep.game,
                                 ep.games, 0.95)
    assert np.var(returns) > 0
    net = make_ac(31, hidden=32)
    with torch.no_grad():
        start = outputs(net, ep)[0].double().numpy()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(80):
        opt.zero_grad()
        accumulate_gradients(net, ep, returns, minibatch=128, value_coef=0.5)
        opt.step()
    with torch.no_grad():
        end = outputs(net, ep)[0].double().numpy()
    loss_start = np.mean((start - returns) ** 2)
    loss_end = np.mean((end - returns) ** 2)
    assert loss_end < 0.1 * loss_start, (loss_start, loss_end)
    assert explained_variance(returns, end) > 0.5


def type_prob(net, obs: np.ndarray, mask: np.ndarray, t: int) -> np.ndarray:
    """P(type t) at each row (t takes no argument, so this is the move's probability)."""
    o = torch.as_tensor(obs)
    n = len(obs)
    with torch.no_grad():
        lp, _ = actions.log_prob_entropy(net(o), torch.as_tensor(mask), actions.phase_of(o),
                                         torch.full((n,), t), torch.zeros(n, dtype=torch.int64))
    return lp.exp().numpy()


@steps.all_steps
def test_the_actor_favours_moves_with_positive_advantage(recorded, positions):
    """Returns of 1 after the batch's END_TURN moves and 0 after the rest: while the critic
    predicts less than 1, END_TURN moves do better than it expects (positive advantage) and
    the rest worse, so a few Adam steps raise P(END_TURN) at held-out positions where it
    competes with other types."""
    ep = recorded
    target = int(AT.END_TURN)
    returns = (ep.types == target).astype(np.float32)
    assert 0 < returns.sum() < len(returns)
    held = positions.where(lambda i: positions.mask[i, target]
                           and positions.mask[i, :ce.NUM_ACTION_TYPES].sum() >= 2)
    assert len(held) >= 10
    net = make_ac(31, hidden=32)
    p0 = type_prob(net, positions.obs[held], positions.mask[held], target)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(15):
        opt.zero_grad()
        accumulate_gradients(net, ep, returns, minibatch=128, value_coef=0.5)
        opt.step()
    p1 = type_prob(net, positions.obs[held], positions.mask[held], target)
    assert p1.mean() > p0.mean() + 0.2, (p0.mean(), p1.mean())
    assert (p1 > p0).mean() > 0.9


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
