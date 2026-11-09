"""What the stage 3 and stage 4 tests share: networks and Episodes built by hand, stage 3's
stand-ins and definitions, reference gradients, and helpers for training runs.

The stand-ins replace stage 3's three steps (use_stage3_stand_ins) so the plumbing around them
is tested before they are written; see the `impl` fixtures of test_actor_critic.py and
test_gae.py.
"""

import copy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from catan_rl import actions, actor_critic
from catan_rl import trainer as base
from catan_rl.actor_critic import Losses
from catan_rl.engine import ce
from catan_rl.policy import PolicyNet
from catan_rl.rollout import Episodes, play


def gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def make_net(seed: int = 0, **kwargs) -> PolicyNet:
    """A PolicyNet built under a fixed torch seed, without touching the global RNG state."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return PolicyNet(**kwargs)


def make_ac(seed: int = 0, **kwargs) -> PolicyNet:
    """actor_critic.ActorCriticNet (yours, or the stand-in while `impl` patches it), built
    under a fixed torch seed without touching the global RNG state."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return actor_critic.ActorCriticNet(**kwargs)


# ---- building Episodes by hand (as in test_reinforce.py) -----------------------------------------

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


# ---- discounted_returns' definition --------------------------------------------------------------

def definition_returns(rewards, game, games: int, gamma: float) -> np.ndarray:
    """discounted_returns' docstring, evaluated literally and slowly: move m's return is the sum,
    over m's game's moves n_0 = m, n_1, n_2, ... from m on, of gamma^j * rewards[n_j]."""
    rewards = np.asarray(rewards, dtype=np.float64)
    game = np.asarray(game)
    out = np.zeros(len(rewards))
    for m in range(len(rewards)):
        later = m + np.flatnonzero(game[m:] == game[m])  # m's game's moves from m on, in order
        out[m] = sum(gamma ** j * rewards[n] for j, n in enumerate(later))
    return out.astype(np.float32)


# ---- stand-ins for stage 3's three steps (deliberately NOT the solution) ------------------------

class StandInNet(PolicyNet):
    """Not a critic: its "value" is the mean of the 13 type logits. It has a gradient and
    depends on the observation, which is all the plumbing needs."""

    def logits_and_value(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self(obs)
        return logits, logits[:, :actions.NUM_TYPES].mean(dim=1)


def stand_in_loss(log_prob, entropy, value, returns, value_coef, entropy_coef) -> Losses:
    """Not the actor-critic loss: some smooth per-move means of all four inputs, combined as
    Losses.total documents."""
    policy = (log_prob * returns.cos()).mean()
    value_part = (value.sin() * (1 + returns)).mean()
    mean_entropy = entropy.mean()
    return Losses(total=policy + value_coef * value_part - entropy_coef * mean_entropy,
                  policy=policy, value=value_part, entropy=mean_entropy)


def use_stage3_stand_ins(monkeypatch) -> None:
    """Swaps the stand-ins in for stage 3's three steps. The Trainers and accumulate_gradients
    look them up in actor_critic when called, so patching the module swaps them."""
    monkeypatch.setattr(actor_critic, "ActorCriticNet", StandInNet)
    monkeypatch.setattr(actor_critic, "discounted_returns", definition_returns)
    monkeypatch.setattr(actor_critic, "actor_critic_loss", stand_in_loss)


# ---- gradients and outputs -----------------------------------------------------------------------

def grads_of(net: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {n: (p.grad.clone() if p.grad is not None else torch.zeros_like(p))
            for n, p in net.named_parameters()}


def assert_grads_close(got: dict, want: dict):
    assert got.keys() == want.keys()
    for n in want:
        torch.testing.assert_close(got[n], want[n], rtol=2e-4, atol=2e-6, msg=n)


def outputs(net, ep: Episodes) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(value, log_prob, entropy), each (M,), of all of ep's moves at once, with gradient."""
    obs = torch.as_tensor(ep.obs)
    logits, value = net.logits_and_value(obs)
    log_prob, entropy = actions.log_prob_entropy(
        logits, torch.as_tensor(ep.mask), actions.phase_of(obs), torch.as_tensor(ep.types),
        torch.as_tensor(ep.args))
    return value, log_prob, entropy


def num(t: torch.Tensor) -> float:
    return float(t.detach())


def fresh_copy(net: torch.nn.Module) -> torch.nn.Module:
    net = copy.deepcopy(net)
    for p in net.parameters():
        p.grad = None
    return net


def reference(net, ep: Episodes, returns, value_coef: float = 0.5,
              entropy_coef: float = 0.0) -> SimpleNamespace:
    """The module's actor_critic_loss (the stand-in's or yours) on all the moves at once,
    backward on a copy of `net`: the gradient, the four parts, the mean log-prob, the values."""
    net = fresh_copy(net)
    value, log_prob, entropy = outputs(net, ep)
    losses = actor_critic.actor_critic_loss(
        log_prob, entropy, value, torch.as_tensor(returns, dtype=torch.float32), value_coef,
        entropy_coef)
    losses.total.backward()
    return SimpleNamespace(
        grads=grads_of(net), loss=num(losses.total), policy_loss=num(losses.policy),
        value_loss=num(losses.value), entropy=num(losses.entropy), log_prob=num(log_prob.mean()),
        values=value.detach().double().numpy())


def documented_loss(net, ep: Episodes, returns, value_coef: float,
                    entropy_coef: float) -> SimpleNamespace:
    """The module docstring's loss on all the moves at once, computed here without
    actor_critic_loss: its parts in numpy, and its gradient pushed through a copy of `net` by
    autograd from the gradient with respect to the net's three outputs, worked out by hand with
    the advantage held constant: -(G - value) / M for each log-prob, value_coef * 2 *
    (value - G) / M for each value and -entropy_coef / M for each entropy."""
    net = fresh_copy(net)
    value, log_prob, entropy = outputs(net, ep)
    M = ep.moves
    G = torch.as_tensor(returns, dtype=torch.float32)
    v = value.detach()
    torch.autograd.backward(
        [log_prob, value, entropy],
        [-(G - v) / M, value_coef * 2 * (v - G) / M, torch.full_like(entropy, -entropy_coef / M)])
    G, v = G.double().numpy(), v.double().numpy()
    lp, h = log_prob.detach().double().numpy(), entropy.detach().double().numpy()
    policy = -np.mean((G - v) * lp)
    value_loss = np.mean((v - G) ** 2)
    return SimpleNamespace(
        grads=grads_of(net), loss=policy + value_coef * value_loss - entropy_coef * h.mean(),
        policy_loss=policy, value_loss=value_loss, entropy=h.mean(), log_prob=lp.mean(),
        values=v)


def by_hand_explained_variance(returns, values) -> float:
    returns = np.asarray(returns, np.float64)
    return 1 - np.var(returns - np.asarray(values, np.float64)) / np.var(returns)


def log_probs(net, ep: Episodes) -> torch.Tensor:
    with torch.no_grad():
        return outputs(net, ep)[1]


# ---- training runs -------------------------------------------------------------------------------

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


def eval_updates(config: base.Config, start: int = 0) -> set[int]:
    """Updates whose metrics line carries an evaluation in a run from `start` (0 = fresh)."""
    done = {0} if start == 0 else set()
    done |= {u for u in range(start + 1, config.updates + 1) if u % config.eval_every == 0}
    done.add(config.updates)
    return done


def adam_step(net: torch.nn.Module, grads: dict, config: base.Config) -> float:
    """What step() does after learn(): clip the gradient's norm to max_grad_norm, then one Adam
    step with lr. Returns the norm from before clipping."""
    for n, p in net.named_parameters():
        p.grad = grads[n].clone()
    norm = float(torch.nn.utils.clip_grad_norm_(net.parameters(), config.max_grad_norm))
    torch.optim.Adam(net.parameters(), lr=config.lr).step()
    return norm


def assert_same_step(got: torch.nn.Module, want: torch.nn.Module, before: torch.nn.Module,
                     lr: float):
    b, w, g = before.state_dict(), want.state_dict(), got.state_dict()
    for k in b:
        torch.testing.assert_close(g[k] - b[k], w[k] - b[k], rtol=1e-2, atol=lr * 1e-3, msg=k)


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
