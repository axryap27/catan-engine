"""Tests for catan_rl.policy (PolicyNet, Agent), written from the contract docstrings."""

import numpy as np
import pytest
import torch
from torch import nn

from catan_rl import actions
from catan_rl.engine import ce
from catan_rl.policy import Agent, PolicyNet


def make_net(seed: int = 0, **kwargs) -> PolicyNet:
    """A PolicyNet built under a fixed torch seed, without touching the global RNG state."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return PolicyNet(**kwargs)


@pytest.fixture(scope="module")
def net() -> PolicyNet:
    return make_net(0)


def gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def linears(net: nn.Module) -> tuple[nn.Linear, list[nn.Linear], nn.Linear]:
    """(input layer, hidden->hidden layers in registration order, output layer)."""
    lin = [m for m in net.modules() if isinstance(m, nn.Linear)]
    first = [m for m in lin if m.in_features == ce.OBS_SIZE]
    last = [m for m in lin if m.out_features == ce.MASK_SIZE]
    assert len(first) == 1 and len(last) == 1, "expected one input and one output Linear"
    middle = [m for m in lin if m is not first[0] and m is not last[0]]
    return first[0], middle, last[0]


def net_logits(net: PolicyNet, obs: np.ndarray) -> torch.Tensor:
    with torch.no_grad():
        return net(torch.from_numpy(obs))


# ---- PolicyNet ----------------------------------------------------------------------------

def test_output_shape_and_dtype(net, positions):
    out = net_logits(net, positions.obs)
    assert out.shape == (len(positions), ce.MASK_SIZE) and out.dtype == torch.float32
    assert torch.isfinite(out).all()
    assert net_logits(net, positions.obs[:1]).shape == (1, ce.MASK_SIZE)


def test_uint8_and_float_obs_give_same_logits(net, positions):
    u8 = net_logits(net, positions.obs)
    with torch.no_grad():
        f32 = net(torch.from_numpy(positions.obs).float())
    torch.testing.assert_close(u8, f32, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("hidden,layers", [(None, None), (32, 1), (48, 3)])
def test_architecture(hidden, layers):
    """`layers` x [Linear, LayerNorm, ReLU] of width `hidden`, then Linear to MASK_SIZE
    (defaults hidden=256, layers=2)."""
    kwargs = {} if hidden is None else {"hidden": hidden, "layers": layers}
    hidden = hidden or 256
    layers = layers or 2
    net = make_net(1, **kwargs)
    first, middle, last = linears(net)
    assert len(middle) == layers - 1
    assert first.out_features == hidden
    assert all(m.in_features == m.out_features == hidden for m in middle)
    assert last.in_features == hidden
    norms = [m for m in net.modules() if isinstance(m, nn.LayerNorm)]
    assert len(norms) == layers
    assert all(tuple(m.normalized_shape) == (hidden,) for m in norms)
    assert sum(isinstance(m, nn.Linear) for m in net.modules()) == layers + 1


@pytest.mark.parametrize("hidden,layers", [(32, 1), (48, 3)])
def test_forward_is_log1p_then_mlp(positions, hidden, layers):
    """Rebuild the documented forward pass from the net's own layers: log1p(obs), then
    [Linear, LayerNorm, ReLU] x layers, then the output Linear."""
    net = make_net(2, hidden=hidden, layers=layers)
    first, middle, last = linears(net)
    norms = [m for m in net.modules() if isinstance(m, nn.LayerNorm)]
    obs = torch.from_numpy(positions.obs)
    with torch.no_grad():
        x = torch.log1p(obs.float())
        for lin, norm in zip([first, *middle], norms):
            x = torch.relu(norm(lin(x)))
        want = last(x)
        got = net(obs)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("hidden", [32, 256])
def test_output_layer_init(hidden):
    """Orthogonal init with gain 0.01 (all singular values 0.01) and zero bias."""
    _, _, last = linears(make_net(3, hidden=hidden))
    assert torch.all(last.bias == 0)
    sv = torch.linalg.svdvals(last.weight.detach().double())
    assert len(sv) == min(hidden, ce.MASK_SIZE)
    torch.testing.assert_close(sv, torch.full_like(sv, 0.01), rtol=1e-4, atol=1e-7)


def test_untrained_net_plays_like_zero_logits(net, positions):
    """An untrained net's logits are about 0, so each legal move's probability is close to
    the zero-logit (random-bot) one: |logit| < 0.1 bounds the log-prob change by 0.4."""
    logits = net_logits(net, positions.obs)
    assert logits.abs().max().item() < 0.1
    mask = torch.from_numpy(positions.mask)
    phases = torch.from_numpy(positions.phases)
    rows = [i for i, moves in enumerate(positions.legal) for _ in moves]
    types = torch.tensor([t for moves in positions.legal for t, _ in moves])
    args = torch.tensor([a for moves in positions.legal for _, a in moves])
    rows = torch.tensor(rows)
    lp_net, h_net = actions.log_prob_entropy(logits[rows], mask[rows], phases[rows], types, args)
    lp_zero, h_zero = actions.log_prob_entropy(
        torch.zeros(len(rows), ce.MASK_SIZE), mask[rows], phases[rows], types, args)
    assert (lp_net - lp_zero).abs().max().item() < 0.4
    assert (h_net - h_zero).abs().max().item() < 0.2


# ---- Agent --------------------------------------------------------------------------------

def check_types(types, args, log_prob, n):
    for x in (types, args, log_prob):
        assert type(x) is np.ndarray, f"expected a plain numpy array, got {type(x)}"
        assert x.shape == (n,)
    assert types.dtype == np.int64 and args.dtype == np.int64 and log_prob.dtype == np.float32


def assert_legal(positions, idx, types, args):
    for i, t, a in zip(idx, types.tolist(), args.tolist()):
        assert positions.games[i].is_legal(t, a), (
            f"{ce.Phase(int(positions.phases[i])).name}: illegal ({ce.ActionType(t).name}, {a})")


@pytest.mark.parametrize("deterministic", [False, True])
def test_act_dtypes_shapes_and_legality(net, positions, deterministic):
    agent = Agent(net, generator=gen(1), deterministic=deterministic)
    types, args, log_prob = agent.act(positions.obs, positions.mask)
    check_types(types, args, log_prob, len(positions))
    assert np.isfinite(log_prob).all() and (log_prob <= 1e-6).all()
    assert_legal(positions, range(len(positions)), types, args)


@pytest.mark.parametrize("deterministic", [False, True])
def test_act_empty_batch(net, deterministic):
    agent = Agent(net, generator=gen(1), deterministic=deterministic)
    out = agent.act(np.zeros((0, ce.OBS_SIZE), np.uint8), np.zeros((0, ce.MASK_SIZE), np.uint8))
    check_types(*out, 0)


def test_act_log_prob_matches_net_distribution(net, positions):
    """act's log_prob is log P(move) under the net's logits and the observation's phase."""
    for deterministic in (False, True):
        agent = Agent(net, generator=gen(2), deterministic=deterministic)
        types, args, log_prob = agent.act(positions.obs, positions.mask)
        logits = net_logits(net, positions.obs)
        obs = torch.from_numpy(positions.obs)
        want, _ = actions.log_prob_entropy(
            logits, torch.from_numpy(positions.mask), actions.phase_of(obs),
            torch.from_numpy(types), torch.from_numpy(args))
        np.testing.assert_allclose(log_prob, want.numpy(), rtol=1e-5, atol=1e-5)


def test_deterministic_act_is_mode_and_repeatable(net, positions):
    agent = Agent(net, deterministic=True)
    types, args, _ = agent.act(positions.obs, positions.mask)
    obs = torch.from_numpy(positions.obs)
    want_t, want_a = actions.mode(net_logits(net, positions.obs),
                                  torch.from_numpy(positions.mask), actions.phase_of(obs))
    assert np.array_equal(types, want_t.numpy()) and np.array_equal(args, want_a.numpy())
    again = agent.act(positions.obs, positions.mask)
    assert np.array_equal(types, again[0]) and np.array_equal(args, again[1])


def test_sampling_act_samples(net, positions):
    """deterministic=False (the default) samples: on near-uniform positions with many moves
    it must often differ from the mode."""
    types, args, _ = Agent(net, generator=gen(3)).act(positions.obs, positions.mask)
    mode_t, mode_a, _ = Agent(net, deterministic=True).act(positions.obs, positions.mask)
    many = np.array([len(m) >= 3 for m in positions.legal])
    differs = (types != mode_t) | (args != mode_a)
    assert differs[many].mean() > 0.3


def test_act_generator_reproducible(net, positions):
    first = Agent(net, generator=gen(9)).act(positions.obs, positions.mask)
    second = Agent(net, generator=gen(9)).act(positions.obs, positions.mask)
    for x, y in zip(first, second):
        assert np.array_equal(x, y)
    other = Agent(net, generator=gen(10)).act(positions.obs, positions.mask)
    assert not (np.array_equal(first[0], other[0]) and np.array_equal(first[1], other[1]))


def test_act_without_generator_uses_global_rng(net, positions):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4)
        first = Agent(net).act(positions.obs, positions.mask)
        torch.manual_seed(4)
        second = Agent(net).act(positions.obs, positions.mask)
    for x, y in zip(first, second):
        assert np.array_equal(x, y)


def test_act_tracks_no_gradients(positions):
    """act runs under no_grad: no parameter grads, grad mode restored, plain numpy out."""
    net = make_net(5, hidden=32, layers=1)
    agent = Agent(net, generator=gen(0))
    assert torch.is_grad_enabled()
    out = agent.act(positions.obs, positions.mask)
    assert torch.is_grad_enabled()
    assert all(p.grad is None for p in net.parameters())
    check_types(*out, len(positions))


def test_act_accepts_row_subsets_and_leaves_inputs_alone(net, positions):
    """Rows of a VecGames.observe() batch may be passed as views (e.g. every other row)."""
    obs, mask = positions.obs.copy(), positions.mask.copy()
    view_obs, view_mask = obs[::2], mask[::2]
    agent = Agent(net, deterministic=True)
    from_view = agent.act(view_obs, view_mask)
    from_copy = agent.act(np.ascontiguousarray(view_obs), np.ascontiguousarray(view_mask))
    for x, y in zip(from_view, from_copy):
        assert np.array_equal(x, y)
    assert np.array_equal(obs, positions.obs) and np.array_equal(mask, positions.mask)
    assert_legal(positions, range(0, len(positions), 2), from_view[0], from_view[1])
