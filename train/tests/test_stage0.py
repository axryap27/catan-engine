"""Tests for train/stage0.py. Run one exercise with -k exercise1 (2, 3, 4)."""

import pytest
import torch

import stage0 as s

# ---- Exercise 1 ------------------------------------------------------------------------------


def test_exercise1_fits_a_noisy_line():
    torch.manual_seed(0)
    xs = torch.linspace(-1, 1, 50)
    ys = 3 * xs + 2 + 0.05 * torch.randn(50)
    w, b = s.fit_line(xs, ys)
    assert isinstance(w, float) and isinstance(b, float), "return plain floats (.item())"
    assert w == pytest.approx(3, abs=0.1)
    assert b == pytest.approx(2, abs=0.1)


def test_exercise1_fits_another_line():
    xs = torch.linspace(0, 2, 20)
    w, b = s.fit_line(xs, -1.5 * xs + 0.5, steps=2000)
    assert w == pytest.approx(-1.5, abs=0.01)
    assert b == pytest.approx(0.5, abs=0.01)


# ---- Exercise 2 ------------------------------------------------------------------------------

LOGITS = torch.tensor([[1.0, 2.0, 3.0, 0.5],
                       [0.0, -1.0, 4.0, 2.0],
                       [5.0, 5.0, 5.0, 5.0]])
MASK = torch.tensor([[True, False, True, True],
                     [False, True, False, True],
                     [False, False, True, False]])


def masked_probs():
    """The right answer, computed independently: softmax over the legal entries only."""
    e = torch.exp(LOGITS) * MASK
    return e / e.sum(dim=1, keepdim=True)


def test_exercise2_masked_logits_zero_out_illegal_moves():
    logits, mask = LOGITS.clone(), MASK.clone()
    out = s.masked_logits(logits, mask)
    assert out.shape == LOGITS.shape
    assert torch.equal(logits, LOGITS) and torch.equal(mask, MASK), "don't modify the inputs"
    assert torch.equal(out[MASK], LOGITS[MASK]), "legal logits must be unchanged"
    probs = torch.softmax(out, dim=1)
    assert torch.all(probs[~MASK] == 0), "an illegal move kept some probability"
    assert torch.allclose(probs, masked_probs())


def test_exercise2_samples_are_legal_and_follow_the_probabilities():
    torch.manual_seed(0)
    n = 20000
    logits = LOGITS.repeat(n, 1)
    mask = MASK.repeat(n, 1)
    actions, log_probs = s.sample_actions(logits, mask)
    assert actions.shape == (3 * n,) and log_probs.shape == (3 * n,)
    assert actions.dtype == torch.int64
    assert mask[torch.arange(3 * n), actions].all(), "sampled an illegal move"
    for row in range(3):
        picked = actions[row::3]
        freq = torch.bincount(picked, minlength=4).float() / n
        assert torch.allclose(freq, masked_probs()[row], atol=0.02), f"row {row}: {freq}"
    # Each log-probability belongs to the action actually sampled.
    expected = torch.log(masked_probs()).repeat(n, 1)[torch.arange(3 * n), actions]
    assert torch.allclose(log_probs, expected, atol=1e-5)


def test_exercise2_only_legal_move_is_certain():
    actions, log_probs = s.sample_actions(LOGITS[2:], MASK[2:])
    assert actions.tolist() == [2]
    assert log_probs.item() == pytest.approx(0.0, abs=1e-6)


def test_exercise2_gradient_flows_to_legal_logits_only():
    logits = LOGITS.clone().requires_grad_(True)
    _, log_probs = s.sample_actions(logits, MASK)
    assert log_probs.requires_grad, "log_probs lost the gradient; use torch ops, not .item()"
    log_probs.sum().backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.all(logits.grad[~MASK] == 0), "gradient leaked into an illegal move"


# ---- Exercise 3 ------------------------------------------------------------------------------


def test_exercise3_learns_the_best_arms():
    torch.manual_seed(0)
    logits = s.train_bandit()
    assert logits.shape == (s.N_BANDIT_ARMS,)
    ranked = logits.argsort(descending=True).tolist()
    assert ranked[:3] == [3, 2, 5], f"arms by learned preference: {ranked} (best is 3, 2, 5)"


def test_exercise3_earns_nearly_the_best_possible_reward():
    torch.manual_seed(1)
    logits = s.train_bandit()
    mask = s.bandit_masks(100000)
    probs = torch.softmax(s.masked_logits(logits.expand(100000, -1), mask), dim=1)
    earned = (probs * s.WIN_PROB).sum(dim=1).mean().item()
    best = s.WIN_PROB.expand(100000, -1).masked_fill(~mask, 0).max(dim=1).values.mean().item()
    assert earned >= 0.97 * best, f"expected reward {earned:.3f}, best possible {best:.3f}"


# ---- Exercise 4 ------------------------------------------------------------------------------


def test_exercise4_policy_shapes():
    assert s.Policy()(torch.randn(5, s.N_ARMS)).shape == (5, s.N_ARMS)
    small = s.Policy(n_obs=3, n_actions=4, hidden=16)
    assert small(torch.randn(2, 3)).shape == (2, 4)
    n_params = sum(p.numel() for p in small.parameters())
    assert n_params == 3 * 16 + 16 + 16 * 4 + 4, "expected Linear -> ReLU -> Linear"


def test_exercise4_learns_to_read_the_observation():
    torch.manual_seed(0)
    policy = s.train_contextual()
    torch.manual_seed(1)
    obs, mask = s.contextual_rounds(20000)
    with torch.no_grad():  # evaluating, not training: skip the gradient bookkeeping
        logits = policy(obs)
        top = s.masked_logits(logits, mask).argmax(dim=1)
        top_acc = s.contextual_reward(obs, mask, top).mean().item()
        arms, _ = s.sample_actions(logits, mask)
        sampled_acc = s.contextual_reward(obs, mask, arms).mean().item()
    # Picking uniformly among the legal arms scores about 0.3.
    assert top_acc >= 0.9, f"top-scored arm is right {top_acc:.2f} of the time"
    assert sampled_acc >= 0.85, f"sampled arm is right {sampled_acc:.2f} of the time"
