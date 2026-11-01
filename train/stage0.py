"""Stage 0: PyTorch and policy gradients on toy problems, before Catan.

A worked example: read it alongside train/LEARN.md. Run it to watch each exercise learn:

    .venv/bin/python train/stage0.py

To practice, delete a function's body, rewrite it, and run that exercise's tests:

    .venv/bin/python -m pytest train/tests/test_stage0.py -q -k exercise1

The exercises build on each other: 1 is plain PyTorch, 2 is the masking every Catan move needs,
3 is the policy-gradient learning rule, 4 puts a network in front of it, which is the shape of
the Catan policy (observation in, one score per move out).
"""

import torch
from torch import nn

# ---- Exercise 1: gradient descent ------------------------------------------------------------
# Fit y = w * x + b to the points (xs, ys) by minimizing the mean squared error
# mean((w * x + b - y) ** 2). The loop is the one from LEARN.md section 1:
#     loss = ...;  opt.zero_grad();  loss.backward();  opt.step()


def fit_line(xs: torch.Tensor, ys: torch.Tensor, steps: int = 500,
             lr: float = 0.1) -> tuple[float, float]:
    """xs, ys: shape (n,). Returns the fitted (w, b) as plain floats."""
    w = torch.zeros((), requires_grad=True)
    b = torch.zeros((), requires_grad=True)
    opt = torch.optim.SGD([w, b], lr=lr)
    for _ in range(steps):
        pred = w * xs + b                 # (n,): the line's guess for every point at once
        loss = ((pred - ys) ** 2).mean()  # one number, 0 only for a perfect fit
        opt.zero_grad()                   # clear the last step's gradients (they'd add up)
        loss.backward()                   # w.grad = d(loss)/dw, b.grad = d(loss)/db
        opt.step()                        # w -= lr * w.grad, b -= lr * b.grad
    return w.item(), b.item()             # .item(): a 0-d tensor -> a Python float


# ---- Exercise 2: masked sampling -------------------------------------------------------------
# logits: float (batch, n_actions), one score per action. mask: bool, same shape, True = legal.
# Every row has at least one legal action. Don't modify the inputs.


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """logits with every illegal entry set to -inf."""
    # masked_fill returns a new tensor; masked_fill_ (trailing underscore) would change logits
    # in place. PyTorch uses that naming for every in-place operation.
    return logits.masked_fill(~mask, float("-inf"))


def sample_actions(logits: torch.Tensor,
                   mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample one legal action per row from softmax(masked_logits).

    Returns (actions, log_probs), both shape (batch,): the action indices (int64) and each
    one's log-probability. log_probs must keep the gradient back to `logits` (Exercise 3 trains
    through it), so compute it with torch operations, not .item() or numpy."""
    # Categorical holds one probability distribution per row: softmax of the masked logits, so
    # illegal moves have probability 0.
    dist = torch.distributions.Categorical(logits=masked_logits(logits, mask))
    # A random draw has no gradient: you can't differentiate "which move came up".
    actions = dist.sample()
    # log softmax at the drawn move does have one. This is the quantity REINFORCE pushes up or
    # down, so it is the only path by which the reward reaches the logits.
    return actions, dist.log_prob(actions)


# ---- Exercise 3: REINFORCE on a slot machine -------------------------------------------------
# Arm i pays 1 with probability WIN_PROB[i], else 0. Each round some arms are illegal. The
# policy is one learnable logit per arm, no network; it should learn to prefer arm 3, then 2.

WIN_PROB = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.2, 0.4])
N_BANDIT_ARMS = len(WIN_PROB)


def bandit_masks(batch: int) -> torch.Tensor:
    """(batch, 6) bool: each arm legal with probability 1/2, at least one per row."""
    mask = torch.rand(batch, N_BANDIT_ARMS) < 0.5
    mask[torch.arange(batch), torch.randint(N_BANDIT_ARMS, (batch,))] = True
    return mask


def pull(arms: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Pull one arm per row. Returns float rewards (batch,). Raises on an illegal arm, as the
    engine does on an illegal move."""
    if not mask[torch.arange(len(arms)), arms].all():
        raise ValueError("pulled an illegal arm")
    return (torch.rand(arms.shape) < WIN_PROB[arms]).float()


def train_bandit(rounds: int = 300, batch: int = 256, lr: float = 0.1,
                 log_every: int = 0) -> torch.Tensor:
    """REINFORCE with the batch's mean reward as the baseline. Returns the learned logits (6,).
    log_every > 0 prints the mean reward every that many rounds."""
    logits = torch.zeros(N_BANDIT_ARMS, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=lr)
    for i in range(rounds):
        mask = bandit_masks(batch)
        # Play: one arm per row. expand repeats the single row of logits `batch` times without
        # copying; gradients from every row add up into the one shared row.
        arms, log_probs = sample_actions(logits.expand(batch, -1), mask)
        rewards = pull(arms, mask)
        # Score each play against the batch average: positive = better than usual.
        advantage = rewards - rewards.mean()
        # Minimizing this raises the log-probability of above-average plays and lowers the
        # rest. The minus sign is there because optimizers minimize and we want reward up.
        loss = -(log_probs * advantage).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if log_every and i % log_every == 0:
            print(f"  round {i:4d}: mean reward {rewards.mean().item():.3f}")
    # detach: the same numbers, cut loose from the gradient bookkeeping.
    return logits.detach()


# ---- Exercise 4: a policy network ------------------------------------------------------------
# Now the best arm changes every round. Each round shows 8 numbers x; arm i pays 1 if x[i] is
# the largest among the legal arms, else 0. The policy has to read x, so it's a network.

N_ARMS = 8


class Policy(nn.Module):
    """observation (batch, n_obs) -> logits (batch, n_actions)."""

    def __init__(self, n_obs: int = N_ARMS, n_actions: int = N_ARMS, hidden: int = 64):
        super().__init__()  # required first line of every nn.Module
        # Linear(a, b) is y = x @ W.T + bias with a learnable (b, a) matrix W. ReLU (max(0, y))
        # between the layers lets the network represent more than a straight-line function.
        # Assigning to self registers the layers, so policy.parameters() finds their weights.
        self.net = nn.Sequential(
            nn.Linear(n_obs, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def contextual_rounds(batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(obs (batch, 8) float, mask (batch, 8) bool) for one round per row."""
    obs = torch.randn(batch, N_ARMS)
    mask = torch.rand(batch, N_ARMS) < 0.5
    mask[torch.arange(batch), torch.randint(N_ARMS, (batch,))] = True
    return obs, mask


def contextual_reward(obs: torch.Tensor, mask: torch.Tensor, arms: torch.Tensor) -> torch.Tensor:
    """1.0 where the arm is the legal arm with the largest x, else 0.0. Raises on an illegal arm."""
    if not mask[torch.arange(len(arms)), arms].all():
        raise ValueError("pulled an illegal arm")
    best = obs.masked_fill(~mask, float("-inf")).argmax(dim=1)
    return (arms == best).float()


def train_contextual(rounds: int = 1000, batch: int = 256, lr: float = 3e-3,
                     log_every: int = 0) -> Policy:
    """REINFORCE as in Exercise 3, with the logits coming from policy(obs)."""
    policy = Policy()
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    for i in range(rounds):
        obs, mask = contextual_rounds(batch)
        # The only change from Exercise 3: the logits are computed from the observation, so the
        # gradient flows on through the network into every weight.
        arms, log_probs = sample_actions(policy(obs), mask)
        rewards = contextual_reward(obs, mask, arms)
        advantage = rewards - rewards.mean()
        loss = -(log_probs * advantage).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if log_every and i % log_every == 0:
            print(f"  round {i:4d}: right {rewards.mean().item():.0%} of the time")
    return policy


if __name__ == "__main__":
    torch.manual_seed(0)

    print("Exercise 1: fit y = 3x + 2")
    xs = torch.linspace(-1, 1, 50)
    w, b = fit_line(xs, 3 * xs + 2)
    print(f"  learned y = {w:.3f}x + {b:.3f}\n")

    print("Exercise 3: slot machine, win chances", [f"{p:.1f}" for p in WIN_PROB.tolist()])
    mask = bandit_masks(100000)
    best = WIN_PROB.expand(100000, -1).masked_fill(~mask, 0).max(dim=1).values.mean().item()
    print(f"  (best possible mean reward: {best:.2f}, since the 0.7 arm is legal only half the time)")
    logits = train_bandit(log_every=50)
    probs = torch.softmax(logits, dim=0)
    print("  chance of picking each arm when all are legal:",
          [f"{p:.2f}" for p in probs.tolist()], "\n")

    print("Exercise 4: pick the legal arm with the largest number (random guessing: ~30%)")
    train_contextual(log_every=100)
