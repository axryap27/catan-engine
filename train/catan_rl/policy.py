"""The policy network, and an Agent that plays games with it."""

import numpy as np
import torch
from torch import nn

from catan_rl import actions
from catan_rl.engine import ce


class PolicyNet(nn.Module):
    """observation (B, OBS_SIZE) -> logits (B, MASK_SIZE), one per legal-move mask entry.

    A plain MLP: log1p of the input (the observation is small counts and 0/1 flags; log1p keeps
    0 at 0 and stops large counts such as the turn number from dominating), then `layers` times
    [Linear, LayerNorm, ReLU] of width `hidden`, then a Linear to MASK_SIZE. The output layer
    starts with tiny weights (orthogonal init, gain 0.01) and zero bias, so an untrained net's
    logits are all about 0 and it plays like the random bot: a uniformly random legal type, then
    a uniformly random legal argument."""

    def __init__(self, hidden: int = 256, layers: int = 2):
        super().__init__()
        blocks: list[nn.Module] = []
        width = ce.OBS_SIZE
        for _ in range(layers):
            # LayerNorm rescales each sample's activations to mean 0 and variance 1, which keeps
            # training stable when inputs differ a lot in scale.
            blocks += [nn.Linear(width, hidden), nn.LayerNorm(hidden), nn.ReLU()]
            width = hidden
        self.body = nn.Sequential(*blocks)
        self.head = nn.Linear(width, ce.MASK_SIZE)
        # Tiny output weights make every logit about 0 at the start, so the first policy is
        # (almost) uniform over the legal moves and PPO starts from random play, not from some
        # arbitrary preference.
        nn.init.orthogonal_(self.head.weight, gain=0.01)
        nn.init.zeros_(self.head.bias)

    def features(self, obs: torch.Tensor) -> torch.Tensor:
        """obs (B, OBS_SIZE), uint8 or float, on the net's device -> the body's output (B, hidden):
        what the head turns into logits (and, in stage 3, what a value head can read too)."""
        return self.body(torch.log1p(obs.float()))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: uint8 or float, on the net's device."""
        return self.head(self.features(obs))


class Agent:
    """A PolicyNet that plays: numpy arrays in and out, no gradients.

    `generator` (on `device`) makes sampling reproducible; None uses torch's global RNG.
    deterministic=True plays actions.mode() instead of sampling."""

    def __init__(self, net: PolicyNet, device: str | torch.device = "cpu",
                 generator: torch.Generator | None = None, deterministic: bool = False):
        # Module.to moves the net in place and returns it: the Agent shares the caller's net, so
        # a trainer that updates the net's weights is also updating how this Agent plays.
        self.net = net.to(device)
        self.device = torch.device(device)
        self.generator = generator
        self.deterministic = deterministic

    def act(self, obs: np.ndarray, mask: np.ndarray
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """obs (B, OBS_SIZE) uint8 and mask (B, MASK_SIZE) uint8, as from VecGames.observe() or
        a subset of its rows; B = 0 is allowed. Returns (types int64 (B,), args int64 (B,),
        log_prob float32 (B,)): each chosen move and its log-probability under the net. Runs
        under torch.no_grad()."""
        if len(obs) == 0:
            return (np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float32))
        with torch.no_grad():  # playing, not learning: skip building the autograd graph
            obs_t = torch.as_tensor(obs, device=self.device)
            mask_t = torch.as_tensor(mask, device=self.device)
            logits = self.net(obs_t)
            phases = actions.phase_of(obs_t)
            if self.deterministic:
                types, args = actions.mode(logits, mask_t, phases)
                log_prob, _ = actions.log_prob_entropy(logits, mask_t, phases, types, args)
            else:
                types, args, log_prob = actions.sample(logits, mask_t, phases, self.generator)
        return (types.cpu().numpy(), args.cpu().numpy(),
                log_prob.to(torch.float32).cpu().numpy())
