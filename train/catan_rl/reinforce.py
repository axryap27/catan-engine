"""Stage 2: REINFORCE against bots.

The agent plays one seat against three bots. Each game's result is the reward for every move
the agent made in that game:

    loss = -mean over the agent's moves m of (R[game of m] - baseline) * log pi(m)

R is 1 for a win and 0 otherwise (reward="win"), or the agent's final VP / 10, capped at 1
(reward="vp"). The baseline is the batch's mean R ("mean") or 0 ("none"). The gradient of
this loss is the policy gradient (train/learn/LEARN.md section 4): the dice and the bots'
moves don't depend on the network, so they drop out.

One gradient step per batch of fresh games: the moves were sampled from the current policy,
and after one step they no longer are. (Reusing a batch safely is what PPO adds, stage 5.)
"""

from dataclasses import dataclass

import numpy as np
import torch

from catan_rl import trainer
from catan_rl.policy import PolicyNet
from catan_rl.rollout import Episodes
from catan_rl.trainer import game_rewards, minibatches

BASELINES = ("mean", "none")


@dataclass(frozen=True)
class Config(trainer.Config):
    baseline: str = "mean"         # BASELINES

    def check(self) -> None:
        super().check()
        trainer.check_choice(self, "baseline", BASELINES)


def move_advantages(episodes: Episodes, rewards: np.ndarray, baseline: str) -> np.ndarray:
    """(M,) float32: each move's game reward minus the baseline (rewards.mean() for "mean", 0 for
    "none"). ValueError for an unknown baseline."""
    if baseline == "mean":
        # Subtracting the same number from every move's reward leaves the expected gradient
        # unchanged (the score function has mean zero) but removes the part of the reward that
        # is the same in every game, which is most of the noise: with a win rate near 25%, a
        # plain 0/1 reward would push every move of every game up or down by at least that much.
        base = rewards.mean()
    elif baseline == "none":
        base = 0.0
    else:
        raise ValueError(f"unknown baseline {baseline!r}: use one of {BASELINES}")
    # Every move inherits its whole game's reward: REINFORCE does not try to say which move won.
    return (rewards[episodes.game] - base).astype(np.float32)


def accumulate_gradients(net: PolicyNet, episodes: Episodes, advantages: np.ndarray,
                         minibatch: int = 4096, entropy_coef: float = 0.0) -> dict[str, float]:
    """Adds to every parameter's .grad the gradient of

        loss = -mean_m(advantages[m] * log pi(m)) - entropy_coef * mean_m(entropy at m)

    over all M recorded moves, running the net on at most `minibatch` moves at a time; the
    result equals one full-batch backward. Does not zero the gradients or step the optimizer.
    Runs on the net's device. Returns the means {"loss", "log_prob", "entropy"} over the moves.
    ValueError if the episodes hold no moves."""
    total = episodes.moves
    if total == 0:
        raise ValueError("the episodes hold no moves: play with record=True")
    if minibatch < 1:
        raise ValueError(f"minibatch must be at least 1, got {minibatch}")
    if len(advantages) != total:
        raise ValueError(f"need one advantage per move: {len(advantages)} for {total} moves")
    device = next(net.parameters()).device
    advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=device)

    loss_sum = log_prob_sum = entropy_sum = 0.0
    for batch in minibatches(episodes, minibatch, device):
        log_prob, entropy = batch.log_prob_entropy(net(batch.obs))  # with gradient this time

        # Each chunk's loss is its share of the full loss: a SUM over the chunk's moves divided
        # by the TOTAL number of moves (not the chunk's size). Gradients are linear, so the
        # chunks' gradients add up to the gradient of the full-batch mean -- and backward()
        # adds into .grad. Backward per chunk also frees that chunk's activations, so memory
        # is bounded by `minibatch`, however many moves a batch of games produced.
        chunk_loss = -(advantages_t[batch.rows] * log_prob).sum() / total
        if entropy_coef:  # with 0 there is nothing to differentiate: skip that backward work
            chunk_loss = chunk_loss - entropy_coef * entropy.sum() / total
        chunk_loss.backward()

        loss_sum += chunk_loss.item()
        log_prob_sum += log_prob.detach().sum().item()
        entropy_sum += entropy.detach().sum().item()
    return {"loss": loss_sum, "log_prob": log_prob_sum / total, "entropy": entropy_sum / total}


class Trainer(trainer.Trainer):
    """REINFORCE (see the module docstring) with the shared loop of trainer.Trainer."""

    algorithm = "reinforce"
    config_type = Config
    config: Config

    def make_net(self) -> PolicyNet:
        return PolicyNet(self.config.hidden, self.config.layers)

    def learn(self, episodes: Episodes) -> dict[str, float]:
        """Statistics: reward (mean over games), loss, log_prob and entropy (means over moves)."""
        rewards = game_rewards(episodes, self.config.reward)
        advantages = move_advantages(episodes, rewards, self.config.baseline)
        stats = accumulate_gradients(self.net, episodes, advantages, self.config.minibatch,
                                     self.config.entropy_coef)
        return {"reward": float(rewards.mean()), **stats}
