"""Stage 4: actor-critic with generalized advantage estimation (GAE).

Stage 3 scored move m by advantage[m] = G[m] - value(s[m]): the rest of its game against the
critic's estimate. Every position of a game was trained toward the same final result, so most
of what the critic and the actor saw was the dice of the moves still to come.

Bootstrapping scores a move by how the critic's own estimate changed right after it, the TD
(temporal difference) error:

    delta[m] = r[m] + gamma * value(s[n]) - value(s[m])     n: the next move of m's game
                                                            (value(s[n]) = 0 after its last)

It depends on what happened between two moves and on the critic, not on the rest of the game:
far less noise, but it is only as right as the critic. GAE mixes the two with lambda in [0, 1]:

    A[m] = delta[m] + (gamma * lambda) * delta[n1] + (gamma * lambda)^2 * delta[n2] + ...

lambda = 0 is the TD error alone. lambda = 1 adds up to G[m] - value(s[m]) (the values cancel
in pairs), stage 3's advantage. In between, a TD error k moves later counts (gamma * lambda)^k.

The values in A are the critic's from before the update (value_estimates), held fixed while
learning. The critic is trained toward A[m] + value(s[m]), the lambda-return, with stage 3's
loss: the net still has the weights that gave those values, so the loss's advantage,
target - value, is exactly A.

td_errors and gae_advantages are stage 4's two steps; the rest is stage 3's (actor_critic.py)
and the shared loop (trainer.Trainer).
"""

from dataclasses import dataclass

import numpy as np
import torch

from catan_rl import actor_critic, trainer
from catan_rl.rollout import Episodes
from catan_rl.trainer import game_rewards, minibatches


@dataclass(frozen=True)
class Config(actor_critic.Config):
    gae_lambda: float = 0.95       # 1: stage 3's advantages G - value; 0: the TD errors alone

    def check(self) -> None:
        super().check()
        trainer.check_number(self, "gae_lambda", low=0, high=1)


def value_estimates(net: actor_critic.ActorCriticNet, episodes: Episodes,
                    minibatch: int = 4096) -> np.ndarray:
    """(M,) float32: the critic's value at each recorded move under the net's current weights,
    computed without gradient, at most `minibatch` moves at a time. Runs on the net's device.
    ValueError if minibatch < 1."""
    device = next(net.parameters()).device
    values = np.empty(episodes.moves, dtype=np.float32)
    with torch.no_grad():
        for batch in minibatches(episodes, minibatch, device):
            values[batch.rows] = net.logits_and_value(batch.obs)[1].cpu().numpy()
    return values


def td_errors(rewards: np.ndarray, values: np.ndarray, game: np.ndarray, games: int,
              gamma: float) -> np.ndarray:
    """(M,) float32 TD error of each move: how much better things looked right after it than
    when it was made,

        delta[m] = rewards[m] + gamma * values[n] - values[m]   n: the next move of m's game
                                                                (values[n] is 0 if there is
                                                                none: the game is over)

    rewards (M,) float: the reward the agent got right after move m. values (M,) float: the
    critic's value at move m. game (M,) int64: the game, 0 .. games - 1, that move m was made
    in. Moves are in the order they were played, the games' moves interleaved (as in Episodes).
    M = 0 gives an empty array. Does not modify its inputs."""
    # Stage 4, step 1.
    raise NotImplementedError("stage 4, step 1: td_errors")


def gae_advantages(rewards: np.ndarray, values: np.ndarray, game: np.ndarray, games: int,
                   gamma: float, lam: float) -> np.ndarray:
    """(M,) float32 GAE advantage of each move: its TD error plus the later TD errors of its
    game, discounted by gamma * lam per move,

        A[m] = delta[m] + (gamma * lam) * delta[n1] + (gamma * lam)^2 * delta[n2] + ...
             = delta[m] + gamma * lam * A[n1]       n1, n2, ...: the later moves of m's game,
                                                    in order (A[m] = delta[m] if there are none)

    where delta = td_errors(rewards, values, game, games, gamma). The arguments are td_errors';
    lam is in [0, 1]. M = 0 gives an empty array. Does not modify its inputs."""
    # Stage 4, step 2.
    raise NotImplementedError("stage 4, step 2: gae_advantages")


class Trainer(actor_critic.Trainer):
    """Actor-critic with GAE advantages (see the module docstring) on the shared loop."""

    algorithm = "gae"
    config_type = Config
    config: Config

    def learn(self, episodes: Episodes) -> dict[str, float]:
        """Statistics: reward (mean over games), return (mean G over moves: what happened),
        target (mean lambda-return: what the critic is trained toward), advantage_std, and
        accumulate_gradients' (its value loss is against the targets), with the explained
        variance measured against G."""
        config = self.config
        game_reward = game_rewards(episodes, config.reward)
        rewards = actor_critic.final_move_rewards(episodes, game_reward)
        # The critic as it is before this update. Its values go into the advantages, which then
        # stay fixed while the gradient is computed, as stage 3's did.
        values = value_estimates(self.net, episodes, config.minibatch)
        advantages = gae_advantages(rewards, values, episodes.game, episodes.games,
                                    config.gamma, config.gae_lambda)
        # The lambda-return. With these targets stage 3's loss scores each move by
        # target - value = advantage: accumulate_gradients runs the same weights on the same
        # moves, so its values are these.
        targets = advantages + values
        stats = actor_critic.accumulate_gradients(self.net, episodes, targets, config.minibatch,
                                                  config.value_coef, config.entropy_coef)
        returns = actor_critic.discounted_returns(rewards, episodes.game, episodes.games,
                                                  config.gamma)
        return {
            "reward": float(game_reward.mean()),
            "return": float(returns.mean()),
            "target": float(targets.mean()),
            "advantage_std": float(advantages.std()),
            **stats,
            # Against what happened (G), as in stage 3, so runs with any lambda compare. Against
            # the targets, which contain the critic's own values, a small lambda would look
            # better than it is.
            "explained_variance": actor_critic.explained_variance(returns, values),
        }
