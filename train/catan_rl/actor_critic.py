"""Stage 3: actor-critic against bots.

The network gets a second output, the critic: value(s), its estimate of the return from
position s. With reward "win" and gamma 1 that is its chance of winning from s. Each move's
advantage is then

    advantage[m] = G[m] - value(s[m])          G[m]: the move's return (discounted_returns)

how much better the game went than the critic expected when the move was made. REINFORCE
(stage 2) gave every move of a game the same G - mean(R). Here the baseline depends on the
position: a game won from a position the critic rated hopeless gives its moves a lot of credit,
a game that was won already gives little. The baseline still doesn't depend on which move was
chosen, so the policy gradient stays unbiased; the better the critic predicts G, the less noise
is left in it (explained_variance measures how much).

    loss    = policy + value_coef * value - entropy_coef * entropy
    policy  = -mean over moves of advantage * log pi(move), the advantage held constant
    value   = mean over moves of (value(s) - G)^2: the critic learns to predict G
    entropy = mean over moves of the move distribution's entropy

ActorCriticNet, discounted_returns and actor_critic_loss are stage 3's three steps; the rest
is stage 2's loop (trainer.Trainer).
"""

from dataclasses import dataclass

import numpy as np
import torch

from catan_rl import trainer
from catan_rl.policy import PolicyNet
from catan_rl.rollout import Episodes
from catan_rl.trainer import game_rewards, minibatches


@dataclass(frozen=True)
class Config(trainer.Config):
    gamma: float = 1.0             # discount per agent move; 1: every move's return is R
    value_coef: float = 0.5        # weight of the critic's loss in the total

    def check(self) -> None:
        super().check()
        trainer.check_number(self, "gamma", low=0, high=1)
        trainer.check_number(self, "value_coef", low=0)


class ActorCriticNet(PolicyNet):
    """A PolicyNet (the actor) plus a critic: logits_and_value(obs) gives both outputs.

    forward(obs) is still PolicyNet's logits, so an Agent plays with it unchanged. The critic
    gives one number per observation, its estimate of the return from there. Whether it reads
    the actor's features (PolicyNet.features: one body, two heads) or has a body of its own is
    a design choice. Either way the actor is exactly a PolicyNet's, built first, so an
    ActorCriticNet built under a torch seed starts with the same policy as a PolicyNet built
    under that seed."""

    def __init__(self, hidden: int = 256, layers: int = 2):
        super().__init__(hidden, layers)
        # Stage 3, step 1: the critic's layers.
        raise NotImplementedError("stage 3, step 1: ActorCriticNet's critic")

    def logits_and_value(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """obs (B, OBS_SIZE), uint8 or float, on the net's device -> (logits (B, MASK_SIZE),
        equal to forward(obs); value (B,) float32)."""
        # Stage 3, step 1.
        raise NotImplementedError("stage 3, step 1: ActorCriticNet.logits_and_value")


def final_move_rewards(episodes: Episodes, rewards: np.ndarray) -> np.ndarray:
    """(M,) float32 reward after each recorded move: game g's reward rewards[g] after the
    agent's last move in game g, 0 after every other move."""
    out = np.zeros(episodes.moves, dtype=np.float32)
    # The moves are in the order they were played, so a game's last move is its largest index.
    last = np.full(episodes.games, -1, dtype=np.int64)
    np.maximum.at(last, episodes.game, np.arange(episodes.moves))
    played = last >= 0
    out[last[played]] = rewards[played]
    return out


def discounted_returns(rewards: np.ndarray, game: np.ndarray, games: int,
                       gamma: float) -> np.ndarray:
    """(M,) float32 return of each move: its reward plus the later rewards of its game,
    discounted by gamma per move,

        G[m] = rewards[m] + gamma * rewards[n1] + gamma^2 * rewards[n2] + ...
             = rewards[m] + gamma * G[n1]       n1, n2, ...: the later moves of m's game, in order
                                                (G[m] = rewards[m] if there are none)

    rewards (M,) float: the reward the agent got right after move m. game (M,) int64: the game,
    0 .. games - 1, that move m was made in. Moves are in the order they were played, the games'
    moves interleaved (as in Episodes). M = 0 gives an empty array. Does not modify its
    inputs."""
    # Stage 3, step 2.
    raise NotImplementedError("stage 3, step 2: discounted_returns")


@dataclass(frozen=True)
class Losses:
    """actor_critic_loss's result: scalar tensors, each a mean over the moves."""

    total: torch.Tensor    # policy + value_coef * value - entropy_coef * entropy: backward() this
    policy: torch.Tensor   # -mean(advantage * log_prob), advantage = returns - value
    value: torch.Tensor    # mean((value - returns)^2)
    entropy: torch.Tensor  # mean(entropy)


def actor_critic_loss(log_prob: torch.Tensor, entropy: torch.Tensor, value: torch.Tensor,
                      returns: torch.Tensor, value_coef: float, entropy_coef: float) -> Losses:
    """The stage 3 loss (module docstring) on B moves: log_prob, entropy and value (B,) are the
    net's outputs, with gradient; returns (B,) is the moves' returns G. The advantage is a
    constant: no gradient flows through it, so the policy term trains only the actor and the
    critic learns only from the value term."""
    # Stage 3, step 3.
    raise NotImplementedError("stage 3, step 3: actor_critic_loss")


def explained_variance(returns: np.ndarray, values: np.ndarray) -> float:
    """1 - Var(returns - values) / Var(returns): the share of the returns' variance the critic
    predicts. 1 is perfect, 0 is no better than predicting the mean, negative is worse. NaN if
    the returns are all equal (nothing to explain)."""
    if np.ptp(returns) == 0:  # exact: np.var of equal float32s can come out at 1e-15, not 0
        return float("nan")
    returns = np.asarray(returns, dtype=np.float64)
    return 1.0 - float(np.var(returns - values)) / float(np.var(returns))


def accumulate_gradients(net: ActorCriticNet, episodes: Episodes, returns: np.ndarray,
                         minibatch: int = 4096, value_coef: float = 0.5,
                         entropy_coef: float = 0.0) -> dict[str, float]:
    """Adds to every parameter's .grad the gradient of actor_critic_loss over all M recorded
    moves (returns (M,): each move's G), running the net on at most `minibatch` moves at a time;
    the result equals one full-batch backward. Does not zero the gradients or step the
    optimizer. Runs on the net's device. Returns the means over the moves of loss, policy_loss,
    value_loss, entropy, log_prob and value, and the critic's explained_variance. ValueError if
    the episodes hold no moves."""
    total = episodes.moves
    if total == 0:
        raise ValueError("the episodes hold no moves: play with record=True")
    if len(returns) != total:
        raise ValueError(f"need one return per move: {len(returns)} for {total} moves")
    device = next(net.parameters()).device
    returns_t = torch.as_tensor(returns, dtype=torch.float32, device=device)

    sums = dict.fromkeys(("loss", "policy_loss", "value_loss", "entropy", "log_prob"), 0.0)
    values = np.empty(total, dtype=np.float32)
    for batch in minibatches(episodes, minibatch, device):
        logits, value = net.logits_and_value(batch.obs)
        log_prob, entropy = batch.log_prob_entropy(logits)
        losses = actor_critic_loss(log_prob, entropy, value, returns_t[batch.rows], value_coef,
                                   entropy_coef)
        # The losses are means over this chunk. Weighted by the chunk's share of all the moves,
        # the chunks' gradients add up to the gradient of the full-batch mean (as in stage 2).
        weight = len(batch) / total
        (weight * losses.total).backward()

        sums["loss"] += weight * losses.total.item()
        sums["policy_loss"] += weight * losses.policy.item()
        sums["value_loss"] += weight * losses.value.item()
        sums["entropy"] += weight * losses.entropy.item()
        sums["log_prob"] += log_prob.detach().sum().item() / total
        values[batch.rows] = value.detach().cpu().numpy()
    return {**sums, "value": float(values.mean()),
            "explained_variance": explained_variance(returns, values)}


class Trainer(trainer.Trainer):
    """Actor-critic (see the module docstring) with the shared loop of trainer.Trainer."""

    algorithm = "actor-critic"
    config_type = Config
    config: Config

    def make_net(self) -> ActorCriticNet:
        return ActorCriticNet(self.config.hidden, self.config.layers)

    def learn(self, episodes: Episodes) -> dict[str, float]:
        """Statistics: reward (mean over games), return (mean G over moves), and
        accumulate_gradients'."""
        rewards = game_rewards(episodes, self.config.reward)
        returns = discounted_returns(final_move_rewards(episodes, rewards), episodes.game,
                                     episodes.games, self.config.gamma)
        stats = accumulate_gradients(self.net, episodes, returns, self.config.minibatch,
                                     self.config.value_coef, self.config.entropy_coef)
        return {"reward": float(rewards.mean()), "return": float(returns.mean()), **stats}

    def log_line(self, metrics: dict) -> str:
        return (f"{super().log_line(metrics)}  value loss {metrics['value_loss']:.3f}  "
                f"explained {metrics['explained_variance']:.2f}")
