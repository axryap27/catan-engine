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

import dataclasses
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from catan_rl import actions
from catan_rl.evaluate import MatchResult, play_match
from catan_rl.policy import Agent, PolicyNet
from catan_rl.rollout import Episodes, play

REWARDS = ("win", "vp")
BASELINES = ("mean", "none")
OPPONENTS = ("random", "greedy")  # self-play is a later stage

# Evaluation seeds: far from the training generators' (seed and seed + 1) and from each other.
_EVAL_AGENT_OFFSET = 1_000_000_007
_EVAL_BOT_OFFSET = 2_000_000_011


@dataclass(frozen=True)
class Config:
    updates: int = 200             # total updates to train for (raise it and resume to go on)
    games_per_update: int = 64
    lr: float = 1e-3               # Adam (3e-3 learns faster at first, then collapses)
    reward: str = "win"            # REWARDS
    baseline: str = "mean"         # BASELINES
    entropy_coef: float = 0.0      # subtract entropy_coef * mean entropy from the loss
    max_grad_norm: float = 1.0     # clip the gradient's total norm before each step
    minibatch: int = 4096          # moves per forward/backward chunk (memory, not learning)
    hidden: int = 256              # PolicyNet size
    layers: int = 2
    opponent: str = "random"       # "random" or "greedy": who it trains and is scored against
    eval_every: int = 10           # updates between evaluations
    eval_games: int = 200          # games per evaluation
    seed: int = 0                  # network init and every random generator
    train_first_seed: int = 1_000_000  # training games use board seeds from here up; evaluation
                                       # uses play_match's seeds 1.., so the two never overlap


def game_rewards(episodes: Episodes, reward: str) -> np.ndarray:
    """(G,) float32 reward of each game: see the module docstring. ValueError for an unknown
    reward."""
    if reward == "win":
        return episodes.won.astype(np.float32)
    if reward == "vp":
        # Victory points are a denser signal than win/lose: a game lost 9 to 10 still scores
        # 0.9. Dividing by 10 and capping at 1 puts it on the same 0..1 scale as "win".
        return np.minimum(episodes.vp / 10, 1.0).astype(np.float32)
    raise ValueError(f"unknown reward {reward!r}: use one of {REWARDS}")


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
    for start in range(0, total, minibatch):
        chunk = slice(start, start + minibatch)
        obs = torch.as_tensor(episodes.obs[chunk], device=device)
        mask = torch.as_tensor(episodes.mask[chunk], device=device)
        types = torch.as_tensor(episodes.types[chunk], device=device)
        args = torch.as_tensor(episodes.args[chunk], device=device)

        logits = net(obs)  # with gradient this time, unlike Agent.act
        log_prob, entropy = actions.log_prob_entropy(
            logits, mask, actions.phase_of(obs), types, args)

        # Each chunk's loss is its share of the full loss: a SUM over the chunk's moves divided
        # by the TOTAL number of moves (not the chunk's size). Gradients are linear, so the
        # chunks' gradients add up to the gradient of the full-batch mean -- and backward()
        # adds into .grad. Backward per chunk also frees that chunk's activations, so memory
        # is bounded by `minibatch`, however many moves a batch of games produced.
        chunk_loss = -(advantages_t[chunk] * log_prob).sum() / total
        if entropy_coef:  # with 0 there is nothing to differentiate: skip that backward work
            chunk_loss = chunk_loss - entropy_coef * entropy.sum() / total
        chunk_loss.backward()

        loss_sum += chunk_loss.item()
        log_prob_sum += log_prob.detach().sum().item()
        entropy_sum += entropy.detach().sum().item()
    return {"loss": loss_sum, "log_prob": log_prob_sum / total, "entropy": entropy_sum / total}


def _check_config(config: Config) -> None:
    """ValueError for a Config no run could use."""
    def is_int(x) -> bool:
        return isinstance(x, int) and not isinstance(x, bool)

    for name in ("updates", "games_per_update", "minibatch", "hidden", "layers", "eval_every",
                 "eval_games"):
        value = getattr(config, name)
        if not is_int(value) or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    for name in ("seed", "train_first_seed"):
        if not is_int(getattr(config, name)):
            raise ValueError(f"{name} must be an integer, got {getattr(config, name)!r}")
    for name, low_ok in (("lr", False), ("max_grad_norm", False), ("entropy_coef", True)):
        value = getattr(config, name)
        ok = (isinstance(value, (int, float)) and not isinstance(value, bool)
              and math.isfinite(value) and (value >= 0 if low_ok else value > 0))
        if not ok:
            raise ValueError(f"{name} must be a finite number {'>= 0' if low_ok else '> 0'}, "
                             f"got {value!r}")
    for name, choices in (("reward", REWARDS), ("baseline", BASELINES), ("opponent", OPPONENTS)):
        if getattr(config, name) not in choices:
            raise ValueError(f"unknown {name} {getattr(config, name)!r}: use one of {choices}")


class Trainer:
    """Runs REINFORCE, saving everything needed to resume exactly.

    run_dir holds checkpoint.pt (net and optimizer state, config, completed updates, the next
    training seed, and the state of every random generator) and metrics.jsonl (one JSON object
    per line: update 0 is the evaluation before training, then one line per update; see step()
    and run()). If checkpoint.pt exists, the Trainer resumes from it: its config is kept, and a
    `config` that differs in anything but `updates` raises ValueError naming the fields. Lines of
    metrics.jsonl past the checkpoint's update (from a run that died after its last save) are
    dropped. Stopping after k updates and resuming gives the same weights as never stopping."""

    def __init__(self, config: Config, run_dir: str | Path, device: str = "cpu"):
        self.config: Config
        self.run_dir: Path
        self.net: PolicyNet
        self.optimizer: torch.optim.Optimizer
        self.update: int  # completed updates
        _check_config(config)
        self.config = config
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)

        checkpoint = None
        if self.checkpoint_path.exists():
            # weights_only=True: the file holds only tensors and plain Python values, and loading
            # it then cannot run code from a file someone else sent you.
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
            saved = config_from_dict(checkpoint["config"])
            changed = [f.name for f in dataclasses.fields(Config)
                       if f.name != "updates" and getattr(saved, f.name) != getattr(config, f.name)]
            if changed:
                details = ", ".join(f"{n}: checkpoint {getattr(saved, n)!r}, now "
                                    f"{getattr(config, n)!r}" for n in changed)
                raise ValueError(
                    f"config differs from the checkpoint in {self.run_dir} ({details}); only "
                    "'updates' may change on resume; use a new run directory to train with "
                    "other settings")

        # Seeding right before building the net makes the initial weights a function of
        # config.seed alone. fork_rng puts torch's global generator back afterwards, so making
        # a Trainer doesn't reseed the rest of the program. (Everything random after this draws
        # from the two generators below.)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            self.net = PolicyNet(config.hidden, config.layers)
        self.net.to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config.lr)
        # Two generators, one per job, so neither job's draws depend on the other's. The agent's
        # lives on the net's device (its logits are there); the random bot works on the CPU.
        self.agent_gen = torch.Generator(device=self.device).manual_seed(config.seed)
        self.bot_gen = torch.Generator().manual_seed(config.seed + 1)
        self.update = 0
        self.next_seed = config.train_first_seed  # board seed of the next batch's first game

        if checkpoint is not None:
            self.net.load_state_dict(checkpoint["net"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.update = checkpoint["update"]
            self.next_seed = checkpoint["next_seed"]
            self.agent_gen.set_state(checkpoint["agent_gen"])
            self.bot_gen.set_state(checkpoint["bot_gen"])
        # Metrics from after the last save (a crash) or from a run that never saved are stale:
        # the resumed run will write them again.
        self._truncate_metrics(self.update if checkpoint is not None else -1)

    @property
    def checkpoint_path(self) -> Path:
        return self.run_dir / "checkpoint.pt"

    @property
    def metrics_path(self) -> Path:
        return self.run_dir / "metrics.jsonl"

    def step(self) -> dict:
        """One update: play config.games_per_update recorded games against config.opponent on
        the next training seeds, compute rewards and advantages, accumulate the gradient, clip
        it, take one optimizer step. Returns the metrics: update (after this step), games, moves,
        win_rate, mean_vp, mean_turns, capped, reward (mean), loss, log_prob, entropy,
        grad_norm (before clipping), rollout_seconds, update_seconds."""
        config = self.config
        start = time.perf_counter()

        # The Agent shares self.net, so it plays with the weights as they are right now.
        agent = Agent(self.net, self.device, generator=self.agent_gen)
        episodes = play(agent, config.opponent, config.games_per_update,
                        first_seed=self.next_seed, generator=self.bot_gen, record=True)
        rewards = game_rewards(episodes, config.reward)
        advantages = move_advantages(episodes, rewards, config.baseline)

        # One gradient step per batch. The moves were sampled from the policy as it was; after
        # this step it is a different policy, so the same moves would give a biased gradient.
        self.optimizer.zero_grad()
        stats = accumulate_gradients(self.net, episodes, advantages, config.minibatch,
                                     config.entropy_coef)
        # Clipping caps how far one noisy batch can move the weights. It returns the norm from
        # before clipping, which is the useful number to watch.
        grad_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), config.max_grad_norm)
        self.optimizer.step()

        self.update += 1
        # Advance by the batch size: this batch's counted games used seeds next_seed ..
        # next_seed + games - 1 (restarts of finished slots use later seeds, but nobody scores
        # those games), so the next batch's counted games are new boards.
        self.next_seed += config.games_per_update
        return {
            "update": self.update,
            "games": episodes.games,
            "moves": episodes.moves,
            "win_rate": float(episodes.won.mean()),
            "mean_vp": float(episodes.vp.mean()),
            "mean_turns": float(episodes.turns.mean()),
            "capped": int(episodes.capped.sum()),
            "reward": float(rewards.mean()),
            "loss": stats["loss"],
            "log_prob": stats["log_prob"],
            "entropy": stats["entropy"],
            "grad_norm": float(grad_norm),
            "rollout_seconds": episodes.seconds,
            "update_seconds": time.perf_counter() - start,
        }

    def evaluate(self) -> MatchResult:
        """play_match against config.opponent with config.eval_games games on its default seeds,
        sampling from the current net with a generator seeded from (config.seed, self.update),
        so evaluation never disturbs the training generators."""
        # Own generators, seeded by plain arithmetic from (seed, update): evaluation draws
        # nothing from the training generators, so evaluating or not leaves training unchanged
        # (which is what lets a resumed run match an uninterrupted one), and the same net at
        # the same update always evaluates the same way. The offsets keep these seeds away from
        # the training generators' (seed and seed + 1) and from each other's.
        base = self.config.seed * 1_000_003 + self.update
        agent = Agent(self.net, self.device,
                      generator=torch.Generator(device=self.device).manual_seed(
                          base + _EVAL_AGENT_OFFSET))
        bot_gen = torch.Generator().manual_seed(base + _EVAL_BOT_OFFSET)
        # The default first_seed: the same evaluation boards every time, so differences between
        # evaluations come from the net, not from easier or harder boards.
        return play_match(agent, self.config.opponent, games=self.config.eval_games,
                          generator=bot_gen)

    def save(self) -> None:
        """Writes checkpoint.pt atomically (to a temporary file, then renamed)."""
        # The generators' states are saved too: the next batch of games depends on where each
        # random stream stopped, and a resumed run that restarted the streams would play
        # different games and end up with different weights.
        state = {
            "config": dataclasses.asdict(self.config),
            "net": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update": self.update,
            "next_seed": self.next_seed,
            "agent_gen": self.agent_gen.get_state(),
            "bot_gen": self.bot_gen.get_state(),
        }
        # Write the whole file under another name, then rename it over the checkpoint (atomic):
        # a crash while writing leaves the previous checkpoint intact, never half a file.
        tmp = self.checkpoint_path.with_name(self.checkpoint_path.name + ".tmp")
        with open(tmp, "wb") as f:
            torch.save(state, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.checkpoint_path)

    def _truncate_metrics(self, keep_through: int) -> None:
        """Drops the lines of metrics.jsonl whose update is past keep_through, and any line that
        is not a complete JSON object with an update (a write cut short by a crash)."""
        if not self.metrics_path.exists():
            return
        lines = self.metrics_path.read_text().splitlines()
        kept = []
        for line in lines:
            try:
                if json.loads(line)["update"] <= keep_through:
                    kept.append(line)
            except (ValueError, KeyError, TypeError):
                pass
        if len(kept) != len(lines):
            tmp = self.metrics_path.with_name(self.metrics_path.name + ".tmp")
            tmp.write_text("".join(line + "\n" for line in kept))
            os.replace(tmp, self.metrics_path)

    def _append_metrics(self, metrics: dict) -> None:
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")

    def run(self, log: Callable[[str], None] | None = print) -> None:
        """Trains until config.updates updates are done. Evaluates before the first update (on
        a fresh run), every config.eval_every updates and after the last, adding eval_win_rate,
        eval_ci_low, eval_ci_high and eval_games to that update's metrics line. Appends every
        metrics line to metrics.jsonl and saves a checkpoint after each evaluation. `log` gets
        one human-readable line per update and per evaluation (None: silent)."""
        config = self.config
        say = log if log is not None else (lambda line: None)
        width = len(str(config.updates))

        def evaluate_now() -> dict:
            result = self.evaluate()
            say(f"eval   update {self.update:>{width}}: {result}")
            low, high = result.ci
            return {"eval_win_rate": result.win_rate, "eval_ci_low": low, "eval_ci_high": high,
                    "eval_games": result.games}

        if not self.checkpoint_path.exists():  # a fresh run: how good is the untrained net?
            self._append_metrics({"update": self.update, **evaluate_now()})
            self.save()

        while self.update < config.updates:
            metrics = self.step()
            say(f"update {metrics['update']:>{width}}/{config.updates}  "
                f"win {100 * metrics['win_rate']:.0f}% (batch)  vp {metrics['mean_vp']:.1f}  "
                f"entropy {metrics['entropy']:.2f}  {metrics['update_seconds']:.1f} s")
            due = self.update % config.eval_every == 0 or self.update == config.updates
            if due:
                metrics.update(evaluate_now())
            # The metrics line goes to disk before the checkpoint: a crash in between leaves a
            # line past the checkpoint (dropped on resume), never a checkpoint with a missing line.
            self._append_metrics(metrics)
            if due:
                self.save()


def config_from_dict(d: dict) -> Config:
    """Inverse of dataclasses.asdict(config); ValueError for unknown keys."""
    unknown = sorted(set(d) - {f.name for f in dataclasses.fields(Config)})
    if unknown:
        raise ValueError(f"unknown config field(s): {', '.join(unknown)}")
    return Config(**d)


def saved_config(run_dir: str | Path) -> Config | None:
    """The config in run_dir's checkpoint, or None if there is no checkpoint."""
    path = Path(run_dir) / "checkpoint.pt"
    if not path.exists():
        return None
    return config_from_dict(torch.load(path, map_location="cpu", weights_only=True)["config"])
