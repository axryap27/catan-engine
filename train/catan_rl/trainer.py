"""The training loop every algorithm shares: play games, learn from them, evaluate, checkpoint.

An algorithm is a Trainer subclass: a Config subclass with its own settings, make_net() (the
untrained network) and learn() (the gradient of its loss on a batch of freshly played games).
The base class does everything else, the same way for every algorithm:

    step()   play config.games_per_update recorded games against config.opponent, learn(),
             clip the gradient, take one Adam step
    run()    step until config.updates, evaluating every config.eval_every updates and saving a
             checkpoint after each evaluation; a run directory resumes exactly where it stopped
"""

import abc
import dataclasses
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, ClassVar, Iterator

import numpy as np
import torch

from catan_rl import actions
from catan_rl.evaluate import MatchResult, play_match
from catan_rl.policy import Agent, PolicyNet
from catan_rl.rollout import Episodes, play

REWARDS = ("win", "vp")
OPPONENTS = ("random", "greedy")  # self-play is a later stage

# Evaluation seeds: far from the training generators' (seed and seed + 1) and from each other.
_EVAL_AGENT_OFFSET = 1_000_000_007
_EVAL_BOT_OFFSET = 2_000_000_011


@dataclass(frozen=True)
class Config:
    """The settings every algorithm has; each algorithm's Config adds its own after these."""

    updates: int = 200             # total updates to train for (raise it and resume to go on)
    games_per_update: int = 64
    lr: float = 1e-3               # Adam (3e-3 learns faster at first, then collapses)
    reward: str = "win"            # REWARDS
    entropy_coef: float = 0.0      # subtract entropy_coef * mean entropy from the loss
    max_grad_norm: float = 1.0     # clip the gradient's total norm before each step
    minibatch: int = 4096          # moves per forward/backward chunk (memory, not learning)
    hidden: int = 256              # network size
    layers: int = 2
    opponent: str = "random"       # OPPONENTS: who it trains and is scored against
    eval_every: int = 10           # updates between evaluations
    eval_games: int = 200          # games per evaluation
    seed: int = 0                  # network init and every random generator
    train_first_seed: int = 1_000_000  # training games use board seeds from here up; evaluation
                                       # uses play_match's seeds 1.., so the two never overlap

    def check(self) -> None:
        """ValueError for settings no run could use. A subclass with fields of its own extends
        this (calling super().check() first)."""
        for name in ("updates", "games_per_update", "minibatch", "hidden", "layers",
                     "eval_every", "eval_games"):
            check_int(self, name, low=1)
        # Seeds stay far below the integer limits of the generators (int64) and of the
        # engine's board seeds (uint64), which the evaluation offsets and next_seed add to.
        check_int(self, "seed", low=0, high=2**32 - 1)
        check_int(self, "train_first_seed", low=0, high=2**62)
        check_number(self, "lr", low=0, low_ok=False)
        check_number(self, "max_grad_norm", low=0, low_ok=False)
        check_number(self, "entropy_coef", low=0)
        check_choice(self, "reward", REWARDS)
        check_choice(self, "opponent", OPPONENTS)


def check_int(config: Config, name: str, low: int | None = None,
              high: int | None = None) -> None:
    """ValueError unless config.<name> is an int (not a bool) in [low, high] (either bound may
    be None: no bound)."""
    value = getattr(config, name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if low is not None and value < low:
        raise ValueError(f"{name} must be at least {low}, got {value!r}")
    if high is not None and value > high:
        raise ValueError(f"{name} must be at most {high}, got {value!r}")


def check_number(config: Config, name: str, low: float, high: float = math.inf,
                 low_ok: bool = True) -> None:
    """ValueError unless config.<name> is a finite int or float in [low, high] (or (low, high]
    with low_ok=False)."""
    value = getattr(config, name)
    ok = (isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
          and (value >= low if low_ok else value > low) and value <= high)
    if not ok:
        bounds = f"{'>=' if low_ok else '>'} {low}" + (f" and <= {high}" if high < math.inf else "")
        raise ValueError(f"{name} must be a finite number {bounds}, got {value!r}")


def check_choice(config: Config, name: str, choices: tuple[str, ...]) -> None:
    """ValueError unless config.<name> is one of `choices`."""
    if getattr(config, name) not in choices:
        raise ValueError(f"unknown {name} {getattr(config, name)!r}: use one of {choices}")


def game_rewards(episodes: Episodes, reward: str) -> np.ndarray:
    """(G,) float32 reward of each game: 1 if the agent won, else 0 (reward="win"); or its final
    VP / 10, capped at 1 (reward="vp"). ValueError for an unknown reward."""
    if reward == "win":
        return episodes.won.astype(np.float32)
    if reward == "vp":
        # Victory points are a denser signal than win/lose: a game lost 9 to 10 still scores
        # 0.9. Dividing by 10 and capping at 1 puts it on the same 0..1 scale as "win".
        return np.minimum(episodes.vp / 10, 1.0).astype(np.float32)
    raise ValueError(f"unknown reward {reward!r}: use one of {REWARDS}")


@dataclass(frozen=True)
class Minibatch:
    """Moves `rows` of an Episodes, as tensors on one device."""

    rows: slice
    obs: torch.Tensor     # (B, OBS_SIZE) uint8
    mask: torch.Tensor    # (B, MASK_SIZE) uint8
    types: torch.Tensor   # (B,) int64
    args: torch.Tensor    # (B,) int64
    phases: torch.Tensor  # (B,) int64, read from obs

    def __len__(self) -> int:
        return len(self.types)

    def log_prob_entropy(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """actions.log_prob_entropy of these moves under `logits` (B, MASK_SIZE)."""
        return actions.log_prob_entropy(logits, self.mask, self.phases, self.types, self.args)


def minibatches(episodes: Episodes, size: int, device: torch.device) -> Iterator[Minibatch]:
    """The episodes' moves in order, at most `size` at a time (made one minibatch at a time, as
    they are iterated). ValueError if size < 1, raised by this call."""
    if size < 1:
        raise ValueError(f"minibatch must be at least 1, got {size}")
    return _minibatches(episodes, size, device)


def _minibatches(episodes: Episodes, size: int, device: torch.device) -> Iterator[Minibatch]:
    for start in range(0, episodes.moves, size):
        rows = slice(start, start + size)
        obs = torch.as_tensor(episodes.obs[rows], device=device)
        yield Minibatch(rows=rows, obs=obs,
                        mask=torch.as_tensor(episodes.mask[rows], device=device),
                        types=torch.as_tensor(episodes.types[rows], device=device),
                        args=torch.as_tensor(episodes.args[rows], device=device),
                        phases=actions.phase_of(obs))


class Trainer(abc.ABC):
    """Trains one algorithm, saving everything needed to resume exactly.

    run_dir holds checkpoint.pt (the algorithm's name, its config, the net and optimizer state,
    completed updates, the next training seed, and the state of every random generator) and
    metrics.jsonl (one JSON object per line: update 0 is the evaluation before training, then one
    line per update; see step() and run(); a statistic that is undefined for a batch, NaN in
    memory, is null there). `config` must be exactly the algorithm's config_type (TypeError
    otherwise). If checkpoint.pt exists, the Trainer resumes from it: a checkpoint of another
    algorithm, or a `config` that differs in anything but `updates`, raises ValueError. Lines
    of metrics.jsonl past the checkpoint's update (from a run that died after its last save) are
    dropped. Stopping after k updates and resuming gives the same weights as never stopping."""

    algorithm: ClassVar[str]            # its name, saved in (and checked against) checkpoints
    config_type: ClassVar[type[Config]]

    def __init__(self, config: Config, run_dir: str | Path, device: str = "cpu"):
        self.config: Config
        self.run_dir: Path
        self.net: PolicyNet
        self.optimizer: torch.optim.Optimizer
        self.update: int  # completed updates
        if type(config) is not self.config_type:
            raise TypeError(f"{type(self).__name__} needs a {self.config_type.__qualname__}, "
                            f"got a {type(config).__qualname__}")
        config.check()
        self.config = config
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)

        checkpoint = self._read_checkpoint(self.checkpoint_path)
        if checkpoint is not None:
            saved = self.config_from_dict(checkpoint["config"])
            changed = [f.name for f in dataclasses.fields(config)
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
            self.net = self.make_net()
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

    # ---- what an algorithm defines ----------------------------------------------------------

    @abc.abstractmethod
    def make_net(self) -> PolicyNet:
        """The untrained network for self.config (called once, under a fixed torch seed)."""

    @abc.abstractmethod
    def learn(self, episodes: Episodes) -> dict[str, float]:
        """Adds to the net's (zeroed) .grad the gradient of the algorithm's loss on `episodes`,
        games just played by the current net, and returns that batch's statistics for the
        metrics line: at least "reward" (the mean game reward), "loss" and "entropy"."""

    def log_line(self, metrics: dict) -> str:
        """The line run() logs after each update. A subclass may append its own statistics."""
        return (f"update {metrics['update']:>{len(str(self.config.updates))}}/"
                f"{self.config.updates}  win {100 * metrics['win_rate']:.0f}% (batch)  "
                f"vp {metrics['mean_vp']:.1f}  entropy {metrics['entropy']:.2f}  "
                f"{metrics['update_seconds']:.1f} s")

    # ---- the loop ---------------------------------------------------------------------------

    @property
    def checkpoint_path(self) -> Path:
        return self.run_dir / "checkpoint.pt"

    @property
    def metrics_path(self) -> Path:
        return self.run_dir / "metrics.jsonl"

    def step(self) -> dict:
        """One update: play config.games_per_update recorded games against config.opponent on
        the next training seeds, learn() from them, clip the gradient, take one optimizer step.
        Returns the metrics: update (after this step), games, moves, win_rate, mean_vp,
        mean_turns, capped, learn()'s statistics, grad_norm (before clipping), rollout_seconds,
        update_seconds."""
        config = self.config
        start = time.perf_counter()

        # The Agent shares self.net, so it plays with the weights as they are right now.
        agent = Agent(self.net, self.device, generator=self.agent_gen)
        episodes = play(agent, config.opponent, config.games_per_update,
                        first_seed=self.next_seed, generator=self.bot_gen, record=True)

        # One gradient step per batch. The moves were sampled from the policy as it was; after
        # this step it is a different policy, so the same moves would give a biased gradient.
        self.optimizer.zero_grad()
        stats = self.learn(episodes)
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
            **stats,
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
            "algorithm": self.algorithm,
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

    def run(self, log: Callable[[str], None] | None = print) -> None:
        """Trains until config.updates updates are done. Evaluates before the first update (on
        a fresh run), every config.eval_every updates and after the last, adding eval_win_rate,
        eval_ci_low, eval_ci_high and eval_games to that update's metrics line. Appends every
        metrics line to metrics.jsonl and saves a checkpoint after each evaluation. `log` gets
        one human-readable line per update and per evaluation (None: silent)."""
        config = self.config
        say = log if log is not None else (lambda line: None)

        def evaluate_now() -> dict:
            result = self.evaluate()
            say(f"eval   update {self.update:>{len(str(config.updates))}}: {result}")
            low, high = result.ci
            return {"eval_win_rate": result.win_rate, "eval_ci_low": low, "eval_ci_high": high,
                    "eval_games": result.games}

        if not self.checkpoint_path.exists():  # a fresh run: how good is the untrained net?
            self._append_metrics({"update": self.update, **evaluate_now()})
            self.save()

        while self.update < config.updates:
            metrics = self.step()
            say(self.log_line(metrics))
            due = self.update % config.eval_every == 0 or self.update == config.updates
            if due:
                metrics.update(evaluate_now())
            # The metrics line goes to disk before the checkpoint: a crash in between leaves a
            # line past the checkpoint (dropped on resume), never a checkpoint with a missing line.
            self._append_metrics(metrics)
            if due:
                self.save()

    # ---- checkpoints and metrics files ------------------------------------------------------

    @classmethod
    def config_from_dict(cls, d: dict) -> Config:
        """Inverse of dataclasses.asdict(config) for this algorithm's Config; ValueError for
        unknown keys."""
        unknown = sorted(set(d) - {f.name for f in dataclasses.fields(cls.config_type)})
        if unknown:
            raise ValueError(f"unknown config field(s): {', '.join(unknown)}")
        return cls.config_type(**d)

    @classmethod
    def saved_config(cls, run_dir: str | Path) -> Config | None:
        """The config in run_dir's checkpoint, or None if there is no checkpoint. ValueError if
        the checkpoint is another algorithm's."""
        checkpoint = cls._read_checkpoint(Path(run_dir) / "checkpoint.pt")
        return None if checkpoint is None else cls.config_from_dict(checkpoint["config"])

    @classmethod
    def _read_checkpoint(cls, path: Path) -> dict | None:
        """The checkpoint at `path`, or None if there is none. ValueError if it is another
        algorithm's."""
        if not path.exists():
            return None
        # weights_only=True: the file holds only tensors and plain Python values, and loading it
        # then cannot run code from a file someone else sent you.
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        # Checkpoints from before there was more than one algorithm are all REINFORCE runs.
        algorithm = checkpoint.get("algorithm", "reinforce")
        if algorithm != cls.algorithm:
            raise ValueError(f"{path.parent} holds a {algorithm} run, not {cls.algorithm}; use "
                             "a new run directory")
        return checkpoint

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
        # NaN (say, the explained variance of a batch whose returns were all equal) isn't JSON:
        # json.dumps would write a bare NaN, which strict parsers reject. It goes in as null.
        line = {k: None if isinstance(v, float) and not math.isfinite(v) else v
                for k, v in metrics.items()}
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(line, allow_nan=False) + "\n")
