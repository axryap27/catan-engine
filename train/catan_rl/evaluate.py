"""Measuring an agent: the random bot, and matches against random, greedy or itself."""

import math
import time
from dataclasses import dataclass

import numpy as np
import torch

from catan_rl import actions
from catan_rl.engine import ce
from catan_rl.policy import Agent


def random_actions(obs: np.ndarray, mask: np.ndarray,
                   generator: torch.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The random bot: a uniformly random legal type, then a uniformly random legal argument
    (actions.sample with all-zero logits). Arrays as for Agent.act; returns (types, args) int64.

    The C++ random policy (eval/) is uniform over all legal moves instead, so it ends its turn
    far less often; the two are different baselines."""
    # All-zero logits give every legal type the same probability, and then every legal argument
    # the same probability: exactly the random bot, with no network needed.
    logits = torch.zeros(len(obs), ce.MASK_SIZE)
    types, args, _ = actions.sample(
        logits, torch.as_tensor(mask), actions.phase_of(torch.as_tensor(obs)), generator)
    return types.numpy(), args.numpy()


def wilson(wins: int, games: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for the win rate wins / games (95% for z = 1.96); (0, 1) when
    games is 0."""
    if games == 0:
        return 0.0, 1.0
    # Unlike the textbook p +- z * sqrt(p (1 - p) / n), this interval is sensible for small n
    # and for p near 0 or 1: it never leaves [0, 1] and is not a single point at p = 0.
    p = wins / games
    z2 = z * z
    denom = 1 + z2 / games
    center = (p + z2 / (2 * games)) / denom
    half = z * math.sqrt(p * (1 - p) / games + z2 / (4 * games * games)) / denom
    return max(0.0, center - half), min(1.0, center + half)


@dataclass(frozen=True)
class MatchResult:
    opponent: str
    games: int            # games counted
    wins: int             # counted games the agent won
    capped: int           # counted games the turn cap ended with no winner
    mean_turns: float     # over counted games
    agent_decisions: int  # moves the agent chose in counted games (in "self": every seat's)
    seconds: float        # wall time of the whole match

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    @property
    def ci(self) -> tuple[float, float]:
        return wilson(self.wins, self.games)

    def __str__(self) -> str:
        """One line, e.g. "vs random: won 104/400 = 26.0% [21.9%, 30.5%], 2 capped,
        366 turns/game, 12.3 s"."""
        low, high = self.ci
        return (f"vs {self.opponent}: won {self.wins}/{self.games} = {100 * self.win_rate:.1f}% "
                f"[{100 * low:.1f}%, {100 * high:.1f}%], {self.capped} capped, "
                f"{self.mean_turns:.0f} turns/game, {self.seconds:.1f} s")


def play_match(agent: Agent, opponent: str = "random", games: int = 400, first_seed: int = 1,
               standard_board: bool = False,
               generator: torch.Generator | None = None) -> MatchResult:
    """The agent plays one seat of each game, `opponent` the other three: "random"
    (random_actions), "greedy" (the C++ greedy bot, VecGames.greedy_actions) or "self" (the agent
    in every seat; by symmetry it wins about 25%, so this is a smoke test).

    Game i runs in slot i of VecGames(games, first_seed, standard_board) with the agent in seat
    i % 4, so every seat is played equally often. Only each slot's first game counts: counting
    games as they finish would over-count short ones. The loop steps every slot until all first
    games have ended; a slot whose first game is over keeps playing (random_actions for every
    seat) and is ignored. Each step calls agent.act once, on just the rows it decides.

    `generator` drives the random bot (the agent has its own). The same arguments, agent and
    generator seeds give the same result. ValueError for an unknown opponent or games < 1."""
    if opponent not in ("random", "greedy", "self"):
        raise ValueError(f"unknown opponent {opponent!r}: use 'random', 'greedy' or 'self'")
    if games < 1:
        raise ValueError(f"games must be at least 1, got {games}")
    start = time.perf_counter()

    env = ce.VecGames(games, first_seed, standard_board)
    seat = np.arange(games) % ce.NUM_PLAYERS    # the agent's seat in each slot
    finished = np.zeros(games, dtype=bool)      # has the slot's first game ended?
    wins = capped = turn_total = agent_decisions = 0

    while not finished.all():
        obs, mask, decider = env.observe()
        types = np.zeros(games, dtype=np.int64)
        args = np.zeros(games, dtype=np.int64)

        # Who moves in each slot? The agent, whenever its seat is to act (in "self": every seat
        # is the agent) -- but only in slots still being counted.
        if opponent == "self":
            agent_rows = ~finished
        else:
            agent_rows = ~finished & (decider == seat)
        # A finished slot restarts as a fresh game that nobody scores; random play is the
        # cheapest way to keep it running, since the loop steps all slots together.
        random_rows = finished.copy()
        if opponent == "random":
            random_rows |= ~agent_rows
        elif opponent == "greedy":
            greedy_rows = ~agent_rows & ~finished
            if greedy_rows.any():
                greedy_types, greedy_args = env.greedy_actions()
                types[greedy_rows] = greedy_types[greedy_rows]
                args[greedy_rows] = greedy_args[greedy_rows]

        if random_rows.any():
            types[random_rows], args[random_rows] = random_actions(
                obs[random_rows], mask[random_rows], generator)
        if agent_rows.any():
            types[agent_rows], args[agent_rows], _ = agent.act(obs[agent_rows], mask[agent_rows])
            agent_decisions += int(agent_rows.sum())

        done, winner, _, turns = env.step(types, args)

        # Only a slot's FIRST game counts. If every finished game counted, slots whose games
        # happen to be short would finish (and restart, and finish again) more often and so
        # be over-represented, which biases both the win rate and the game length.
        newly = done & ~finished
        wins += int((winner[newly] == seat[newly]).sum())
        capped += int((winner[newly] == -1).sum())  # the turn cap ended it: nobody won
        turn_total += int(turns[newly].astype(np.int64).sum())
        finished |= newly

    return MatchResult(opponent=opponent, games=games, wins=wins, capped=capped,
                       mean_turns=turn_total / games, agent_decisions=agent_decisions,
                       seconds=time.perf_counter() - start)
