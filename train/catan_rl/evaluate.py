"""Measuring an agent: the random bot, and matches against random, greedy or itself."""

import math
from dataclasses import dataclass

import torch

from catan_rl.policy import Agent
from catan_rl.rollout import play, random_actions  # noqa: F401  (random_actions: re-exported)


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
    # The game loop lives in rollout.play (training records with it; here nothing is kept).
    episodes = play(agent, opponent, games, first_seed, standard_board, generator, record=False)
    return MatchResult(opponent=opponent, games=episodes.games, wins=int(episodes.won.sum()),
                       capped=int(episodes.capped.sum()),
                       mean_turns=int(episodes.turns.sum()) / episodes.games,
                       agent_decisions=episodes.decisions, seconds=episodes.seconds)
