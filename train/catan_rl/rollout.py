"""Playing batches of games against bots, optionally recording every move the agent makes.

play() is the one game loop: play_match (evaluate.py) measures with it and training learns from
what it records. Slot i of a VecGames(games, first_seed, standard_board) is game i, with the
agent in seat i % 4 so every seat is played equally often. Only each slot's first game counts:
counting games as they finish would over-count short ones (and, in training, bias the gradient
toward whatever makes games short). The loop steps every slot until all first games have ended;
a slot whose first game is over keeps playing (random_actions for every seat) and is ignored.
Each step calls agent.act once, on just the rows it decides.
"""

import time
from dataclasses import dataclass

import numpy as np
import torch

from catan_rl import actions
from catan_rl.engine import ce
from catan_rl.policy import Agent

OPPONENTS = ("random", "greedy", "self")


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


@dataclass(frozen=True)
class Episodes:
    """What happened in one batch of games: every move the agent made (M rows, in the order they
    were played) and how each game ended (G = games rows, by slot). All numpy arrays.

    Without record=True the move arrays have 0 rows; `decisions` still counts the moves."""

    obs: np.ndarray        # (M, OBS_SIZE) uint8: what the agent saw
    mask: np.ndarray       # (M, MASK_SIZE) uint8: its legal moves
    types: np.ndarray      # (M,) int64: the move it chose
    args: np.ndarray       # (M,) int64
    log_prob: np.ndarray   # (M,) float32: log-probability of that move when it was played
    game: np.ndarray       # (M,) int64: the slot the move was made in
    seat: np.ndarray       # (G,) int64: the agent's seat in each game
    won: np.ndarray        # (G,) bool: the agent won
    capped: np.ndarray     # (G,) bool: the turn cap ended the game with no winner
    vp: np.ndarray         # (G,) int64: the agent's true final victory points
    turns: np.ndarray      # (G,) int64: the game's length in turns
    decisions: int         # moves the agent made in counted games (in "self": every seat's)
    seconds: float         # wall time

    @property
    def games(self) -> int:
        return len(self.seat)

    @property
    def moves(self) -> int:
        return len(self.types)


def play(agent: Agent, opponent: str = "random", games: int = 64, first_seed: int = 1,
         standard_board: bool = False, generator: torch.Generator | None = None,
         record: bool = False) -> Episodes:
    """Plays `games` complete games, the agent in one seat and `opponent` in the other three:
    "random" (random_actions), "greedy" (the C++ greedy bot, VecGames.greedy_actions) or "self"
    (the agent in every seat). See the module docstring for slots, seats and counting.

    record=True keeps every move the agent made in counted games. `generator` drives the random
    bot (the agent has its own). The same arguments, agent and generator seeds give the same
    result. ValueError for an unknown opponent, games < 1, or record=True with "self"
    (self-play training is a later stage)."""
    if opponent not in OPPONENTS:
        raise ValueError(f"unknown opponent {opponent!r}: use 'random', 'greedy' or 'self'")
    if games < 1:
        raise ValueError(f"games must be at least 1, got {games}")
    if record and opponent == "self":
        raise ValueError("record=True is not supported with opponent 'self' "
                         "(self-play training is a later stage)")
    start = time.perf_counter()

    env = ce.VecGames(games, first_seed, standard_board)
    seat = np.arange(games, dtype=np.int64) % ce.NUM_PLAYERS  # the agent's seat in each slot
    finished = np.zeros(games, dtype=bool)                    # has the slot's first game ended?
    won = np.zeros(games, dtype=bool)
    capped = np.zeros(games, dtype=bool)
    vp = np.zeros(games, dtype=np.int64)
    turns = np.zeros(games, dtype=np.int64)
    decisions = 0
    # One small chunk per step; they are joined once at the end. Growing one big array move by
    # move would copy it again and again.
    chunks: dict[str, list[np.ndarray]] = {
        "obs": [], "mask": [], "types": [], "args": [], "log_prob": [], "game": []}

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

        # The random bot draws before the agent does, step by step: when both share one
        # generator (or torch's global one) this order fixes who gets which random numbers.
        if random_rows.any():
            types[random_rows], args[random_rows] = random_actions(
                obs[random_rows], mask[random_rows], generator)
        if agent_rows.any():
            agent_obs, agent_mask = obs[agent_rows], mask[agent_rows]
            agent_types, agent_args, agent_log_prob = agent.act(agent_obs, agent_mask)
            types[agent_rows], args[agent_rows] = agent_types, agent_args
            decisions += int(agent_rows.sum())
            if record:
                chunks["obs"].append(agent_obs)
                chunks["mask"].append(agent_mask)
                chunks["types"].append(agent_types)
                chunks["args"].append(agent_args)
                chunks["log_prob"].append(agent_log_prob)
                chunks["game"].append(np.flatnonzero(agent_rows))

        done, winner, final_vp, final_turns = env.step(types, args)

        # Only a slot's FIRST game counts. If every finished game counted, slots whose games
        # happen to be short would finish (and restart, and finish again) more often and so
        # be over-represented, which biases both the win rate and the game length.
        newly = done & ~finished
        won[newly] = winner[newly] == seat[newly]
        capped[newly] = winner[newly] == -1  # the turn cap ended it: nobody won
        vp[newly] = final_vp[newly, seat[newly]]  # step reports every seat's VP: take the agent's
        turns[newly] = final_turns[newly]
        finished |= newly

    if record and chunks["types"]:
        moves = {k: np.concatenate(v) for k, v in chunks.items()}
        moves["types"] = moves["types"].astype(np.int64, copy=False)
        moves["args"] = moves["args"].astype(np.int64, copy=False)
        moves["log_prob"] = moves["log_prob"].astype(np.float32, copy=False)
        moves["game"] = moves["game"].astype(np.int64, copy=False)
    else:
        moves = {
            "obs": np.empty((0, ce.OBS_SIZE), np.uint8),
            "mask": np.empty((0, ce.MASK_SIZE), np.uint8),
            "types": np.empty(0, np.int64), "args": np.empty(0, np.int64),
            "log_prob": np.empty(0, np.float32), "game": np.empty(0, np.int64)}

    return Episodes(**moves, seat=seat, won=won, capped=capped, vp=vp, turns=turns,
                    decisions=decisions, seconds=time.perf_counter() - start)
