import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

# catan_rl lives in train/, one level up.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def pytest_configure(config):
    # The tags lab_steps' STAGEn.step(k) puts on each lab step's tests: `-m stage4_step1`
    # selects them.
    from lab_steps import STAGES
    for stage in STAGES:
        for k, name in stage.names.items():
            config.addinivalue_line(
                "markers", f"{stage.tag(k)}: the tests of stage {stage.number}, step {k} ({name})")


@dataclass(frozen=True)
class Positions:
    """Real decision points from random-play games, stacked for batched calls.

    games[i] is an independent copy of the game at position i (deciding seat to act);
    obs/mask are what that game encodes for its decider; phases[i] == int(games[i].phase);
    legal[i] == games[i].legal_actions()."""

    games: list
    obs: np.ndarray        # (N, OBS_SIZE) uint8
    mask: np.ndarray       # (N, MASK_SIZE) uint8
    phases: np.ndarray     # (N,) int64
    legal: list            # N lists of (type, arg)
    seconds: float         # build time

    def __len__(self) -> int:
        return len(self.games)

    def where(self, pred) -> list[int]:
        """Indices i with pred(i) true."""
        return [i for i in range(len(self)) if pred(i)]


# Phases with roughly one decision in a few hundred: keep every one we see.
_RARE_PHASES = {6, 7, 8}  # ROAD_BUILDING, YOP_PICK, MONOPOLY_PICK
_SEEDS = range(1, 41)
_KEEP_COMMON = 0.1       # chance of keeping a common-phase position (spreads picks over games)
_CAP_PHASE = 30          # positions per phase (rare phases, and phases bucketed by phase only)
_CAP_TYPESET = {1: 12, 5: 6}  # positions per legal type set: ROLL (2 sets), MAIN (~30)


def _build_positions() -> Positions:
    from catan_rl.engine import ce

    start = time.perf_counter()
    rng = np.random.default_rng(12345)
    kept: dict[tuple, int] = {}
    games, obs, mask, phases, legal = [], [], [], [], []
    for seed in _SEEDS:
        g = ce.Game(seed)
        while not g.done:
            moves = g.legal_actions()
            phase = int(g.phase)
            if phase in (int(ce.Phase.ROLL), int(ce.Phase.MAIN)):
                # Bucket by legal type set so MAIN positions with many types (and ROLL ones
                # where a knight may be played) are represented, not just the common sets.
                bucket = (phase, frozenset(t for t, _ in moves))
                cap = _CAP_TYPESET[phase]
            else:
                bucket, cap = (phase,), _CAP_PHASE
            keep = phase in _RARE_PHASES or rng.random() < _KEEP_COMMON
            if keep and kept.get(bucket, 0) < cap:
                kept[bucket] = kept.get(bucket, 0) + 1
                games.append(g.copy())
                obs.append(g.observation())
                mask.append(g.mask())
                phases.append(phase)
                legal.append(moves)
            t, a = moves[rng.integers(len(moves))]
            g.step(t, a)
    return Positions(
        games=games,
        obs=np.stack(obs),
        mask=np.stack(mask),
        phases=np.asarray(phases, dtype=np.int64),
        legal=legal,
        seconds=time.perf_counter() - start,
    )


@pytest.fixture(scope="session")
def positions() -> Positions:
    """Every phase but TERMINAL, from 40 random-play games (see test_fixture_* in
    test_actions.py for the coverage guarantees)."""
    return _build_positions()
