"""Which of stage 3's three steps (catan_rl/actor_critic.py) are written yet.

Each step's tests are marked with step(n): they are skipped while the step still raises
NotImplementedError, so the suite stays green until you write it, and then they run. The mark
also tags them stepN, so `pytest train/tests -m step2` runs just step 2's. Tests that need all
three steps (a real actor-critic Trainer) are marked all_steps.

A step counts as "not written" only if its probe raises NotImplementedError. Any other error
means the step is written but broken: its tests run, and show you the error.
"""

import numpy as np
import pytest
import torch

from catan_rl import actor_critic
from catan_rl.engine import ce

NAMES = {1: "ActorCriticNet", 2: "discounted_returns", 3: "actor_critic_loss"}
WHERE = "catan_rl/actor_critic.py"


def _step1() -> None:
    net = actor_critic.ActorCriticNet(8, 1)
    net.logits_and_value(torch.zeros((1, ce.OBS_SIZE), dtype=torch.uint8))


def _step2() -> None:
    actor_critic.discounted_returns(np.zeros(1, np.float32), np.zeros(1, np.int64), 1, 0.9)


def _step3() -> None:
    log_prob, entropy, value, returns = (torch.zeros(1) for _ in range(4))
    actor_critic.actor_critic_loss(log_prob, entropy, value, returns, 0.5, 0.0)


def _written(probe) -> bool:
    # The probes run once, at import. fork_rng: building a net here must not move torch's
    # global generator, or importing this module would change what other tests draw.
    with torch.random.fork_rng(devices=[]):
        try:
            probe()
        except NotImplementedError:
            return False
        except Exception:  # written, but broken: let its tests run and show the error
            return True
    return True


WRITTEN = {1: _written(_step1), 2: _written(_step2), 3: _written(_step3)}


def step(n: int):
    """Tags a test stepN (conftest.py registers the tags) and skips it unless stage 3, step n is
    written."""
    skip = pytest.mark.skipif(
        not WRITTEN[n], reason=f"stage 3, step {n} ({NAMES[n]}) is not written yet: {WHERE}")
    tag = getattr(pytest.mark, f"step{n}")

    def mark(test):
        return tag(skip(test))
    return mark


_missing = [n for n in NAMES if not WRITTEN[n]]

# Skip unless steps 1-3 are all written.
all_steps = pytest.mark.skipif(
    bool(_missing),
    reason=(f"needs stage 3, steps 1-3; not written yet: "
            f"{', '.join(f'step {n} ({NAMES[n]})' for n in _missing)}: {WHERE}"))
