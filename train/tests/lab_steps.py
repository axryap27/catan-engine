"""Which lab steps (the RL pieces you write, from stage 3 on) are written yet.

Each step's tests are marked with STAGEn.step(k): they are skipped while the step still raises
NotImplementedError, so the suite stays green until you write it, and then they run. The mark
also tags them stagen_stepk (conftest.py registers the tags), so `pytest train/tests -m
stage4_step2` runs just that step's tests. Tests that need all of a stage's steps, and those of
the stages it builds on, are marked STAGEn.all_steps.

A step counts as not written only while its probe raises NotImplementedError. Any other error
means it is written but broken: its tests run and show you the error.
"""

from typing import Callable

import numpy as np
import pytest
import torch

from catan_rl import actor_critic, gae
from catan_rl.engine import ce


def _not_written(probe: Callable[[], object]) -> str | None:
    """The NotImplementedError's message if the probe raises one (say "stage 4, step 1:
    td_errors": a step that calls an unwritten one names that one), else None."""
    # The probes run once, at import. fork_rng: building a net here must not move torch's
    # global generator, or importing this module would change what other tests draw.
    with torch.random.fork_rng(devices=[]):
        try:
            probe()
        except NotImplementedError as e:
            return str(e)
        except Exception:  # written, but broken: let its tests run and show the error
            return None
    return None


class Stage:
    """One stage's steps: {k: (name, probe)}, written in `where`. A stage that builds on
    another (`needs`) runs its all_steps tests only once that one's steps are written too."""

    def __init__(self, number: int, where: str, steps: dict[int, tuple[str, Callable]],
                 needs: "Stage | None" = None):
        self.number = number
        self.where = where
        self.names = {k: name for k, (name, _) in steps.items()}
        self.needs = needs
        self.missing = {k: msg for k, (_, probe) in steps.items()
                        if (msg := _not_written(probe)) is not None}

    def tag(self, k: int) -> str:
        return f"stage{self.number}_step{k}"

    def step(self, k: int):
        """Tags a test stagen_stepk and skips it unless step k is written."""
        this = f"stage {self.number}, step {k}"
        missing = self.missing.get(k, "")
        why = "not written yet" if missing.startswith(this) else f"waiting for {missing}"
        skip = pytest.mark.skipif(
            k in self.missing, reason=f"{this} ({self.names[k]}): {why} ({self.where})")
        tag = getattr(pytest.mark, self.tag(k))

        def mark(test):
            return tag(skip(test))
        return mark

    def _all_missing(self) -> list[str]:
        earlier = self.needs._all_missing() if self.needs else []
        return earlier + [f"stage {self.number}, step {k} ({self.names[k]})"
                          for k in sorted(self.missing)]

    @property
    def all_steps(self):
        """Skips a test unless every step of this stage, and of the stages it needs, is
        written."""
        missing = self._all_missing()
        return pytest.mark.skipif(bool(missing),
                                  reason=f"needs every step through stage {self.number}; not "
                                         f"written yet: {', '.join(missing)}")


# ---- the probes: each calls one step on the smallest input it takes -----------------------------

def _stage3_step1() -> None:
    net = actor_critic.ActorCriticNet(8, 1)
    net.logits_and_value(torch.zeros((1, ce.OBS_SIZE), dtype=torch.uint8))


def _stage3_step2() -> None:
    actor_critic.discounted_returns(np.zeros(1, np.float32), np.zeros(1, np.int64), 1, 0.9)


def _stage3_step3() -> None:
    log_prob, entropy, value, returns = (torch.zeros(1) for _ in range(4))
    actor_critic.actor_critic_loss(log_prob, entropy, value, returns, 0.5, 0.0)


def _stage4_step1() -> None:
    gae.td_errors(np.zeros(1, np.float32), np.zeros(1, np.float32), np.zeros(1, np.int64), 1,
                  0.9)


def _stage4_step2() -> None:
    gae.gae_advantages(np.zeros(1, np.float32), np.zeros(1, np.float32), np.zeros(1, np.int64),
                       1, 0.9, 0.95)


STAGE3 = Stage(3, "catan_rl/actor_critic.py", {
    1: ("ActorCriticNet", _stage3_step1),
    2: ("discounted_returns", _stage3_step2),
    3: ("actor_critic_loss", _stage3_step3),
})
STAGE4 = Stage(4, "catan_rl/gae.py", {
    1: ("td_errors", _stage4_step1),
    2: ("gae_advantages", _stage4_step2),
}, needs=STAGE3)
STAGES = (STAGE3, STAGE4)
