"""Train the policy against bots.

    .venv/bin/python train/train.py reinforce --run runs/reinforce               # stage 2
    .venv/bin/python train/train.py actor-critic --run runs/ac                   # stage 3
    .venv/bin/python train/train.py actor-critic --run runs/ac --updates 400     # resume, go on

Rerunning with the same --run resumes from its checkpoint, with the run's own settings: flags
you leave out keep them, and a flag that changes one (other than --updates) is an error. Every
Config field has a flag (--games-per-update, --lr, --reward win|vp, --entropy-coef, ...); see
`train/train.py ALGORITHM --help` and catan_rl/trainer.py.
"""

import argparse
import dataclasses
import sys

from catan_rl import actor_critic, reinforce, trainer

ALGORITHMS: dict[str, type[trainer.Trainer]] = {
    "reinforce": reinforce.Trainer,
    "actor-critic": actor_critic.Trainer,
}
CHOICES = {"reward": trainer.REWARDS, "baseline": reinforce.BASELINES,
           "opponent": trainer.OPPONENTS}
HELP = {
    "updates": "total updates to train for (raise it and rerun to go on)",
    "games_per_update": "games played (and learned from) per update",
    "lr": "Adam learning rate",
    "reward": "what each game is worth: 1 for a win, or final VP / 10 capped at 1",
    "entropy_coef": "weight of the entropy bonus subtracted from the loss",
    "max_grad_norm": "clip the gradient's total norm to this before each step",
    "minibatch": "moves per forward/backward chunk (memory, not learning)",
    "hidden": "network width",
    "layers": "network depth",
    "opponent": "the three bots the agent trains and is evaluated against",
    "eval_every": "updates between evaluations",
    "eval_games": "games per evaluation",
    "seed": "network init and every random generator",
    "train_first_seed": "training games use board seeds from here up",
    "baseline": "subtract the batch's mean reward from every move's reward, or nothing",
    "gamma": "discount per agent move (1: every move's return is its game's reward)",
    "value_coef": "weight of the critic's loss",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    algorithms = parser.add_subparsers(dest="algorithm", required=True, metavar="ALGORITHM",
                                       help=" or ".join(ALGORITHMS))
    for name, trainer_type in ALGORITHMS.items():
        about = sys.modules[trainer_type.__module__].__doc__  # "Stage N: ...", then the details
        sub = algorithms.add_parser(name, help=about.splitlines()[0], description=about,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
        sub.add_argument("--run", required=True, help="run directory (checkpoint.pt, "
                         "metrics.jsonl); if it holds a checkpoint, training resumes from it")
        sub.add_argument("--device", default="cpu", help="torch device (default %(default)s)")
        # One flag per Config field, typed from the dataclass. SUPPRESS leaves a flag that
        # wasn't given out of the parsed args, so main() can tell "not given" from "given the
        # default".
        for field in dataclasses.fields(trainer_type.config_type):
            sub.add_argument(
                "--" + field.name.replace("_", "-"), dest=field.name, type=field.type,
                default=argparse.SUPPRESS, choices=CHOICES.get(field.name),
                help=f"{HELP[field.name]} (default {field.default})")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    trainer_type = ALGORITHMS[args.algorithm]
    fields = dataclasses.fields(trainer_type.config_type)
    given = {f.name: getattr(args, f.name) for f in fields if hasattr(args, f.name)}
    try:
        # Resuming starts from the run's own settings, so only the flags given here can differ
        # (and the Trainer rejects any difference but --updates). A new run starts from defaults.
        saved = trainer_type.saved_config(args.run)
        config = dataclasses.replace(saved or trainer_type.config_type(), **given)
        run = trainer_type(config, args.run, args.device)
    except ValueError as e:  # a bad setting, or a run directory these settings don't fit
        print(f"error: {e}", file=sys.stderr)
        return 2
    run.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
