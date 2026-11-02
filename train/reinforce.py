"""Train the policy with REINFORCE against bots (stage 2).

    .venv/bin/python train/reinforce.py --run runs/reinforce
    .venv/bin/python train/reinforce.py --run runs/reinforce --updates 400   # resume, go further

Rerunning with the same --run resumes from its checkpoint, with the run's own settings: flags
you leave out keep them, and a flag that changes one (other than --updates) is an error. Every
Config field has a flag (--games-per-update, --lr, --reward win|vp, --baseline mean|none,
--entropy-coef, ...); see --help and catan_rl/reinforce.py.
"""

import argparse
import dataclasses
import sys

from catan_rl.reinforce import BASELINES, OPPONENTS, REWARDS, Config, Trainer, saved_config

CHOICES = {"reward": REWARDS, "baseline": BASELINES, "opponent": OPPONENTS}
HELP = {
    "updates": "total updates to train for (raise it and rerun to go on)",
    "games_per_update": "games played (and learned from) per update",
    "lr": "Adam learning rate",
    "reward": "what each game is worth: 1 for a win, or final VP / 10 capped at 1",
    "baseline": "subtract the batch's mean reward from every move's reward, or nothing",
    "entropy_coef": "weight of the entropy bonus subtracted from the loss",
    "max_grad_norm": "clip the gradient's total norm to this before each step",
    "minibatch": "moves per forward/backward chunk (memory, not learning)",
    "hidden": "PolicyNet width",
    "layers": "PolicyNet depth",
    "opponent": "the three bots the agent trains and is evaluated against",
    "eval_every": "updates between evaluations",
    "eval_games": "games per evaluation",
    "seed": "network init and every random generator",
    "train_first_seed": "training games use board seeds from here up",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="run directory (checkpoint.pt, metrics.jsonl);"
                        " if it holds a checkpoint, training resumes from it")
    parser.add_argument("--device", default="cpu", help="torch device (default %(default)s)")
    # One flag per Config field, typed from the dataclass. SUPPRESS leaves a flag that wasn't
    # given out of the parsed args, so main() can tell "not given" from "given the default".
    for field in dataclasses.fields(Config):
        parser.add_argument(
            "--" + field.name.replace("_", "-"), dest=field.name, type=field.type,
            default=argparse.SUPPRESS, choices=CHOICES.get(field.name),
            help=f"{HELP[field.name]} (default {field.default})")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    given = {f.name: getattr(args, f.name) for f in dataclasses.fields(Config)
             if hasattr(args, f.name)}
    try:
        # Resuming starts from the run's own settings, so only the flags given here can differ
        # (and the Trainer rejects any difference but --updates). A new run starts from defaults.
        config = dataclasses.replace(saved_config(args.run) or Config(), **given)
        trainer = Trainer(config, args.run, args.device)
    except ValueError as e:  # a bad setting, or settings that differ from the run's checkpoint
        print(f"error: {e}", file=sys.stderr)
        return 2
    trainer.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
