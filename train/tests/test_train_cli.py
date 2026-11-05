"""Tests for the train/train.py CLI, run as a subprocess, written from its docstring.

The actor-critic runs that train need stage 3's three steps (a subprocess can't be given the
stand-ins test_actor_critic.py uses), so they are skipped until those are written.
"""

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import stage3_steps as steps
from catan_rl import actor_critic, reinforce

CLI = Path(__file__).resolve().parents[1] / "train.py"
CONFIGS = {"reinforce": reinforce.Config, "actor-critic": actor_critic.Config}
TRAINERS = {"reinforce": reinforce.Trainer, "actor-critic": actor_critic.Trainer}


def cli(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, str(CLI), *args], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=300)


def flag(field: dataclasses.Field) -> str:
    return "--" + field.name.replace("_", "-")


def read_metrics(run_dir) -> list[dict]:
    text = (Path(run_dir) / "metrics.jsonl").read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def assert_one_line_error(r: subprocess.CompletedProcess, *words: str):
    assert r.returncode == 2, (r.stdout, r.stderr)
    err = r.stderr.strip()
    assert err and "\n" not in err, r.stderr
    assert "Traceback" not in r.stderr
    for word in words:
        assert word in err, (word, err)


CLI_ARGS = ["--updates", "1", "--games-per-update", "4", "--eval-games", "4", "--hidden", "16",
            "--opponent", "greedy"]
CLI_CONFIG = reinforce.Config(updates=1, games_per_update=4, eval_games=4, hidden=16,
                              opponent="greedy")


# ---- help and usage errors -----------------------------------------------------------------------

def test_help_lists_both_algorithms(tmp_path):
    r = cli("--help", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert "reinforce" in r.stdout and "actor-critic" in r.stdout


@pytest.mark.parametrize("algorithm", ["reinforce", "actor-critic"])
def test_algorithm_help_lists_every_config_flag(tmp_path, algorithm):
    r = cli(algorithm, "--help", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    for f in dataclasses.fields(CONFIGS[algorithm]):
        assert flag(f) in r.stdout, f.name
    assert "--run" in r.stdout and "--device" in r.stdout


def test_each_algorithm_has_only_its_own_flags(tmp_path):
    ac = cli("actor-critic", "--help", cwd=tmp_path).stdout
    assert "--gamma" in ac and "--value-coef" in ac and "--baseline" not in ac
    rf = cli("reinforce", "--help", cwd=tmp_path).stdout
    assert "--baseline" in rf and "--gamma" not in rf and "--value-coef" not in rf


@pytest.mark.parametrize("args", [[], ["--run", "run"], ["ppo", "--run", "run"]],
                         ids=["nothing", "no-algorithm", "unknown-algorithm"])
def test_no_known_algorithm_is_a_usage_error(tmp_path, args):
    r = cli(*args, cwd=tmp_path)
    assert r.returncode == 2, (r.stdout, r.stderr)
    assert "Traceback" not in r.stderr
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("algorithm,args", [
    ("actor-critic", ["--baseline", "mean"]),
    ("reinforce", ["--gamma", "0.9"]),
    ("actor-critic", ["--reward", "points"]),
])
def test_a_flag_the_algorithm_lacks_or_a_bad_choice_is_an_argparse_error(tmp_path, algorithm, args):
    r = cli(algorithm, "--run", str(tmp_path / "run"), *args, cwd=tmp_path)
    assert r.returncode == 2, (r.stdout, r.stderr)
    assert "Traceback" not in r.stderr and args[0] in r.stderr
    assert not (tmp_path / "run").exists()


# ---- reinforce (ported from the old train/reinforce.py tests) ------------------------------------

def test_reinforce_trains(tmp_path):
    run_dir = tmp_path / "runs" / "reinforce"
    r = cli("reinforce", "--run", str(run_dir), *CLI_ARGS, cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert (run_dir / "checkpoint.pt").is_file()
    lines = read_metrics(run_dir)
    assert [x["update"] for x in lines] == [0, 1]
    assert lines[1]["games"] == 4 and lines[1]["eval_games"] == 4
    # Every other field took the dataclass default: the checkpoint resumes under CLI_CONFIG.
    trainer = reinforce.Trainer(CLI_CONFIG, run_dir)
    assert trainer.update == 1 and trainer.config == CLI_CONFIG


def test_reinforce_config_mismatch_is_one_line_and_status_2(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    reinforce.Trainer(CLI_CONFIG, run_dir).save()
    before = (run_dir / "checkpoint.pt").read_bytes()
    r = cli("reinforce", "--run", str(run_dir), *CLI_ARGS, "--lr", "0.01", cwd=tmp_path)
    assert_one_line_error(r, "lr")
    assert (run_dir / "checkpoint.pt").read_bytes() == before


# ---- actor-critic --------------------------------------------------------------------------------

@pytest.mark.parametrize("args", [[], CLI_ARGS], ids=["no-flags", "flags"])
def test_actor_critic_on_a_reinforce_run_is_one_line_and_status_2(tmp_path, args):
    """Needs no stage 3 code: the run directory is checked before any network is built."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    reinforce.Trainer(CLI_CONFIG, run_dir).save()
    before = (run_dir / "checkpoint.pt").read_bytes()
    r = cli("actor-critic", "--run", str(run_dir), *args, cwd=tmp_path)
    assert_one_line_error(r, "reinforce", "actor-critic")
    assert (run_dir / "checkpoint.pt").read_bytes() == before
    assert not (run_dir / "metrics.jsonl").exists()


def test_an_invalid_actor_critic_setting_is_one_line_and_status_2(tmp_path):
    r = cli("actor-critic", "--run", str(tmp_path / "run"), "--gamma", "1.5", cwd=tmp_path)
    assert_one_line_error(r, "gamma")


FIRST_RUN = {
    "reinforce": ["--baseline", "none"],
    "actor-critic": ["--gamma", "0.9", "--value-coef", "0.25"],
}


@pytest.mark.parametrize("algorithm", ["reinforce",
                                       pytest.param("actor-critic", marks=steps.all_steps)])
def test_resuming_with_only_updates_keeps_the_runs_settings(tmp_path, algorithm):
    """The second run gives only --updates: every other setting, defaults or not, is the
    first run's (it would be an error otherwise, since only updates may change)."""
    run_dir = tmp_path / "run"
    first = ["--updates", "1", "--games-per-update", "2", "--eval-games", "2", "--hidden", "16",
             "--layers", "1", "--opponent", "greedy", "--reward", "vp", "--seed", "3",
             *FIRST_RUN[algorithm]]
    r = cli(algorithm, "--run", str(run_dir), *first, cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    saved = TRAINERS[algorithm].saved_config(run_dir)
    r = cli(algorithm, "--run", str(run_dir), "--updates", "2", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert TRAINERS[algorithm].saved_config(run_dir) == dataclasses.replace(saved, updates=2)
    assert saved.hidden == 16 and saved.layers == 1 and saved.seed == 3
    assert [x["update"] for x in read_metrics(run_dir)] == [0, 1, 2]


@steps.all_steps
def test_actor_critic_trains(tmp_path):
    run_dir = tmp_path / "runs" / "ac"
    r = cli("actor-critic", "--run", str(run_dir), *CLI_ARGS, "--gamma", "0.95",
            "--value-coef", "0.25", "--reward", "vp", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert "value loss" in r.stdout and "explained" in r.stdout
    lines = read_metrics(run_dir)
    assert [x["update"] for x in lines] == [0, 1]
    assert lines[1]["games"] == 4 and lines[1]["eval_games"] == 4
    assert {"return", "policy_loss", "value_loss", "value", "explained_variance"} <= set(lines[1])
    want = actor_critic.Config(updates=1, games_per_update=4, eval_games=4, hidden=16,
                               opponent="greedy", gamma=0.95, value_coef=0.25, reward="vp")
    assert actor_critic.Trainer.saved_config(run_dir) == want
