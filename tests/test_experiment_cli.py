from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from src.baselines import BaselineRunner


ROOT = Path(__file__).resolve().parents[1]


def test_baseline_runner_is_importable() -> None:
    assert "no_memory" in BaselineRunner.METHODS
    assert "all_memory" in BaselineRunner.METHODS


def test_run_experiment_help_imports_successfully() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "run_experiment.py"),
            "--help",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--dataset" in completed.stdout
    assert "--methods" in completed.stdout
    assert "--output" in completed.stdout
