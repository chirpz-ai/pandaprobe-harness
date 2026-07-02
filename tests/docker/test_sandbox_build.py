"""Opt-in smoke test: the sandbox Docker image builds from Dockerfile.sandbox."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[2]


@pytest.mark.skipif(
    shutil.which("docker") is None or not os.environ.get("HARNESS_DOCKER_TESTS"),
    reason="docker build smoke test: set HARNESS_DOCKER_TESTS=1 with docker available",
)
def test_sandbox_image_builds() -> None:
    proc = subprocess.run(
        [
            "docker",
            "build",
            "-f",
            "Dockerfile.sandbox",
            "-t",
            "pandaprobe-harness-sandbox:test",
            ".",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        timeout=1200,
    )
    stderr_tail = proc.stderr.decode("utf-8", errors="replace")[-2000:]
    assert proc.returncode == 0, (
        f"docker build exited {proc.returncode}; stderr tail:\n{stderr_tail}"
    )
