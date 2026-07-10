"""tau2-bench runner.

Dry-run uses the generic mock runner. The real integration registers a custom
agent against tau2's orchestrator (with an LLM user simulator on a fixed cheap
model, all routed through our LiteLLM wrapper); see ``adapters/tau2_agent``. It
is filled in during the tau2 phase.
"""

from __future__ import annotations

from .base import SingleTaskRunner
from .mock import MockTaskRunner

__all__ = ["build_tau2_runner"]


def build_tau2_runner(*, dry_run: bool) -> SingleTaskRunner:
    if dry_run:
        return MockTaskRunner("tau2")
    raise NotImplementedError(
        "Real tau2 runs execute in tau2's OWN isolated venv (it pins litellm<1.82.7, "
        "conflicting with the core's 1.91) and need the repo's data/ tree. Prerequisites "
        "(see IMPLEMENTATION_NOTES.md):\n"
        "  uv venv <tau2venv> --python 3.12\n"
        "  uv pip install --python <tau2venv>/bin/python "
        "'git+https://github.com/sierra-research/tau2-bench.git@v0.2.0' \\\n"
        "      && install pandabench into <tau2venv> (litellm resolves to <1.82.7)\n"
        "  export TAU2_DATA_DIR=<clone>/data   # data is not shipped\n"
        "Then, in that venv, drive tau2.orchestrator.Orchestrator per (task x trial) with "
        "pandabench.adapters.tau2_agent:PandaBenchTau2Agent (user simulator on a fixed "
        "model), evaluate, run on_turn_end/refresh/drain, and ingest "
        "sim.reward_info.reward (success = reward~=1.0) into TrialRecords. The agent "
        "adapter is implemented in pandabench/adapters/tau2_agent.py; this orchestration "
        "was not run this session (no live creds; isolated env not provisioned)."
    )
