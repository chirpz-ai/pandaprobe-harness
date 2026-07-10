"""Terminal-Bench 2.x runner (via Harbor).

Dry-run uses the generic mock runner. The real integration shells out to
``harbor run -d terminal-bench/terminal-bench-2 --agent-import-path
pandabench.adapters.harbor_agent:PandaBenchAgent ...`` (needs Docker + the
``harbor`` tool) and reads per-attempt artifacts; see ``adapters/harbor_agent``.
It is filled in during the Harbor phase.
"""

from __future__ import annotations

from .base import SingleTaskRunner
from .mock import MockTaskRunner

__all__ = ["build_terminal_runner"]


def build_terminal_runner(*, dry_run: bool) -> SingleTaskRunner:
    if dry_run:
        return MockTaskRunner("terminal_bench")
    raise NotImplementedError(
        "Real Terminal-Bench runs are driven by Harbor, not the pandabench loop. "
        "Prerequisites (see IMPLEMENTATION_NOTES.md): Docker running; `harbor` installed "
        "in an env that ALSO has pandabench + its deps (so the agent import resolves); "
        "PandaProbe + LLM creds. Then, per (model x arm x seed) and per phase:\n"
        "  harbor run -d terminal-bench@2.0 "
        "-a pandabench.adapters.harbor_agent:PandaBenchAgent -m <model> -k <k> -n 1 "
        "-o results/runs/<run_id>/raw "
        "--ak arm=<arm> --ak seed=<seed> --ak model_key=<key> --ak backend=<b> "
        "--ak capture=<true|false> --ak harness_root=<abs> "
        "--include-task-name '<learning|eval subset>'\n"
        "then ingest each attempt's raw/<job>/<task>__<id>/result.json "
        "(verifier_result.rewards, single 0/1) into TrialRecords. The agent adapter is "
        "implemented in pandabench/adapters/harbor_agent.py; this orchestration was not "
        "run this session (Docker was down)."
    )
