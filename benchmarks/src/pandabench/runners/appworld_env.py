"""AppWorld environment access — over HTTP, out of process.

AppWorld pins ``pydantic<2``, irreconcilable with the study's LiteLLM
(``pydantic>=2.10``). So AppWorld runs as its own *environment server* in an
isolated venv (``appworld serve environment``) and we drive it over the REST
API it exposes for exactly this reason — pandabench never imports ``appworld``.

Endpoints used (verified against ``appworld==0.1.3.post1``):
- ``POST /initialize {task_id, experiment_name}`` -> ``{output: {instruction,
  supervisor, datetime}}``
- ``GET  /api_docs`` -> per-app API documentation
- ``POST /execute {task_id, code}`` -> ``{output: <stdout|traceback>}``
- ``POST /evaluate {task_id, report}`` -> ``{output: {success, num_tests,
  passes, failures, difficulty}}``
- ``POST /close {task_id}``

Task ids come from ``{APPWORLD_ROOT}/data/datasets/<split>.txt``.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx

logger = logging.getLogger("pandabench.appworld")

__all__ = [
    "AppWorldEnv",
    "AppWorldServer",
    "EvalResult",
    "HttpAppWorldEnv",
    "MockAppWorldEnv",
    "TaskInfo",
    "make_env",
]


@dataclass(frozen=True, slots=True)
class TaskInfo:
    task_id: str
    instruction: str
    supervisor: dict[str, Any]
    datetime: str | None


@dataclass(frozen=True, slots=True)
class EvalResult:
    success: bool
    num_tests: int
    num_passes: int
    difficulty: int
    raw: dict[str, Any]


class AppWorldEnv(Protocol):
    """A driveable AppWorld environment (real HTTP server or mock)."""

    def list_task_ids(self, dataset: str) -> list[str]: ...
    def initialize(self, task_id: str, *, experiment_name: str) -> TaskInfo: ...
    def api_docs(self) -> str: ...
    def execute(self, task_id: str, code: str) -> str: ...
    def evaluate(self, task_id: str) -> EvalResult: ...
    def close(self, task_id: str) -> None: ...


class HttpAppWorldEnv:
    """Drives the AppWorld environment server over HTTP (no appworld import)."""

    def __init__(self, base_url: str, *, appworld_root: Path, timeout: float = 180.0) -> None:
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)
        self._root = appworld_root
        self._api_docs_cache: str | None = None

    def list_task_ids(self, dataset: str) -> list[str]:
        path = self._root / "data" / "datasets" / f"{dataset}.txt"
        if not path.exists():
            raise FileNotFoundError(f"AppWorld dataset file missing: {path}")
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]

    def initialize(self, task_id: str, *, experiment_name: str) -> TaskInfo:
        out = self._post("/initialize", {"task_id": task_id, "experiment_name": experiment_name})
        return TaskInfo(
            task_id=task_id,
            instruction=str(out.get("instruction", "")),
            supervisor=dict(out.get("supervisor", {}) or {}),
            datetime=out.get("datetime"),
        )

    def api_docs(self) -> str:
        if self._api_docs_cache is None:
            resp = self._client.get("/api_docs")
            resp.raise_for_status()
            self._api_docs_cache = _summarize_api_docs(resp.json())
        return self._api_docs_cache

    def execute(self, task_id: str, code: str) -> str:
        out = self._post("/execute", {"task_id": task_id, "code": code})
        return out if isinstance(out, str) else str(out)

    def evaluate(self, task_id: str) -> EvalResult:
        # suppress_errors=True returns the test tracker (with failures) instead of
        # raising when the task is incomplete or a test errors.
        out = self._post(
            "/evaluate", {"task_id": task_id, "suppress_errors": True, "report": False}
        )
        data = out if isinstance(out, dict) else {}
        passes = data.get("passes") or []
        return EvalResult(
            success=bool(data.get("success", False)),
            num_tests=int(data.get("num_tests", 0) or 0),
            num_passes=len(passes) if isinstance(passes, list) else 0,
            difficulty=int(data.get("difficulty", 0) or 0),
            raw=data,
        )

    def close(self, task_id: str) -> None:
        try:
            self._post("/close", {"task_id": task_id})
        except Exception as exc:  # noqa: BLE001 - close is best-effort
            logger.debug("appworld close failed for %s: %s", task_id, exc)

    def aclose(self) -> None:
        self._client.close()

    def _post(self, path: str, body: dict[str, Any]) -> Any:
        resp = self._client.post(path, json=body)
        resp.raise_for_status()
        payload = resp.json()
        # The server wraps returns as {"output": ...}.
        return payload.get("output", payload) if isinstance(payload, dict) else payload


class MockAppWorldEnv:
    """Deterministic, dependency-free AppWorld for ``--dry-run`` and tests."""

    def __init__(self, *, tasks: int = 4) -> None:
        self._ids = [f"mock_{i}" for i in range(1, tasks + 1)]

    def list_task_ids(self, dataset: str) -> list[str]:
        return list(self._ids)

    def initialize(self, task_id: str, *, experiment_name: str) -> TaskInfo:
        return TaskInfo(
            task_id=task_id,
            instruction=f"[mock] complete task {task_id}",
            supervisor={"first_name": "Mock", "last_name": "User"},
            datetime="2026-01-01 00:00:00",
        )

    def api_docs(self) -> str:
        return "[mock] apis.supervisor.complete_task(status='success')"

    def execute(self, task_id: str, code: str) -> str:
        return "[mock] executed"

    def evaluate(self, task_id: str) -> EvalResult:
        return EvalResult(
            success=False, num_tests=2, num_passes=1, difficulty=1, raw={"mock": True}
        )

    def close(self, task_id: str) -> None:
        pass


def _summarize_api_docs(docs: Any) -> str:
    """Condense the (large) api_docs payload into an app/api listing for the prompt.

    The agent can fetch full per-API detail at run time via
    ``apis.api_docs.show_api_doc(app_name=..., api_name=...)``.
    """

    if not isinstance(docs, dict):
        return str(docs)[:4000]
    lines: list[str] = []
    for app_name, apis in docs.items():
        if isinstance(apis, list):
            names = [a.get("api_name", "?") for a in apis if isinstance(a, dict)]
            lines.append(f"- {app_name}: {', '.join(names[:40])}")
        else:
            lines.append(f"- {app_name}")
    return "\n".join(lines)


class AppWorldServer:
    """Launches/stops the isolated AppWorld environment server as a subprocess.

    Requires an isolated venv with appworld installed (pydantic v1) and the data
    downloaded. Configure via env:
      * ``PANDABENCH_APPWORLD_URL``    — use an already-running server (skip launch)
      * ``PANDABENCH_APPWORLD_PYTHON`` — path to the isolated venv's python/appworld
      * ``APPWORLD_ROOT``              — isolated data root (holds data/datasets/*.txt)
    """

    def __init__(self) -> None:
        self.url = os.environ.get("PANDABENCH_APPWORLD_URL")
        root_env = os.environ.get("APPWORLD_ROOT")
        self.root = Path(root_env) if root_env else None
        self._python = os.environ.get("PANDABENCH_APPWORLD_PYTHON")
        self._proc: subprocess.Popen[bytes] | None = None
        self._port = int(os.environ.get("PANDABENCH_APPWORLD_PORT", "9000"))

    def start(self) -> str:
        if self.url:
            logger.info("using existing AppWorld server at %s", self.url)
            return self.url
        if not self._python or not self.root:
            raise RuntimeError(
                "AppWorld server not configured: set PANDABENCH_APPWORLD_URL, or both "
                "PANDABENCH_APPWORLD_PYTHON (isolated appworld venv) and APPWORLD_ROOT. "
                "Run `make setup` to provision the isolated env."
            )
        appworld_bin = str(Path(self._python).with_name("appworld"))
        env = {**os.environ, "APPWORLD_ROOT": str(self.root)}
        logger.info("launching AppWorld server on port %d (root=%s)", self._port, self.root)
        # `--root` is required: the CLI's default ('.') otherwise overrides
        # $APPWORLD_ROOT, so the server can't find ./data.
        self._proc = subprocess.Popen(
            [appworld_bin, "serve", "environment", "--port", str(self._port),
             "--root", str(self.root), "--no-show-usage"],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self.url = f"http://127.0.0.1:{self._port}"
        self._await_health()
        return self.url

    def _await_health(self, attempts: int = 60) -> None:
        assert self.url is not None
        for _ in range(attempts):
            try:
                if httpx.get(f"{self.url}/", timeout=2.0).status_code < 500:
                    return
            except Exception:  # noqa: BLE001 - server still coming up
                pass
            time.sleep(1.0)
        raise RuntimeError(f"AppWorld server did not become healthy at {self.url}")

    def stop(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None


def make_env(*, dry_run: bool) -> tuple[AppWorldEnv, AppWorldServer | None, Path]:
    """Build the env for a run: mock for dry-run, else the HTTP server env."""

    if dry_run:
        return MockAppWorldEnv(), None, Path()
    server = AppWorldServer()
    url = server.start()
    root = server.root or Path(os.environ.get("APPWORLD_ROOT", "."))
    return HttpAppWorldEnv(url, appworld_root=root), server, root
