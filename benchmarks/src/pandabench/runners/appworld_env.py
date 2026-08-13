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
- ``POST /evaluate {task_id, report}`` -> ``{output: {success, difficulty,
  num_tests, passes, failures}}`` (``TestTracker.to_dict(stats_only=False)``), where
  each ``passes`` entry is ``{requirement, label}`` and each ``failures`` entry is
  ``{requirement, trace, label}``. We keep the failing ``requirement`` texts and drop
  ``trace``: a full stack trace plus source context, too large for every record.
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
from typing import Any, BinaryIO, Protocol

import httpx

logger = logging.getLogger("pandabench.appworld")

_MAX_ERROR_BODY_CHARS = 4096

# Bounds on the persisted failing-test identities. Measured over all 5,208
# requirement strings in the AppWorld task data: mean 92 chars, p95 197, max 545.
_MAX_FAILURES = 12
_MAX_FAILURE_CHARS = 240

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
    #: The ``requirement`` text of each failed test, bounded. Which tests failed
    #: separates a systematic agent behavior a rule could fix from an environment
    #: artifact; the counts alone cannot.
    failures: tuple[str, ...] = ()
    failures_truncated: bool = False


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

    def __init__(
        self,
        base_url: str,
        *,
        appworld_root: Path,
        timeout: float = 180.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"), timeout=timeout, transport=transport
        )
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
        # raising when the task is incomplete or a test errors. Retry on 5xx: the
        # single-world server can transiently 500 under long runs (evaluate is
        # read-only scoring, so retrying is safe — unlike execute).
        body = {"task_id": task_id, "suppress_errors": True, "report": False}
        out: Any = {}
        for attempt in range(3):
            try:
                out = self._post("/evaluate", body)
                break
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code >= 500 and attempt < 2:
                    logger.warning(
                        "appworld /evaluate %s for %s (attempt %d/3); retrying",
                        exc.response.status_code, task_id, attempt + 1,
                    )
                    time.sleep(2.0 * (attempt + 1))
                    continue
                raise
        data = out if isinstance(out, dict) else {}
        passes = data.get("passes") or []
        failures, truncated = _failure_requirements(data.get("failures"))
        return EvalResult(
            success=bool(data.get("success", False)),
            num_tests=int(data.get("num_tests", 0) or 0),
            num_passes=len(passes) if isinstance(passes, list) else 0,
            difficulty=int(data.get("difficulty", 0) or 0),
            raw=data,
            failures=failures,
            failures_truncated=truncated,
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
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError:
            response_body = resp.text[:_MAX_ERROR_BODY_CHARS]
            logger.error(
                "appworld HTTP failure endpoint=%s status=%d response=%r",
                path,
                resp.status_code,
                response_body,
            )
            raise
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
            success=False, num_tests=2, num_passes=1, difficulty=1, raw={"mock": True},
            failures=("[mock] assert answers match.",),
        )

    def close(self, task_id: str) -> None:
        pass


def _failure_requirements(failures: Any) -> tuple[tuple[str, ...], bool]:
    """Extract the failing tests' ``requirement`` texts, bounded.

    Bounded here rather than at the call site so no caller can persist an
    unbounded payload. The sibling ``trace`` field is deliberately not read.
    """

    if not isinstance(failures, list):
        return (), False
    truncated = len(failures) > _MAX_FAILURES
    kept: list[str] = []
    for failure in failures[:_MAX_FAILURES]:
        if not isinstance(failure, dict):
            continue
        text = " ".join(str(failure.get("requirement") or "").split())
        if not text:
            continue
        if len(text) > _MAX_FAILURE_CHARS:
            text = text[: _MAX_FAILURE_CHARS - 1].rstrip() + "…"
            truncated = True
        kept.append(text)
    return tuple(kept), truncated


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
      * ``PANDABENCH_APPWORLD_LOG``    — append-only combined server output log
      * ``APPWORLD_ROOT``              — isolated data root (holds data/datasets/*.txt)
    """

    def __init__(self) -> None:
        self._external_url = os.environ.get("PANDABENCH_APPWORLD_URL")
        self.url = self._external_url
        root_env = os.environ.get("APPWORLD_ROOT")
        self.root = Path(root_env) if root_env else None
        self._python = os.environ.get("PANDABENCH_APPWORLD_PYTHON")
        self._proc: subprocess.Popen[bytes] | None = None
        self._port = int(os.environ.get("PANDABENCH_APPWORLD_PORT", "9000"))
        configured_log = os.environ.get("PANDABENCH_APPWORLD_LOG")
        self.log_path = (
            Path(configured_log)
            if configured_log
            else (self.root / "logs" / "pandabench-appworld-server.log" if self.root else None)
        )
        self._log_handle: BinaryIO | None = None

    @property
    def owns_process(self) -> bool:
        return self._external_url is None

    def start(self) -> str:
        if self._external_url:
            logger.info("using existing AppWorld server at %s", self._external_url)
            return self._external_url
        if not self._python or not self.root:
            raise RuntimeError(
                "AppWorld server not configured: set PANDABENCH_APPWORLD_URL, or both "
                "PANDABENCH_APPWORLD_PYTHON (isolated appworld venv) and APPWORLD_ROOT. "
                "Run `make setup` to provision the isolated env."
            )
        appworld_bin = str(Path(self._python).with_name("appworld"))
        env = {**os.environ, "APPWORLD_ROOT": str(self.root)}
        assert self.log_path is not None
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("ab")
        logger.info(
            "launching AppWorld server on port %d (root=%s, log=%s)",
            self._port,
            self.root,
            self.log_path,
        )
        # `--root` is required: the CLI's default ('.') otherwise overrides
        # $APPWORLD_ROOT, so the server can't find ./data.
        try:
            self._proc = subprocess.Popen(
                [appworld_bin, "serve", "environment", "--port", str(self._port),
                 "--root", str(self.root), "--no-show-usage"],
                env=env, stdout=self._log_handle, stderr=subprocess.STDOUT,
            )
            self.url = f"http://127.0.0.1:{self._port}"
            self._await_health()
        except Exception:
            self.stop()
            raise
        assert self.url is not None
        return self.url

    def restart(self) -> str:
        if not self.owns_process:
            assert self._external_url is not None
            logger.warning(
                "AppWorld returned HTTP 5xx, but %s is externally managed; "
                "restart it before the next trial",
                self._external_url,
            )
            return self._external_url
        self.stop()
        return self.start()

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
        try:
            if self._proc is not None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait(timeout=10)
                self._proc = None
        finally:
            if self._log_handle is not None:
                self._log_handle.close()
                self._log_handle = None
            if self.owns_process:
                self.url = None


def make_env(*, dry_run: bool) -> tuple[AppWorldEnv, AppWorldServer | None, Path]:
    """Build the env for a run: mock for dry-run, else the HTTP server env."""

    if dry_run:
        return MockAppWorldEnv(), None, Path()
    server = AppWorldServer()
    url = server.start()
    root = server.root or Path(os.environ.get("APPWORLD_ROOT", "."))
    return HttpAppWorldEnv(url, appworld_root=root), server, root
