# Contributing to PandaProbe Harness

Thanks for your interest in improving the harness! This guide covers everything
you need to get a change from idea to merged PR.

## Development setup

Requirements: Python ≥ 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/chirpz-ai/pandaprobe-harness && cd pandaprobe-harness
make install          # uv sync
```

The everyday commands:

```bash
make lint             # ruff
make typecheck        # mypy --strict over src/
make test             # the full offline suite (unit + e2e)
make example          # run the offline self-heal demo
```

All three of `make lint`, `make typecheck`, and `make test` must pass before a
PR can merge — CI enforces the same gates.

## Project invariants

These are non-negotiable; PRs that break them will be asked to change:

1. **Zero runtime dependencies in the core.** New third-party packages go
   behind optional extras in `pyproject.toml`, never into `dependencies`.
2. **The test suite is fully offline.** No network, no real `pandaprobe` CLI.
   Model platform behavior by extending `tests/fakes/fake_cli_client.py`
   (in-process) or `tests/bin/fake_pandaprobe` (subprocess-level).
3. **Never block or raise into the host agent loop.** Blocking I/O on async
   paths goes through `asyncio.to_thread`; every failure path degrades
   gracefully (a pending score, a log line, a journal event).
4. **One platform seam.** All platform access shells out to `pandaprobe`
   through the `CliClient` interface — never the REST API directly.
5. **Workspace discipline.** All persistence uses the atomic helpers in
   `src/pandaprobe_harness/workspace/_io.py`; stores are `threading.Lock`
   guarded; JSON parsing is forgiving (skip, don't crash).
6. `mypy --strict` and `ruff` stay clean; frozen dataclasses for config and
   value types.

## Making a change

1. Fork (or branch) and make your change, following the patterns already in
   the file you're editing.
2. Add or update tests that cover it. Behavior changes need e2e or hook-level
   coverage, not just unit tests.
3. Update `CHANGELOG.md` and, when user-facing, `README.md` and the
   [documentation](https://docs.pandaprobe.com/harness/get-started/quickstart).
4. Run `make lint typecheck test` and open a PR — the template walks you
   through the checklist.

## Bugs, features, and questions

- **Bugs and feature requests:** use the
  [issue templates](https://github.com/chirpz-ai/pandaprobe-harness/issues/new/choose).
- **Questions and usage help:**
  [GitHub Discussions](https://github.com/chirpz-ai/pandaprobe-harness/discussions).

## Releases (maintainers)

Releases are driven by the version file: bump `__version__` in
`src/pandaprobe_harness/_version.py` on `main` (with a matching `CHANGELOG.md`
entry). The release workflow runs the full CI suite as a gate, tags
`v<version>`, creates a GitHub Release, and publishes to PyPI via trusted
publishing.

## License

By contributing, you agree that your contributions are licensed under the
[MIT license](LICENSE) that covers the project.
