"""PandaBench — A/B benchmark study for the PandaProbe Harness.

A self-contained suite that runs public agent benchmarks (AppWorld,
Terminal-Bench via Harbor, tau2-bench) in two arms — baseline (no harness) and
harness (full self-heal loop) — over a single LiteLLM provider layer, and emits
paper-ready per-trial records, pass@1/pass^k metrics, and harness telemetry.
"""

__version__ = "0.1.0"
